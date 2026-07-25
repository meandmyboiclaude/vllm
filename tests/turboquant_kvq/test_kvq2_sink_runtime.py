# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVQ-2 runtime side buffer: allocation, store, gather — CPU tests.

Run:
  tests/turboquant_kvq$ ~/shared/needfit/lens-venv/bin/python \
      test_kvq2_sink_runtime.py

Covers the three pieces of the runtime path that can be checked without a GPU:
the buffer's sizing/addressing, the two-phase store (including what happens
when slots collide), and the gather selection rule (position inside the sink
window and a matching tag → full precision, otherwise dequantized). Also pins
the invariants the GPU code depends on: that the device kernels and this host
reference share one hash definition, that a sink-free config is unchanged, and
that the downstream P101/P40 patches still fit the call shapes they pin.
"""

import os

import torch

from _codec_ref import fp16_retain, nuqv_value_codec
from _run import expect_raises
from _sink_ref import (
    SinkTable,
    gather_scores,
    gather_values,
    hadamard,
    rotate_key,
)
from _tqload import (
    BACKEND_SRC,
    KERNEL_DECODE_SRC,
    KERNEL_STORE_SRC,
    SINK_EMPTY_TAG,
    SINK_HASH_MULT,
    SINK_HASH_SHIFT,
    SINK_OVERPROVISION_ENV,
    TurboQuantConfig,
    build_sink_spec,
    resolve_tag_claims,
    sink_cache_slots,
    sink_eligible,
    sink_lookup,
    sink_row_for_slot,
)

HEAD_DIM = 128
NUM_KV_HEADS = 4
SINK = 32
BLOCK_SIZE = 16


# ---------------------------------------------------------------------------
# 1. Buffer allocation
# ---------------------------------------------------------------------------


def test_slots_zero_when_sink_disabled():
    assert sink_cache_slots(max_num_seqs=64, sink_tokens=0) == 0
    assert sink_cache_slots(max_num_seqs=0, sink_tokens=32) == 0


def test_slots_cover_live_sinks_and_are_power_of_two():
    for max_seqs in (1, 7, 32, 256):
        for over in (1, 2, 4):
            n = sink_cache_slots(max_seqs, SINK, overprovision=over)
            assert n >= max_seqs * SINK * over, (max_seqs, over, n)
            assert n & (n - 1) == 0, n


def test_slots_scale_with_concurrency_not_context():
    # The whole point of sink retention: cost is fixed per sequence.
    a = sink_cache_slots(32, SINK, overprovision=2)
    b = sink_cache_slots(32, SINK, overprovision=2)
    assert a == b
    assert sink_cache_slots(64, SINK, overprovision=2) == 2 * a


def test_overprovision_env_override():
    prev = os.environ.get(SINK_OVERPROVISION_ENV)
    try:
        os.environ[SINK_OVERPROVISION_ENV] = "8"
        assert sink_cache_slots(32, SINK) == sink_cache_slots(
            32, SINK, overprovision=8
        )
        os.environ[SINK_OVERPROVISION_ENV] = "not-a-number"
        assert sink_cache_slots(32, SINK) == sink_cache_slots(
            32, SINK, overprovision=2
        )
        os.environ[SINK_OVERPROVISION_ENV] = "0"
        assert sink_cache_slots(32, SINK) == sink_cache_slots(
            32, SINK, overprovision=1
        )
    finally:
        if prev is None:
            os.environ.pop(SINK_OVERPROVISION_ENV, None)
        else:
            os.environ[SINK_OVERPROVISION_ENV] = prev


def test_spec_geometry_matches_kernel_strides():
    spec = build_sink_spec(32, SINK, NUM_KV_HEADS, HEAD_DIM, overprovision=2)
    assert spec.enabled
    assert spec.kv_shape == (spec.num_slots, NUM_KV_HEADS, 2 * HEAD_DIM)
    assert spec.tag_shape == (spec.num_slots,)
    # Strides the kernels are launched with must equal a contiguous layout.
    contiguous = torch.zeros(spec.kv_shape, dtype=torch.float16)
    assert contiguous.stride(0) == spec.stride_slot
    assert contiguous.stride(1) == spec.stride_head


def test_spec_cost_is_bounded_and_reported():
    spec = build_sink_spec(32, SINK, NUM_KV_HEADS, HEAD_DIM, overprovision=2)
    assert spec.kv_bytes == spec.num_slots * NUM_KV_HEADS * 2 * HEAD_DIM * 2
    assert spec.tag_bytes == spec.num_slots * 8
    assert spec.total_bytes == spec.kv_bytes + spec.tag_bytes
    # 2048 rows x 4 heads x 256 fp16 = 4 MiB payload for this geometry.
    assert spec.num_slots == 2048
    assert spec.kv_bytes == 4 * 1024 * 1024


def test_spec_disabled_allocates_nothing():
    spec = build_sink_spec(32, 0, NUM_KV_HEADS, HEAD_DIM)
    assert not spec.enabled
    assert spec.num_slots == 0
    assert spec.total_bytes == 0


def test_spec_matches_shipped_accounting():
    # The v2 accounting promised sink_tokens * heads * 4 * D bytes per
    # sequence; the runtime table must not exceed that times the slack.
    cfg = TurboQuantConfig.from_cache_dtype(
        "turboquant_3bit_nuqv_sink32", HEAD_DIM
    )
    per_seq = cfg.sink_side_bytes_per_seq(NUM_KV_HEADS)
    spec = build_sink_spec(32, cfg.sink_tokens, NUM_KV_HEADS, HEAD_DIM,
                           overprovision=1)
    assert spec.kv_bytes == 32 * per_seq


# ---------------------------------------------------------------------------
# 2. Addressing
# ---------------------------------------------------------------------------


def test_row_is_in_range_and_deterministic():
    n = 2048
    for slot in range(0, 100000, 37):
        row = sink_row_for_slot(slot, n)
        assert 0 <= row < n
        assert row == sink_row_for_slot(slot, n)


def test_row_matches_kernel_expression():
    n = 4096
    for slot in (0, 1, 15, 16, 1023, 99991):
        expected = ((slot * SINK_HASH_MULT) >> SINK_HASH_SHIFT) & (n - 1)
        assert sink_row_for_slot(slot, n) == expected


def test_consecutive_slots_self_collide_on_small_tables():
    # A sequence's sink positions are consecutive slots inside a block, so the
    # hash has to spread a run of SINK consecutive slots. It does — but only
    # once the table is big enough. Below 256 rows the run folds onto itself
    # and a sequence evicts its own sinks; those sizes are exactly what small
    # max_num_seqs deployments get, so the loss is pinned here rather than
    # assumed away.
    worst = {64: 18, 128: 31}  # upper bound on distinct rows for a run of 32
    for n, cap in worst.items():
        for base in (0, 1000, 65536):
            rows = {sink_row_for_slot(base + i, n) for i in range(SINK)}
            assert len(rows) <= cap, (n, base, len(rows))
            assert len(rows) < SINK, (n, base)
    for n in (256, 1024, 2048):
        for base in (0, 1000, 65536):
            rows = {sink_row_for_slot(base + i, n) for i in range(SINK)}
            assert len(rows) == SINK, (n, base, len(rows))


def test_table_size_for_small_concurrency_is_a_colliding_one():
    # Ties the row counts above to the configs that produce them, so a change
    # to sink_cache_slots that moves a deployment into the colliding regime
    # shows up here.
    assert sink_cache_slots(max_num_seqs=1, sink_tokens=SINK, overprovision=2) == 64
    assert sink_cache_slots(max_num_seqs=2, sink_tokens=SINK, overprovision=2) == 128
    assert sink_cache_slots(max_num_seqs=4, sink_tokens=SINK, overprovision=2) == 256
    # Over-provisioning buys the run back at small concurrency.
    assert sink_cache_slots(max_num_seqs=2, sink_tokens=SINK, overprovision=8) == 512
    rows = {sink_row_for_slot(1000 + i, 512) for i in range(SINK)}
    assert len(rows) == SINK


def test_row_requires_enabled_table():
    expect_raises(ValueError, sink_row_for_slot, 5, 0)


def test_slot_arithmetic_matches_store_and_decode():
    # Store decomposes slot -> (block, offset); decode recomposes it from the
    # block table. The sink tag is only valid if both agree.
    for block, off in ((0, 0), (3, 15), (129, 7)):
        slot = block * BLOCK_SIZE + off
        assert slot // BLOCK_SIZE == block
        assert slot % BLOCK_SIZE == off
        assert (slot // BLOCK_SIZE) * BLOCK_SIZE + (slot % BLOCK_SIZE) == slot


# ---------------------------------------------------------------------------
# 3. Store path
# ---------------------------------------------------------------------------


def test_store_gate_is_the_sink_window():
    assert sink_eligible(0, SINK)
    assert sink_eligible(31, SINK)
    assert not sink_eligible(32, SINK)
    assert not sink_eligible(-1, SINK)
    assert not sink_eligible(0, 0)


def test_store_writes_only_sink_positions():
    g = torch.Generator().manual_seed(11)
    seq = 64
    slots = list(range(500, 500 + seq))
    positions = list(range(seq))
    k = torch.randn(seq, HEAD_DIM, generator=g)
    v = torch.randn(seq, HEAD_DIM, generator=g)
    table = SinkTable(2048, HEAD_DIM)
    table.store(slots, positions, k, v, SINK)

    claimed = [t for t in table.tags if t != SINK_EMPTY_TAG]
    assert len(claimed) == SINK
    assert sorted(claimed) == slots[:SINK]
    for p in range(SINK, seq):
        assert table.lookup(slots[p], p, SINK) is None


def test_store_skips_negative_slots():
    table = SinkTable(1024, HEAD_DIM)
    k = torch.randn(4, HEAD_DIM)
    table.store([-1, -1, 10, 11], [0, 1, 2, 3], k, k, SINK)
    assert sorted(t for t in table.tags if t != SINK_EMPTY_TAG) == [10, 11]


def test_store_is_incremental_across_chunks():
    # Chunked prefill writes positions 0..7 first, then 8..31.
    g = torch.Generator().manual_seed(12)
    k = torch.randn(SINK, HEAD_DIM, generator=g)
    v = torch.randn(SINK, HEAD_DIM, generator=g)
    slots = [900 + i for i in range(SINK)]
    table = SinkTable(2048, HEAD_DIM)
    table.store(slots[:8], list(range(8)), k[:8], v[:8], SINK)
    assert sum(t != SINK_EMPTY_TAG for t in table.tags) == 8
    table.store(slots[8:], list(range(8, SINK)), k[8:], v[8:], SINK)
    assert sum(t != SINK_EMPTY_TAG for t in table.tags) == SINK
    for i, (slot, pos) in enumerate(zip(slots, range(SINK))):
        row = table.lookup(slot, pos, SINK)
        assert row is not None
        assert torch.equal(
            table.kv[row, HEAD_DIM:].to(torch.float32), fp16_retain(v[i])
        )


def test_collision_never_leaves_tag_and_payload_disagreeing():
    # The failure mode the two-phase store exists to prevent: a row tagged
    # with one slot but holding another slot's K/V.
    n = 16  # deliberately tiny so collisions are guaranteed
    seq = 64
    g = torch.Generator().manual_seed(13)
    slots = [7 * i + 3 for i in range(seq)]
    positions = [i % SINK for i in range(seq)]
    k = torch.randn(seq, HEAD_DIM, generator=g)
    v = torch.randn(seq, HEAD_DIM, generator=g)
    table = SinkTable(n, HEAD_DIM)
    table.store(slots, positions, k, v, SINK)

    for row, tag in enumerate(table.tags):
        if tag == SINK_EMPTY_TAG:
            continue
        i = slots.index(tag)
        assert torch.equal(
            table.kv[row, HEAD_DIM:].to(torch.float32), fp16_retain(v[i])
        )
        assert torch.equal(
            table.kv[row, :HEAD_DIM].to(torch.float32), fp16_retain(k[i])
        )


def test_collision_loser_falls_back_not_corrupts():
    n = 16
    seq = 64
    g = torch.Generator().manual_seed(14)
    slots = [7 * i + 3 for i in range(seq)]
    positions = [i % SINK for i in range(seq)]
    v = torch.randn(seq, HEAD_DIM, generator=g)
    table = SinkTable(n, HEAD_DIM)
    table.store(slots, positions, v, v, SINK)

    hits = sum(
        table.lookup(s, p, SINK) is not None for s, p in zip(slots, positions)
    )
    assert 0 < hits < seq  # some lost the row — that is the degradation
    out = gather_values(v, slots, positions, table, SINK)
    ref = nuqv_value_codec(v, 3)
    for i, (s, p) in enumerate(zip(slots, positions)):
        if table.lookup(s, p, SINK) is None:
            assert torch.equal(out[i], ref[i])  # exactly the no-sink result
        else:
            assert torch.equal(out[i], fp16_retain(v[i]))


def test_tag_claim_is_last_writer_wins():
    n = 8
    slots = [3, 3 + n * 1, 3 + n * 2]  # may or may not collide; check the model
    tags = resolve_tag_claims(slots, n)
    for row, tag in tags.items():
        # The winner must be the last slot in launch order mapping to that row.
        winners = [s for s in slots if sink_row_for_slot(s, n) == row]
        assert tag == winners[-1]


def test_stale_row_is_rejected_by_tag():
    # Block reuse: a row keeps data from a slot that has since been freed.
    table = SinkTable(1024, HEAD_DIM)
    v = torch.randn(1, HEAD_DIM)
    table.store([4242], [0], v, v, SINK)
    assert table.lookup(4242, 0, SINK) is not None
    # A different sequence asks for a different slot that happens to hash to
    # the same row: the tag check refuses it.
    row = sink_row_for_slot(4242, table.num_slots)
    other = next(
        s for s in range(100000) if s != 4242
        and sink_row_for_slot(s, table.num_slots) == row
    )
    assert table.lookup(other, 0, SINK) is None


# ---------------------------------------------------------------------------
# 4. Gather selection
# ---------------------------------------------------------------------------


def test_gather_selects_full_precision_inside_window():
    g = torch.Generator().manual_seed(21)
    seq = 96
    v = torch.randn(seq, HEAD_DIM, generator=g) * 3.0
    slots = list(range(2000, 2000 + seq))
    positions = list(range(seq))
    table = SinkTable(4096, HEAD_DIM)
    table.store(slots, positions, v, v, SINK)

    out = gather_values(v, slots, positions, table, SINK)
    assert torch.equal(out[:SINK], fp16_retain(v[:SINK]))
    assert torch.equal(out[SINK:], nuqv_value_codec(v, 3)[SINK:])


def test_gather_lookup_rule():
    tags = {sink_row_for_slot(100, 256): 100}
    assert sink_lookup(100, 0, SINK, 256, tags) is not None
    assert sink_lookup(100, SINK, SINK, 256, tags) is None  # outside window
    assert sink_lookup(101, 0, SINK, 256, tags) is None  # tag mismatch
    assert sink_lookup(100, 0, 0, 256, tags) is None  # sinks disabled
    assert sink_lookup(100, 0, SINK, 0, tags) is None  # no table


def test_gather_reduces_value_error_on_sinks():
    g = torch.Generator().manual_seed(22)
    v = torch.randn(SINK, HEAD_DIM, generator=g)
    slots = list(range(300, 300 + SINK))
    positions = list(range(SINK))
    table = SinkTable(2048, HEAD_DIM)
    table.store(slots, positions, v, v, SINK)
    out = gather_values(v, slots, positions, table, SINK)
    err_sink = ((out - v) ** 2).mean().item()
    err_quant = ((nuqv_value_codec(v, 3) - v) ** 2).mean().item()
    assert err_sink < err_quant * 1e-3


def test_rotated_key_scores_exactly():
    # The design hinge: the sink key is stored Hadamard-rotated so the decode
    # kernel's existing q_rot scores it with a plain dot product. H is
    # orthonormal and symmetric, so <q@H, k@H> == <q, k>.
    g = torch.Generator().manual_seed(23)
    pit = hadamard(HEAD_DIM)
    q = torch.randn(HEAD_DIM, generator=g)
    k = torch.randn(SINK, HEAD_DIM, generator=g)
    q_rot = q @ pit
    k_rot = rotate_key(k, pit)
    stored = k_rot.to(torch.float16).to(torch.float32)
    got = stored @ q_rot
    want = k @ q
    assert torch.allclose(got, want, rtol=2e-3, atol=2e-3), (got - want).abs().max()


def test_sink_scores_beat_quantized_scores():
    g = torch.Generator().manual_seed(24)
    pit = hadamard(HEAD_DIM)
    q = torch.randn(HEAD_DIM, generator=g)
    k = torch.randn(SINK, HEAD_DIM, generator=g)
    scale = HEAD_DIM**-0.5
    q_rot = q @ pit

    exact = (k @ q) * scale
    # 3-bit MSE key reconstruction stand-in: round the rotated unit vector to
    # 8 levels, then rescale by the stored norm (what the kernel does).
    norms = k.norm(dim=-1, keepdim=True)
    y = (k / (norms + 1e-8)) @ pit
    lo, hi = y.min(), y.max()
    step = (hi - lo) / 7.0
    y_q = torch.round((y - lo) / step) * step + lo
    quant_scores = ((y_q * norms) @ q_rot) * scale

    slots = list(range(700, 700 + SINK))
    positions = list(range(SINK))
    table = SinkTable(2048, HEAD_DIM)
    table.store(slots, positions, rotate_key(k, pit), k, SINK)
    sink_scores = gather_scores(
        q_rot, k, quant_scores, slots, positions, table, SINK, scale
    )

    err_sink = (sink_scores - exact).abs().max().item()
    err_quant = (quant_scores - exact).abs().max().item()
    assert err_sink < err_quant * 1e-2, (err_sink, err_quant)


# ---------------------------------------------------------------------------
# 5. The default path must be unchanged
# ---------------------------------------------------------------------------


def test_disabled_sink_gather_is_bit_identical_to_nuqv():
    g = torch.Generator().manual_seed(31)
    seq = 80
    v = torch.randn(seq, HEAD_DIM, generator=g)
    slots = list(range(seq))
    positions = list(range(seq))
    table = SinkTable(0, HEAD_DIM)
    table.store(slots, positions, v, v, 0)
    assert all(t == SINK_EMPTY_TAG for t in table.tags)
    out = gather_values(v, slots, positions, table, 0)
    assert torch.equal(out, nuqv_value_codec(v, 3))


def test_sink_free_presets_get_no_buffer():
    for name in (
        "turboquant_3bit_nc",
        "turboquant_3bit_nuqv",
        "turboquant_3bit_nuqv_out1",
        "turboquant_4bit_nc",
        "turboquant_k8v4",
    ):
        cfg = TurboQuantConfig.from_cache_dtype(name, HEAD_DIM)
        spec = build_sink_spec(32, cfg.sink_tokens, NUM_KV_HEADS, HEAD_DIM)
        assert not spec.enabled
        assert spec.num_slots == 0


def test_sink_presets_get_a_buffer():
    for name in (
        "turboquant_3bit_nuqv_sink32",
        "turboquant_3bit_nuqv_out1_sink32",
    ):
        cfg = TurboQuantConfig.from_cache_dtype(name, HEAD_DIM)
        spec = build_sink_spec(32, cfg.sink_tokens, NUM_KV_HEADS, HEAD_DIM)
        assert spec.enabled
        assert spec.sink_tokens == 32


def test_sink_still_does_not_change_slot_size():
    sink = TurboQuantConfig.from_cache_dtype(
        "turboquant_3bit_nuqv_sink32", HEAD_DIM
    )
    base = TurboQuantConfig.from_cache_dtype("turboquant_3bit_nuqv", HEAD_DIM)
    assert sink.slot_size_aligned == base.slot_size_aligned


def test_sink_composes_with_outliers():
    # KVQ-3 patches outliers into the quantized vector; a sink hit is exact
    # and must supersede it, so the two never fight over a position.
    g = torch.Generator().manual_seed(32)
    seq = 48
    v = torch.randn(seq, HEAD_DIM, generator=g)
    slots = list(range(4000, 4000 + seq))
    positions = list(range(seq))
    table = SinkTable(2048, HEAD_DIM)
    table.store(slots, positions, v, v, SINK)
    out = gather_values(v, slots, positions, table, SINK)
    assert torch.equal(out[:SINK], fp16_retain(v[:SINK]))


# ---------------------------------------------------------------------------
# 6. Invariants the device code and downstream patches depend on
# ---------------------------------------------------------------------------


def _read(path):
    return path.read_text(encoding="utf-8")


def test_kernels_share_one_hash_definition():
    # If the device code ever inlines its own constants, host reasoning and
    # the CPU tests stop describing what the GPU does.
    for src in (KERNEL_DECODE_SRC, KERNEL_STORE_SRC):
        text = _read(src)
        assert "SINK_HASH_MULT" in text
        assert "SINK_HASH_SHIFT" in text
        assert "from vllm.model_executor.layers.quantization.turboquant.sink" in text
        assert str(SINK_HASH_MULT) not in text  # no hard-coded literal


def test_kernel_hash_expression_matches_host():
    # Same shape as sink_row_for_slot: multiply, logical shift, mask.
    hashed = "* SINK_HASH_MULT) >> SINK_HASH_SHIFT) & ("
    decode = _read(KERNEL_DECODE_SRC)
    store = _read(KERNEL_STORE_SRC)
    assert decode.count(hashed) == 2  # stage1 + full dequant
    assert store.count(hashed) == 2  # claim + write
    # Slot ids must be widened before the multiply or the hash wraps at 32 bit.
    assert "page_off.to(tl.int64)" in decode
    assert ".to(tl.int64)" in store


def test_sink_branch_is_constexpr_gated():
    # SINK_TOKENS == 0 must compile the branch out, which is what keeps a
    # sink-free config byte-identical on the GPU.
    text = _read(KERNEL_DECODE_SRC)
    assert "SINK_TOKENS: tl.constexpr = 0" in text
    assert text.count("if SINK_TOKENS > 0:") >= 4
    store = _read(KERNEL_STORE_SRC)
    assert "sink_active = (" in store


def test_boot_warning_is_gone():
    text = _read(BACKEND_SRC)
    assert "runtime side-buffer path is not yet active" not in text
    assert "KVQ-2 sink retention active" in text


def test_decode_launcher_signature_is_unchanged_for_p40():
    # P40 rebinds triton_turboquant_decode_attention with a wrapper carrying a
    # fixed signature; a new keyword argument here is a TypeError at runtime.
    # KVQ-2 therefore passes its buffers through module state instead.
    import ast

    tree = ast.parse(_read(KERNEL_DECODE_SRC))
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "triton_turboquant_decode_attention"
    )
    names = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
    expected = [
        "query", "kv_cache", "block_table", "seq_lens", "Pi", "centroids",
        "scale", "mse_bits", "key_packed_size", "value_quant_bits", "key_fp8",
        "norm_correction", "PiT", "value_nuq", "val_centroids",
        "value_outliers", "mid_o_buf", "output_buf", "lse_buf", "buf_holder",
        "max_num_kv_splits",
    ]
    assert names == expected, names
    assert "def set_active_sink(" in _read(KERNEL_DECODE_SRC)


def test_p101_anchors_still_match_the_backend_source():
    # P101 text-patches the continuation-prefill decode call verbatim, so its
    # anchor covers the *whole* call including every keyword argument. Pinning
    # only the `if` header (as this test first did) let the KVQ-1/KVQ-3 kwargs
    # break P101 silently: its committed anchor still ends at PiT=PiT, while
    # the tree now passes value_nuq / val_centroids / value_outliers too. Both
    # anchors are reproduced in full here so any further edit to this block
    # fails CI, and so re-anchoring P101 has an exact target to copy.
    text = _read(BACKEND_SRC)
    threshold_anchor = (
        "# do_kv_cache_update already stored all tokens to TQ cache, "
        "so the decode\n"
        "# kernel can read them efficiently. This avoids O(cached_len) "
        "dequant work\n"
        "# per continuation, eliminating the O(N²/chunk_size) collapse "
        "at long context.\n"
        "_CONTINUATION_DECODE_THRESHOLD = 128\n"
    )
    assert threshold_anchor in text
    pad = " " * 16
    body = " " * 20
    arg = " " * 24
    loop_anchor = (
        f"{pad}# Continuation chunk: tokens already stored to TQ cache\n"
        f"{pad}# by do_kv_cache_update. Use decode kernel directly to\n"
        f"{pad}# avoid O(cached_len) full-dequant per continuation.\n"
        f"{pad}# For large continuations, fall back to "
        "_continuation_prefill.\n"
        f"{pad}cached_len = seq_len - q_len\n"
        f"{pad}if q_len <= _CONTINUATION_DECODE_THRESHOLD:\n"
        f"{body}# Fast path: treat each query as a decode request\n"
        f"{body}# with incremental seq_lens for causal masking.\n"
        f"{body}# Slice from pre-built arange (no kernel launch)\n"
        f"{body}synth_seq_lens = _arange_cache[cached_len + 1 : seq_len + 1]\n"
        f"{body}synth_bt = attn_metadata.block_table[i : i + 1]"
        ".expand(q_len, -1)\n"
        f"{body}out = triton_turboquant_decode_attention(\n"
        f"{arg}query=q_seq,\n"
        f"{arg}kv_cache=kv_cache,\n"
        f"{arg}block_table=synth_bt,\n"
        f"{arg}seq_lens=synth_seq_lens,\n"
        f"{arg}Pi=Pi,\n"
        f"{arg}centroids=centroids,\n"
        f"{arg}scale=self.scale,\n"
        f"{arg}mse_bits=self.tq_config.key_mse_bits,\n"
        f"{arg}key_packed_size=self.tq_config.key_packed_size,\n"
        f"{arg}value_quant_bits=(self.tq_config."
        "effective_value_quant_bits),\n"
        f"{arg}key_fp8=self.tq_config.key_fp8,\n"
        f"{arg}norm_correction=self.tq_config.norm_correction,\n"
        f"{arg}PiT=PiT,\n"
        f"{arg}value_nuq=self.tq_config.value_nuq,\n"
        f'{arg}val_centroids=getattr(layer, "_tq_val_centroids", None),\n'
        f"{arg}value_outliers=self.tq_config.n_value_outliers,\n"
        f"{body})\n"
    )
    assert loop_anchor in text
    # P101's own drift markers must stay absent (they signal it already ran).
    assert "_CONTINUATION_DECODE_MAX_CACHED_LEN" not in text
    assert "use_decode_continuation" not in text


def test_store_gate_failure_drops_the_tags():
    # The tag proves which slot claimed a row, not which sequence owns that
    # slot now. Only the store keeps the two in step, so a step that cannot
    # run the store gate has to drop every tag — otherwise a slot freed by one
    # sequence and re-issued to another is read at the old owner's K/V.
    text = _read(BACKEND_SRC)
    assert "def _invalidate_sink_tags(" in text
    assert "self._invalidate_sink_tags(layer)" in text
    assert "tags.fill_(SINK_EMPTY_TAG)" in text


def test_metadata_carries_positions_for_the_store_gate():
    text = _read(BACKEND_SRC)
    assert "token_positions: torch.Tensor | None = None" in text
    assert "token_positions=cam.positions" in text


if __name__ == "__main__":
    from _run import run_module

    raise SystemExit(1 if run_module(globals()) else 0)
