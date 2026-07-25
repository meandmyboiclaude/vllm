# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVQ-2: attention-sink fp16 retention — CPU codec + accounting tests.

Run:
  tests/turboquant_kvq$ ~/shared/needfit/lens-venv/bin/python test_kvq2_sink.py

Validates that sink positions round-trip bit-exactly at fp16 precision, that
the ``turboquant_3bit_nuqv_sink32`` preset composes with KVQ-1 without changing
the per-slot layout (sinks live in a per-sequence side buffer), and that the
side-buffer accounting is correct.
"""

import torch

from _codec_ref import (
    fp16_retain,
    nuqv_value_codec,
    sink_key_codec,
    sink_value_codec,
)
from _tqload import TQ_PRESETS, TurboQuantConfig

HEAD_DIM = 256
NUM_KV_HEADS = 4
SINK = 32


def _mse(a, b):
    return ((a - b) ** 2).mean().item()


def test_preset_composes_nuqv_and_sink():
    cfg = TurboQuantConfig.from_cache_dtype("turboquant_3bit_nuqv_sink32", HEAD_DIM)
    assert cfg.value_nuq is True
    assert cfg.sink_tokens == 32
    assert cfg.sink_enabled is True


def test_sink_does_not_change_slot_size():
    sink = TurboQuantConfig.from_cache_dtype("turboquant_3bit_nuqv_sink32", HEAD_DIM)
    nuqv = TurboQuantConfig.from_cache_dtype("turboquant_3bit_nuqv", HEAD_DIM)
    # Sinks are a per-sequence side buffer, not part of the per-slot page.
    assert sink.slot_size == nuqv.slot_size
    assert sink.slot_size_aligned == nuqv.slot_size_aligned


def test_sink_side_bytes_accounting():
    cfg = TurboQuantConfig.from_cache_dtype("turboquant_3bit_nuqv_sink32", HEAD_DIM)
    # per token per head: fp16 K (D*2) + fp16 V (D*2) = 4*D bytes
    assert cfg.sink_kv_bytes_per_token == 4 * HEAD_DIM
    expected = SINK * NUM_KV_HEADS * 4 * HEAD_DIM
    assert cfg.sink_side_bytes_per_seq(NUM_KV_HEADS) == expected
    # Fixed cost, independent of context length.
    assert cfg.sink_side_bytes_per_seq(NUM_KV_HEADS) == 131072


def test_disabled_sink_zero_cost():
    cfg = TurboQuantConfig.from_cache_dtype("turboquant_3bit_nuqv", HEAD_DIM)
    assert cfg.sink_enabled is False
    assert cfg.sink_side_bytes_per_seq(NUM_KV_HEADS) == 0


def test_sink_values_bit_exact_fp16():
    g = torch.Generator().manual_seed(3)
    seq = 128
    v = torch.randn(seq, HEAD_DIM, generator=g, dtype=torch.float32) * 5.0
    sink_mask = torch.zeros(seq, dtype=torch.bool)
    sink_mask[:SINK] = True
    out = sink_value_codec(v, sink_mask, bits=3)
    # Retained sink positions reconstruct exactly as their fp16 cast.
    assert torch.equal(out[:SINK], fp16_retain(v[:SINK]))
    # Non-sink positions match the pure nuqv codec (sink override untouched).
    ref = nuqv_value_codec(v, 3)
    assert torch.equal(out[SINK:], ref[SINK:])


def test_sink_keys_bit_exact_fp16():
    g = torch.Generator().manual_seed(4)
    seq = 64
    k = torch.randn(seq, HEAD_DIM, generator=g, dtype=torch.float32)
    sink_mask = torch.zeros(seq, dtype=torch.bool)
    sink_mask[:SINK] = True
    out = sink_key_codec(k, sink_mask)
    assert torch.equal(out[:SINK], fp16_retain(k[:SINK]))


def test_sink_reduces_error_on_retained_positions():
    # On sink positions, fp16 retention error is far below 3-bit quant error.
    g = torch.Generator().manual_seed(5)
    v = torch.randn(SINK, HEAD_DIM, generator=g, dtype=torch.float32)
    err_fp16 = _mse(fp16_retain(v), v)
    err_quant = _mse(nuqv_value_codec(v, 3), v)
    assert err_fp16 < err_quant * 1e-3


def test_other_presets_have_no_sink():
    for name in ("turboquant_3bit_nc", "turboquant_3bit_nuqv",
                 "turboquant_4bit_nc", "turboquant_k8v4"):
        cfg = TurboQuantConfig.from_cache_dtype(name, HEAD_DIM)
        assert cfg.sink_tokens == 0
        assert cfg.sink_enabled is False


def test_preset_registered():
    assert "turboquant_3bit_nuqv_sink32" in TQ_PRESETS
    assert TQ_PRESETS["turboquant_3bit_nuqv_sink32"]["sink_tokens"] == 32


if __name__ == "__main__":
    from _run import run_module

    raise SystemExit(1 if run_module(globals()) else 0)
