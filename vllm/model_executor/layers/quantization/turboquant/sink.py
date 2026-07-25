# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVQ-2 attention-sink side buffer: addressing, sizing and reference logic.

The first ``sink_tokens`` positions of a sequence carry outsized attention
mass, so TurboQuant keeps them at full fp16 precision in a side buffer while
the rest of the KV cache stays 3-bit. This module owns everything about that
side buffer that is *not* a GPU kernel, so the addressing scheme has exactly
one definition that both the Triton kernels and the CPU tests consume.

Addressing
----------
The side buffer is a **tag-validated direct-mapped cache keyed on the physical
KV-cache slot** (``slot = block_number * block_size + offset_in_block``), which
is precisely the value the store path already receives in ``slot_mapping`` and
the value the decode path can reconstruct from ``block_table`` + logical
position. Keying on the physical slot rather than on a request index means the
buffer needs no knowledge of request identity: it survives batch condensation,
preemption/recompute and prefix-cache sharing without any model-runner hook.

Each entry carries a ``tag`` holding the slot it belongs to. A read is only
honoured when ``tag[row] == slot``; otherwise the reader falls back to the
dequantized value. Two properties make this fail-safe:

* **Stale entries can never be read as another token's data.** A physical slot
  is owned by exactly one logical position at a time (a block is either owned
  outright or shared through prefix caching, which is content-addressed and
  therefore implies identical tokens), so a tag match implies the buffered K/V
  really is the K/V of that slot.
* **Hash collisions degrade, they do not corrupt.** When two live slots map to
  the same row only one keeps the tag; the loser fails its tag check and reads
  the quantized value, exactly as if sinks were disabled.

Sizing
------
``sink_cache_slots`` sizes the table from the concurrency limit rather than
from the context length, which is what makes sink retention cheap: the cost is
``max_num_seqs * sink_tokens`` entries, independent of how long the sequences
get. ``VLLM_TQ_SINK_OVERPROVISION`` trades memory for hit rate (a direct-mapped
table at load factor ``1/f`` retains roughly ``f * (1 - exp(-1/f))`` of live
entries).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

# Knuth multiplicative hash constant (2**32 / phi, rounded to an odd integer).
# Shared verbatim with the Triton kernels so host-side reasoning, the CPU
# reference implementation and the device code all agree bit for bit.
SINK_HASH_MULT = 2654435761
SINK_HASH_SHIFT = 16

# Tag value for "this row holds nothing". Physical slots are non-negative.
SINK_EMPTY_TAG = -1

SINK_OVERPROVISION_ENV = "VLLM_TQ_SINK_OVERPROVISION"
_DEFAULT_OVERPROVISION = 2


def sink_overprovision() -> int:
    """Table over-provisioning factor from the environment (default 2).

    Values below 1 are clamped to 1; a malformed value falls back to the
    default rather than taking the engine down at import time.
    """
    raw = os.environ.get(SINK_OVERPROVISION_ENV, "").strip()
    if not raw:
        return _DEFAULT_OVERPROVISION
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_OVERPROVISION


def sink_cache_slots(
    max_num_seqs: int,
    sink_tokens: int,
    overprovision: int | None = None,
) -> int:
    """Number of rows in the sink side buffer (a power of two, or 0 if off).

    Args:
        max_num_seqs: Scheduler concurrency limit — the number of sequences
            that can hold live sinks simultaneously.
        sink_tokens: Retained positions per sequence (0 disables sinks).
        overprovision: Table slack factor; defaults to
            ``VLLM_TQ_SINK_OVERPROVISION``.

    Returns:
        Row count, rounded up to a power of two so the kernels can mask
        instead of dividing. ``0`` when sink retention is disabled.
    """
    if sink_tokens <= 0 or max_num_seqs <= 0:
        return 0
    if overprovision is None:
        overprovision = sink_overprovision()
    live = max_num_seqs * sink_tokens
    return 1 << max(0, (live * max(1, overprovision) - 1)).bit_length()


def sink_row_for_slot(slot: int, num_slots: int) -> int:
    """Row holding the side-buffer entry for physical cache ``slot``.

    Mirrors the Triton expression exactly (int64 multiply, logical shift, mask
    against a power-of-two row count).

    Note that a sequence's own sinks occupy *consecutive* slots, and this hash
    does not keep a run of 32 of them on 32 distinct rows once the table is
    small: at ``num_slots`` 64 a run of 32 covers 17 rows and at 128 it covers
    30, because the step between adjacent slots
    (``SINK_HASH_MULT >> SINK_HASH_SHIFT``, or one more) shares factors with
    the mask. Tables of 256 rows and up are collision-free on a run of 32.
    Small tables come from small ``max_num_seqs``; raise
    ``VLLM_TQ_SINK_OVERPROVISION`` there to buy the rows back.
    """
    if num_slots <= 0:
        raise ValueError("sink side buffer is disabled (num_slots == 0)")
    return ((slot * SINK_HASH_MULT) >> SINK_HASH_SHIFT) & (num_slots - 1)


@dataclass(frozen=True)
class SinkBufferSpec:
    """Shape and cost of one layer's sink side buffer.

    The payload is a flat ``(num_slots, num_kv_heads, 2 * head_dim)`` fp16
    tensor: ``[0:head_dim]`` holds the key, ``[head_dim:2*head_dim]`` the
    value. Keys are stored in the same space the decode kernel scores in —
    Hadamard-rotated for MSE presets, raw for FP8 presets — so the sink branch
    reuses the existing ``q_rot`` without a second projection.
    """

    num_slots: int
    num_kv_heads: int
    head_dim: int
    sink_tokens: int

    @property
    def enabled(self) -> bool:
        return self.num_slots > 0 and self.sink_tokens > 0

    @property
    def kv_shape(self) -> tuple[int, int, int]:
        return (self.num_slots, self.num_kv_heads, 2 * self.head_dim)

    @property
    def tag_shape(self) -> tuple[int]:
        return (self.num_slots,)

    @property
    def stride_slot(self) -> int:
        """Element stride between rows of the payload tensor."""
        return self.num_kv_heads * 2 * self.head_dim

    @property
    def stride_head(self) -> int:
        """Element stride between KV heads within a row."""
        return 2 * self.head_dim

    @property
    def kv_bytes(self) -> int:
        return math.prod(self.kv_shape) * 2  # fp16

    @property
    def tag_bytes(self) -> int:
        return self.num_slots * 8  # int64

    @property
    def total_bytes(self) -> int:
        return self.kv_bytes + self.tag_bytes


def build_sink_spec(
    max_num_seqs: int,
    sink_tokens: int,
    num_kv_heads: int,
    head_dim: int,
    overprovision: int | None = None,
) -> SinkBufferSpec:
    """Side-buffer spec for one layer, or a disabled spec when sinks are off."""
    return SinkBufferSpec(
        num_slots=sink_cache_slots(max_num_seqs, sink_tokens, overprovision),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        sink_tokens=sink_tokens,
    )


# ---------------------------------------------------------------------------
# Reference implementations of the device-side branches.
#
# These are the executable specification the Triton kernels are written
# against; the CPU suite validates the kernels' *logic* through them. They are
# deliberately scalar and allocation-free so they stay readable next to the
# kernel code they describe.
# ---------------------------------------------------------------------------


def sink_eligible(position: int, sink_tokens: int) -> bool:
    """Store-side gate: does this logical position belong in the side buffer?"""
    return sink_tokens > 0 and 0 <= position < sink_tokens


def resolve_tag_claims(
    slots: list[int],
    num_slots: int,
) -> dict[int, int]:
    """Phase A of the store: settle which slot owns each contested row.

    The store writes tags in one kernel and payloads in a second, so that the
    kernel boundary orders them. Within phase A colliding programs race and the
    last writer wins; this helper models "last writer wins" over the launch
    order. Phase B then writes a payload only for the slot whose tag survived,
    which is what keeps tag and payload consistent under collisions.

    Returns:
        ``{row: winning_slot}``.
    """
    tags: dict[int, int] = {}
    for slot in slots:
        tags[sink_row_for_slot(slot, num_slots)] = slot
    return tags


def sink_lookup(
    slot: int,
    position: int,
    sink_tokens: int,
    num_slots: int,
    tags: dict[int, int],
) -> int | None:
    """Decode-side gate: the row to read, or ``None`` to use the dequant path.

    A read is honoured only when the position is inside the sink window *and*
    the row's tag still names this exact physical slot.
    """
    if not sink_eligible(position, sink_tokens) or num_slots <= 0:
        return None
    row = sink_row_for_slot(slot, num_slots)
    if tags.get(row, SINK_EMPTY_TAG) != slot:
        return None
    return row
