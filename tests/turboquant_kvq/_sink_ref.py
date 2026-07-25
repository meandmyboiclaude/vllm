# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU model of the KVQ-2 sink side-buffer store and gather paths.

Reproduces, in plain torch, exactly what the two store kernels
(``_tq_sink_claim`` / ``_tq_sink_write``) and the two reader branches
(``_tq_decode_stage1`` / ``_tq_full_dequant_kv``) do:

* the store gate (logical position inside the sink window),
* the two-phase tag claim that keeps a row's tag and payload consistent when
  slots collide,
* the tag-validated read that falls back to the dequantized value.

Keys are held in the space the decode kernel scores in — Hadamard-rotated for
MSE presets — so the reference rotates them the same way the store launcher
does (``y * ||k||`` where ``y = (k / ||k||) @ PiT``).
"""

import torch

from _codec_ref import fp16_retain, nuqv_value_codec
from _tqload import SINK_EMPTY_TAG, sink_eligible, sink_row_for_slot


def hadamard(d: int) -> torch.Tensor:
    """Orthonormal symmetric Hadamard matrix, matching ``_build_hadamard``."""
    h = torch.tensor([[1.0]])
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / (d**0.5)


def rotate_key(k: torch.Tensor, pit: torch.Tensor) -> torch.Tensor:
    """Key in decode-scoring space: ``(k/||k||) @ PiT * ||k||``.

    Written as the store launcher computes it (normalize, rotate, rescale)
    rather than as ``k @ PiT``, so the reference inherits the same rounding.
    """
    norms = k.norm(dim=-1, keepdim=True)
    y = (k / (norms + 1e-8)) @ pit
    return y * norms


class SinkTable:
    """The side buffer: ``num_slots`` rows of (fp16 key, fp16 value) + tags."""

    def __init__(self, num_slots: int, head_dim: int):
        self.num_slots = num_slots
        self.head_dim = head_dim
        self.tags = [SINK_EMPTY_TAG] * num_slots
        self.kv = torch.zeros(max(num_slots, 1), 2 * head_dim, dtype=torch.float16)

    def store(
        self,
        slots: list[int],
        positions: list[int],
        keys_scoring_space: torch.Tensor,
        values: torch.Tensor,
        sink_tokens: int,
    ) -> None:
        """Two-phase store: settle tags, then write only the surviving slots."""
        if self.num_slots == 0 or sink_tokens <= 0:
            return
        eligible = [
            i
            for i, (slot, pos) in enumerate(zip(slots, positions))
            if slot >= 0 and sink_eligible(pos, sink_tokens)
        ]
        # Phase A — every eligible token claims its row; last writer wins.
        for i in eligible:
            self.tags[sink_row_for_slot(slots[i], self.num_slots)] = slots[i]
        # Kernel boundary. Phase B — only the token whose claim survived
        # writes a payload, so tag and payload can never name different slots.
        for i in eligible:
            row = sink_row_for_slot(slots[i], self.num_slots)
            if self.tags[row] != slots[i]:
                continue
            self.kv[row, : self.head_dim] = keys_scoring_space[i].to(torch.float16)
            self.kv[row, self.head_dim :] = values[i].to(torch.float16)

    def lookup(self, slot: int, position: int, sink_tokens: int) -> int | None:
        """Row to read, or None when the reader must dequantize instead."""
        if self.num_slots == 0 or not sink_eligible(position, sink_tokens):
            return None
        row = sink_row_for_slot(slot, self.num_slots)
        return row if self.tags[row] == slot else None


def gather_values(
    values: torch.Tensor,
    slots: list[int],
    positions: list[int],
    table: SinkTable,
    sink_tokens: int,
    bits: int = 3,
) -> torch.Tensor:
    """Value reconstruction the decode kernel produces, position by position.

    Sink hits read the fp16 payload; everything else takes the KVQ-1 codec,
    which is bit-for-bit what the no-sink preset returns.
    """
    out = nuqv_value_codec(values, bits)
    for i, (slot, pos) in enumerate(zip(slots, positions)):
        row = table.lookup(slot, pos, sink_tokens)
        if row is not None:
            out[i] = table.kv[row, table.head_dim :].to(torch.float32)
    return out


def gather_scores(
    q_rot: torch.Tensor,
    keys_scoring_space: torch.Tensor,
    quant_scores: torch.Tensor,
    slots: list[int],
    positions: list[int],
    table: SinkTable,
    sink_tokens: int,
    scale: float,
) -> torch.Tensor:
    """Scores the decode kernel produces: exact dot product on sink hits."""
    out = quant_scores.clone()
    for i, (slot, pos) in enumerate(zip(slots, positions)):
        row = table.lookup(slot, pos, sink_tokens)
        if row is not None:
            k_sink = table.kv[row, : table.head_dim].to(torch.float32)
            out[i] = torch.dot(q_rot, k_sink) * scale
    return out


def expected_sink_value(v_row: torch.Tensor) -> torch.Tensor:
    """What a sink hit must reconstruct: the plain fp16 cast of the input."""
    return fp16_retain(v_row)
