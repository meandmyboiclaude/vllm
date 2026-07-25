# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-torch CPU reference codecs mirroring the TurboQuant value kernels.

These reproduce the exact arithmetic of the Triton store/decode kernels
(``triton_turboquant_store.py`` / ``triton_turboquant_decode.py``) so the
quantization math can be validated without a GPU. Storage details that affect
numerics are reproduced faithfully: side params (scale/zero) are round-tripped
through float16, and encoding uses the fp32 scale while decoding uses the
stored fp16 scale, matching the kernels.
"""

import torch

from _tqload import get_value_codebook


def _to_fp16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).to(torch.float32)


def uniform_value_codec(v: torch.Tensor, bits: int = 3) -> torch.Tensor:
    """Uniform per-vector min/max quantization (existing value path).

    Mirrors ``_store_quantized_value`` (VQB==3) + the decode dequant.

    Args:
        v: (..., D) float32 value vectors.
        bits: quantization bits (levels = 2**bits).

    Returns:
        Dequantized reconstruction, same shape as ``v``.
    """
    levels = 2**bits - 1
    vmin = v.min(dim=-1, keepdim=True).values
    vmax = v.max(dim=-1, keepdim=True).values
    scale = (vmax - vmin) / levels
    scale = torch.where(scale > 1e-8, scale, torch.full_like(scale, 1e-8))
    q = torch.clamp(((v - vmin) / scale + 0.5).to(torch.int32), 0, levels)
    scale_h = _to_fp16(scale)
    zero_h = _to_fp16(vmin)
    return q.to(torch.float32) * scale_h + zero_h


def nuqv_encode(
    v: torch.Tensor, bits: int = 3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Non-uniform (Lloyd-Max) value encoder (KVQ-1).

    Per-vector mean/std companding onto an N(0,1) Lloyd-Max codebook.

    Returns:
        idx: (..., D) int64 codebook indices.
        std: (..., 1) float32 per-vector scale (stored fp16 as "scale").
        mean: (..., 1) float32 per-vector offset (stored fp16 as "zero").
    """
    _, mid = get_value_codebook(bits)
    mid = mid.to(v.dtype)
    mean = v.mean(dim=-1, keepdim=True)
    var = ((v - mean) ** 2).mean(dim=-1, keepdim=True)
    std = torch.sqrt(var)
    std = torch.where(std > 1e-8, std, torch.full_like(std, 1e-8))
    z = (v - mean) / std
    idx = (z.unsqueeze(-1) >= mid).sum(dim=-1)  # sum(z >= midpoint_m)
    return idx, std, mean


def nuqv_value_codec(v: torch.Tensor, bits: int = 3) -> torch.Tensor:
    """Full non-uniform value round-trip (KVQ-1). Mirrors kernel nuqv path."""
    cent, _ = get_value_codebook(bits)
    cent = cent.to(torch.float32)
    idx, std, mean = nuqv_encode(v, bits)
    std_h = _to_fp16(std)
    mean_h = _to_fp16(mean)
    return cent[idx] * std_h + mean_h


# ---------------------------------------------------------------------------
# KVQ-2: attention-sink fp16 retention
# ---------------------------------------------------------------------------


def fp16_retain(v: torch.Tensor) -> torch.Tensor:
    """Lossless-within-fp16 retention: round-trip through float16.

    The sink side buffer stores raw fp16 K/V, so decode reconstruction of a
    retained position is exactly the fp16 cast of the original.
    """
    return v.to(torch.float16).to(torch.float32)


def sink_value_codec(
    v: torch.Tensor, sink_mask: torch.Tensor, bits: int = 3
) -> torch.Tensor:
    """Value reconstruction with sink retention (KVQ-2).

    Positions where ``sink_mask`` is True are read back from the fp16 side
    buffer (bit-exact fp16); all other positions use the KVQ-1 nuqv codec.

    Args:
        v: (N, D) float32 value vectors.
        sink_mask: (N,) bool — True for retained sink positions.
        bits: value bits for the quantized (non-sink) path.
    """
    out = nuqv_value_codec(v, bits)
    out[sink_mask] = fp16_retain(v[sink_mask])
    return out


def outlier_select(v: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Host-side top-|v| outlier selection (KVQ-3), matching the store launcher.

    Returns:
        idx: (N, n) int64 indices of the largest-magnitude elements.
        val: (N, n) float32 signed values at those indices.
    """
    idx = v.abs().topk(n, dim=-1).indices
    val = torch.gather(v, -1, idx)
    return idx, val


def outlier_value_codec(v: torch.Tensor, bits: int = 3, n: int = 3) -> torch.Tensor:
    """Non-uniform value codec with exact outlier side-channel (KVQ-3).

    Mirrors the kernel: quantize the whole vector with the nuqv codec, then
    overwrite the top-|v| elements with their exact fp16 values.
    """
    out = nuqv_value_codec(v, bits)
    idx, val = outlier_select(v, n)
    rows = torch.arange(v.shape[0], device=v.device).unsqueeze(-1)
    out[rows, idx] = fp16_retain(val)
    return out


def sink_key_codec(k: torch.Tensor, sink_mask: torch.Tensor) -> torch.Tensor:
    """Key reconstruction for sink positions: fp16-retained where masked.

    Non-sink key reconstruction is backend-specific (MSE centroids); this
    helper only models the sink override, which is fp16-exact.
    """
    out = k.clone()
    out[sink_mask] = fp16_retain(k[sink_mask])
    return out
