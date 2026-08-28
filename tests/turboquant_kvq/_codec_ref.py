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


# Largest finite float16. The store kernels saturate every quantity that is
# written to the cache as fp16 at this magnitude (per-vector scale/zero, the
# NUQ (std, mean) pair, the KVQ-3 outlier values, the MSE key norm), because an
# unclamped cast produces inf and one inf poisons the whole reconstructed
# vector. The references below reproduce those clamps so kernel and reference
# still agree once inputs run above the fp16 range.
_FP16_MAX = 65504.0


def _to_fp16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).to(torch.float32)


def _fp16_saturate(x: torch.Tensor) -> torch.Tensor:
    """Clamp to the fp16 finite range, then round-trip through fp16.

    Mirrors the kernels' ``tl.maximum(tl.minimum(x, 65504.0), -65504.0)``
    ahead of an fp16 store, which saturates instead of producing inf.
    """
    return _to_fp16(x.clamp(-_FP16_MAX, _FP16_MAX))


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
    # The kernel saturates the observed range into the fp16 finite range
    # *before* deriving the scale, so that the zero point used to quantize is
    # the one that is stored and read back. Mirror that here, otherwise this
    # reference and the kernel diverge for any vector with |v| > 65504.
    vmin = vmin.clamp(min=-_FP16_MAX)
    vmax = vmax.clamp(max=_FP16_MAX)
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
    # (std, mean) is the stored fp16 pair; the kernel saturates both before
    # companding so the constants used to compand equal the stored ones. This
    # keeps the *stored pair* finite — it does not bound the reconstruction,
    # which is centroid * std + mean with |centroid| up to 2.150 (see the NUQ
    # branch comment in triton_turboquant_store.py).
    std = std.clamp(max=_FP16_MAX)
    mean = mean.clamp(-_FP16_MAX, _FP16_MAX)
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
    # The decode kernels saturate the NUQ reconstruction into the fp16 finite
    # range at the read (centroid * std + mean is not bounded by the stored
    # pair: |centroid| reaches 2.150, so an in-range vector can reconstruct
    # past 65504). Mirror that so reference and kernel agree there too.
    return (cent[idx] * std_h + mean_h).clamp(-_FP16_MAX, _FP16_MAX)


# ---------------------------------------------------------------------------
# KVQ-2: attention-sink fp16 retention
# ---------------------------------------------------------------------------


def fp16_retain(v: torch.Tensor) -> torch.Tensor:
    """Lossless-within-fp16 retention: round-trip through float16.

    The sink side buffer stores raw fp16 K/V, so decode reconstruction of a
    retained position is exactly the fp16 cast of the original — including the
    inf that |v| > 65504 produces. This models the KVQ-2 sink path, which has
    no saturating store of its own. The KVQ-3 outlier channel is a different
    producer: its kernel clamps, so ``outlier_value_codec`` below uses
    ``_fp16_saturate`` instead.
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
    overwrite the top-|v| elements with their fp16 values. The kernel clamps
    each outlier into the fp16 finite range before the store (this channel
    carries the largest magnitudes in the vector, so it is the one most likely
    to leave that range), hence ``_fp16_saturate`` rather than ``fp16_retain``.
    """
    out = nuqv_value_codec(v, bits)
    idx, val = outlier_select(v, n)
    rows = torch.arange(v.shape[0], device=v.device).unsqueeze(-1)
    out[rows, idx] = _fp16_saturate(val)
    return out


def sink_key_codec(k: torch.Tensor, sink_mask: torch.Tensor) -> torch.Tensor:
    """Key reconstruction for sink positions: fp16-retained where masked.

    Non-sink key reconstruction is backend-specific (MSE centroids); this
    helper only models the sink override, which is fp16-exact.
    """
    out = k.clone()
    out[sink_mask] = fp16_retain(k[sink_mask])
    return out
