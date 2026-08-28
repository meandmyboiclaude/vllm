# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp16 side-channel range handling in the TurboQuant store kernels.

Regression coverage for gh-53334 observation 2 and the gh-54085 follow-ups.
Every per-vector quantity the store writes beside the packed data is stored as
fp16: the value (scale, zero) pair, the KVQ-3 outlier values, and — on the MSE
key path — the key norm. An unclamped cast of a magnitude above the fp16
finite range (65504) produces inf, and one inf poisons every element of the
reconstructed vector, since reconstruction is ``q * scale + zero`` for values
and ``centroid * norm`` for keys.

Covers both store copies:
  * AoS: ``vllm/v1/attention/ops/triton_turboquant_store.py``
  * SoA: ``vllm/v1/attention/ops/turboquant_soa/triton_turboquant_store.py``
    (same clamps, per-block metadata region instead of per-slot side bytes;
    it additionally folds 1/||c_t|| into the stored norm under
    ``norm_correction=True``, which can push a saturated norm back out of
    range, so the kernel clamps again after that multiply).
"""

import math

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_store import (
    triton_turboquant_store as soa_turboquant_store,
)

DEVICE_TYPE = current_platform.device_type

D = 128
H = 1
N = 1
KEY_PACKED_SIZE = D
BLOCK_SIZE = 16
NUM_BLOCKS = 1
FP16_MAX = 65504.0

MSE_BITS = 4
MSE_BYTES = D * MSE_BITS // 8
MSE_KEY_PACKED_SIZE = MSE_BYTES + 2  # packed indices + fp16 key norm


def _fp16_at(buf: np.ndarray, byte_off: int) -> np.float32:
    """Decode the little-endian fp16 stored at `byte_off` of a uint8 buffer."""
    raw = buf[byte_off : byte_off + 2].tobytes()
    return np.frombuffer(raw, dtype=np.float16)[0].astype(np.float32)


def _unpack_q(packed: np.ndarray, value_quant_bits: int) -> np.ndarray:
    """Unpack D quantization indices from the packed value bytes."""
    q = np.zeros(D, dtype=np.float32)
    if value_quant_bits == 4:
        q[0::2] = packed & 0x0F
        q[1::2] = (packed >> 4) & 0x0F
    else:  # 3-bit, 8 values packed into 3 bytes
        bits = np.unpackbits(packed, bitorder="little").reshape(-1, 24)
        for i in range(8):
            grp = bits[:, i * 3 : (i + 1) * 3]
            q[i::8] = (grp * (1 << np.arange(3))).sum(axis=1)[: D // 8]
    return q


def _value_vector(low_outlier: float, high_outlier: float | None) -> torch.Tensor:
    """A value vector whose ordinary body is small, with forced extremes.

    Element 0 carries `low_outlier`; when `high_outlier` is given, element 1
    carries it, so the vector saturates at *both* ends of the fp16 range.
    """
    value = torch.zeros(N, H, D, dtype=torch.bfloat16, device=DEVICE_TYPE)
    value[0, 0, 1:] = torch.linspace(-1.0, 4.0, D - 1, device=DEVICE_TYPE).to(
        torch.bfloat16
    )
    value[0, 0, 0] = low_outlier
    if high_outlier is not None:
        value[0, 0, 1] = high_outlier
    return value


def _store_and_reconstruct(
    low_outlier: float, value_quant_bits: int, high_outlier: float | None = None
):
    """Store one value vector through the AoS FP8-key path; reconstruct it."""
    val_data_bytes = D * value_quant_bits // 8
    slot_bytes = KEY_PACKED_SIZE + val_data_bytes + 4

    value = _value_vector(low_outlier, high_outlier)
    key = torch.zeros(N, H, D, dtype=torch.bfloat16, device=DEVICE_TYPE)

    kv_cache = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, H, slot_bytes, dtype=torch.uint8, device=DEVICE_TYPE
    )
    triton_turboquant_store(
        key,
        value,
        kv_cache,
        torch.zeros(N, dtype=torch.int32, device=DEVICE_TYPE),
        torch.eye(D, dtype=torch.float32, device=DEVICE_TYPE),
        torch.zeros(1, dtype=torch.float32, device=DEVICE_TYPE),
        mse_bits=1,
        key_packed_size=KEY_PACKED_SIZE,
        value_quant_bits=value_quant_bits,
        key_fp8=True,
    )
    torch.cuda.synchronize()

    slot = kv_cache[0, 0, 0].cpu().numpy()
    sc_off = KEY_PACKED_SIZE + val_data_bytes
    scale = _fp16_at(slot, sc_off)
    zero = _fp16_at(slot, sc_off + 2)

    packed = slot[KEY_PACKED_SIZE : KEY_PACKED_SIZE + val_data_bytes]
    q = _unpack_q(packed, value_quant_bits)

    return q * scale + zero, value[0, 0].float().cpu().numpy(), scale, zero


def _quant_step(ref: np.ndarray, value_quant_bits: int) -> float:
    """Per-vector quantization step actually used by the kernel.

    The kernel derives the step from the *saturated* range, so the bound has to
    clip both ends — a vector that saturates high as well as low has a step set
    by (65504 - -65504)/levels, not by its raw max.
    """
    levels = 2**value_quant_bits - 1
    sat = np.clip(ref, -FP16_MAX, FP16_MAX)
    return float((sat.max() - sat.min()) / levels)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("value_quant_bits", [4])
@pytest.mark.parametrize(
    "outlier",
    [
        -125000.0,  # beyond fp16 range; measured on a real 27B v-cache sink
        -70000.0,  # just beyond fp16 range
        -42000.0,  # within fp16 range (already worked before the fix)
        -1.0,  # ordinary vector
    ],
)
def test_value_outlier_reconstructs_finite(outlier, value_quant_bits):
    """No value magnitude may reconstruct as inf on the uniform value codec.

    Before the fix, |outlier| > 65504 stored a -inf zero point and every
    element of the vector came back as inf.
    """
    recon, ref, scale, zero = _store_and_reconstruct(outlier, value_quant_bits)

    assert np.isfinite(scale), f"stored scale is not finite: {scale}"
    assert np.isfinite(zero), f"stored zero point is not finite: {zero}"
    assert np.isfinite(recon).all(), (
        f"outlier {outlier} poisoned the vector: "
        f"{np.count_nonzero(~np.isfinite(recon))}/{D} elements non-finite"
    )

    # The elements that are not the outlier must stay usable. The bound is the
    # inherent per-vector quantization step for a vector with this range, which
    # is what an in-range outlier of the same magnitude already produces.
    step = _quant_step(ref, value_quant_bits)
    assert np.abs(recon[1:] - ref[1:]).max() <= step + 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("value_quant_bits", [3, 4])
@pytest.mark.parametrize(
    "low,high",
    [
        (-125000.0, 125000.0),  # both ends beyond fp16 range
        (-70000.0, 90000.0),  # both ends just beyond
        (-125000.0, 4.0),  # low end only (the single-ended case)
        (-1.0, 125000.0),  # high end only — max saturates, min does not
    ],
)
def test_value_saturating_both_ends(low, high, value_quant_bits):
    """A vector saturating at both ends still stores a finite (scale, zero).

    The zero point is clamped up from -inf and the scale is derived from the
    saturated range, so the reconstruction stays finite and the body of the
    vector stays within one (widened) quantization step. Covers the case the
    single-ended parametrization above never exercised, and the high end, which
    the earlier error bound clipped on one side only.
    """
    recon, ref, scale, zero = _store_and_reconstruct(low, value_quant_bits, high)

    assert np.isfinite(scale), f"stored scale is not finite: {scale}"
    assert np.isfinite(zero), f"stored zero point is not finite: {zero}"
    assert np.isfinite(recon).all(), (
        f"({low}, {high}) poisoned the vector: "
        f"{np.count_nonzero(~np.isfinite(recon))}/{D} elements non-finite"
    )
    assert zero >= -FP16_MAX and zero <= FP16_MAX

    step = _quant_step(ref, value_quant_bits)
    body = slice(2, D)  # elements 0 and 1 hold the forced extremes
    assert np.abs(recon[body] - ref[body]).max() <= step + 1e-3


# ---------------------------------------------------------------------------
# MSE key path: the stored per-vector key norm is fp16 too
# ---------------------------------------------------------------------------


def _mse_midpoints() -> torch.Tensor:
    n_centroids = 2**MSE_BITS
    return torch.linspace(
        -1.0, 1.0, n_centroids - 1, dtype=torch.float32, device=DEVICE_TYPE
    )


def _store_mse_and_read_norm(key_elem: float, value_quant_bits: int = 4):
    """Store one key vector on the AoS MSE path; read back the stored norm."""
    val_data_bytes = D * value_quant_bits // 8
    slot_bytes = MSE_KEY_PACKED_SIZE + val_data_bytes + 4

    key = torch.full(
        (N, H, D), key_elem, dtype=torch.bfloat16, device=DEVICE_TYPE
    )
    value = torch.zeros(N, H, D, dtype=torch.bfloat16, device=DEVICE_TYPE)
    kv_cache = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, H, slot_bytes, dtype=torch.uint8, device=DEVICE_TYPE
    )

    triton_turboquant_store(
        key,
        value,
        kv_cache,
        torch.zeros(N, dtype=torch.int32, device=DEVICE_TYPE),
        torch.eye(D, dtype=torch.float32, device=DEVICE_TYPE),
        _mse_midpoints(),
        mse_bits=MSE_BITS,
        key_packed_size=MSE_KEY_PACKED_SIZE,
        value_quant_bits=value_quant_bits,
        key_fp8=False,
    )
    torch.cuda.synchronize()

    slot = kv_cache[0, 0, 0].cpu().numpy()
    return _fp16_at(slot, MSE_BYTES), float(
        key[0, 0].float().norm().cpu()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "key_elem",
    [
        1.0e4,  # ||k|| = 1e4*sqrt(128) = 113137 — beyond fp16 range
        1.0e5,  # ||k|| = 1.13e6 — far beyond
        1.0,  # ||k|| = 11.3 — ordinary vector, must be unaffected
    ],
)
def test_mse_key_norm_stored_finite(key_elem):
    """The stored key norm must never be inf (gh-54085 follow-up 1).

    Key reconstruction is ``centroid * stored_norm``, so an inf norm makes the
    whole key vector non-finite — the same failure the value scale/zero clamp
    fixed, on the third fp16 side channel. The launcher saturates the stored
    copy only; the rotation input keeps the true norm.
    """
    stored, true_norm = _store_mse_and_read_norm(key_elem)

    assert np.isfinite(stored), f"stored key norm is not finite: {stored}"
    if true_norm > FP16_MAX:
        assert stored == pytest.approx(FP16_MAX), (
            f"expected saturation at {FP16_MAX}, got {stored}"
        )
    else:
        assert stored == pytest.approx(true_norm, rel=1e-3)


# ---------------------------------------------------------------------------
# SoA store copy: same clamps, per-block metadata region
# ---------------------------------------------------------------------------


def _soa_store(
    key: torch.Tensor,
    value: torch.Tensor,
    value_quant_bits: int,
    key_fp8: bool,
    norm_correction: bool = False,
    centroids: torch.Tensor | None = None,
):
    """Run the SoA store and return (block bytes, layout constants)."""
    val_data_bytes = D * value_quant_bits // 8
    key_data_bytes = D if key_fp8 else MSE_BYTES
    data_bytes_per_slot = key_data_bytes + val_data_bytes
    num_soa_fields = 2 if key_fp8 else 3
    slot_bytes = data_bytes_per_slot + num_soa_fields * 2

    kv_cache = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, H, slot_bytes, dtype=torch.uint8, device=DEVICE_TYPE
    )
    soa_turboquant_store(
        key,
        value,
        kv_cache,
        torch.zeros(N, dtype=torch.int32, device=DEVICE_TYPE),
        torch.eye(D, dtype=torch.float32, device=DEVICE_TYPE),
        _mse_midpoints(),
        mse_bits=MSE_BITS,
        key_packed_size=key_data_bytes + 2,
        value_quant_bits=value_quant_bits,
        key_fp8=key_fp8,
        centroids=centroids,
        norm_correction=norm_correction,
    )
    torch.cuda.synchronize()

    block = kv_cache[0].cpu().numpy().reshape(-1)
    meta_offset = BLOCK_SIZE * H * data_bytes_per_slot
    meta = np.frombuffer(block[meta_offset:].tobytes(), dtype=np.float16)
    return block, data_bytes_per_slot, num_soa_fields, meta


def _soa_meta(
    meta: np.ndarray, num_fields: int, field: int, head: int = 0, off: int = 0
):
    """Read one SoA metadata scalar: [H, NUM_FIELDS, BLOCK_SIZE] fp16."""
    return meta[head * num_fields * BLOCK_SIZE + field * BLOCK_SIZE + off].astype(
        np.float32
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("value_quant_bits", [3, 4])
@pytest.mark.parametrize(
    "low,high",
    [(-125000.0, 125000.0), (-125000.0, 4.0), (-1.0, 4.0)],
)
def test_soa_value_scale_zero_finite(low, high, value_quant_bits):
    """SoA copy: the V (scale, zero) pair in the metadata region stays finite.

    Same clamp as the AoS copy, different destination — the pair lands in the
    per-block SoA metadata strip rather than in per-slot side bytes.
    """
    value = _value_vector(low, high)
    key = torch.zeros(N, H, D, dtype=torch.bfloat16, device=DEVICE_TYPE)

    block, _, num_fields, meta = _soa_store(
        key, value, value_quant_bits, key_fp8=True
    )
    scale = _soa_meta(meta, num_fields, 0)  # SOA_V_SCALE = 0 on the FP8 path
    zero = _soa_meta(meta, num_fields, 1)  # SOA_V_ZERO = 1

    assert np.isfinite(scale), f"SoA stored scale is not finite: {scale}"
    assert np.isfinite(zero), f"SoA stored zero point is not finite: {zero}"

    val_data_bytes = D * value_quant_bits // 8
    packed = block[D : D + val_data_bytes]
    recon = _unpack_q(packed, value_quant_bits) * scale + zero
    assert np.isfinite(recon).all()

    ref = value[0, 0].float().cpu().numpy()
    step = _quant_step(ref, value_quant_bits)
    body = slice(2, D)
    assert np.abs(recon[body] - ref[body]).max() <= step + 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("key_elem", [1.0e4, 1.0e5, 1.0])
def test_soa_mse_key_norm_stored_finite(key_elem):
    """SoA copy: the SoA K-norm array must never hold inf."""
    key = torch.full((N, H, D), key_elem, dtype=torch.bfloat16, device=DEVICE_TYPE)
    value = torch.zeros(N, H, D, dtype=torch.bfloat16, device=DEVICE_TYPE)

    _, _, num_fields, meta = _soa_store(key, value, 4, key_fp8=False)
    stored = _soa_meta(meta, num_fields, 0)  # SOA_K_NORM = 0 on the MSE path
    true_norm = float(key[0, 0].float().norm().cpu())

    assert np.isfinite(stored), f"SoA stored key norm is not finite: {stored}"
    if true_norm > FP16_MAX:
        assert stored == pytest.approx(FP16_MAX)
    else:
        assert stored == pytest.approx(true_norm, rel=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("centroid_mag", [0.01, 1.0])
def test_soa_norm_correction_key_norm_stored_finite(centroid_mag):
    """SoA copy: norm correction must not leave the stored norm past fp16.

    ``NORM_CORRECTION=1`` stores ||k_t|| / ||c_t||, so the clamp has to sit
    after the divide: with ||c_t|| < 1 the correction pushes an in-range norm
    out of range, and with ||c_t|| > 1 it brings an out-of-range norm back in
    (which a host-side clamp of ||k_t|| would have needlessly saturated).
    """
    key = torch.full((N, H, D), 1.0e4, dtype=torch.bfloat16, device=DEVICE_TYPE)
    value = torch.zeros(N, H, D, dtype=torch.bfloat16, device=DEVICE_TYPE)
    centroids = torch.full(
        (2**MSE_BITS,), centroid_mag, dtype=torch.float32, device=DEVICE_TYPE
    )

    _, _, num_fields, meta = _soa_store(
        key,
        value,
        4,
        key_fp8=False,
        norm_correction=True,
        centroids=centroids,
    )
    stored = _soa_meta(meta, num_fields, 0)

    assert np.isfinite(stored), (
        f"SoA norm-corrected key norm is not finite: {stored} "
        f"(centroid magnitude {centroid_mag})"
    )
    assert stored <= FP16_MAX

    # The correction itself must survive: the stored value is the *true*
    # ||k_t|| / ||c_vec||, saturated only if that quotient is out of range.
    c_norm = centroid_mag * math.sqrt(D)
    true_norm = float(key[0, 0].float().norm().cpu())
    expected = min(true_norm / c_norm, FP16_MAX)
    assert stored == pytest.approx(expected, rel=2e-2)
