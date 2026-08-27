# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernels for TurboQuant KV store.

Two kernels:
1. _tq_fused_store_fp8: FP8 key scatter + value uniform quantization.
2. _tq_fused_store_mse: Fused binary-search bucketize + MSE index
   packing + value quantization.

The launcher `triton_turboquant_store` selects the appropriate kernel.
"""

import math

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_turboquant_decode import _use_fp8_e4b15

# ═══════════════════════════════════════════════════════════════════════
# Shared: value uniform quantization + pack + scale/zero store
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _store_quantized_value(
    Value_ptr,
    KV_cache_ptr,
    base,  # pid * D offset into Value_ptr
    slot_base,  # byte offset into KV_cache_ptr for this slot+head
    d_offs,  # tl.arange(0, BLOCK_D)
    d_mask,  # d_offs < D
    Val_midpoints_ptr,  # [N_VAL_CENTROIDS-1] float32 (used only when VALUE_NUQ)
    Outlier_idx_ptr,  # [NH, N_OUTLIERS] int (used only when N_OUTLIERS>0)
    Outlier_val_ptr,  # [NH, N_OUTLIERS] float32 (used only when N_OUTLIERS>0)
    outlier_row,  # pid: row into the outlier tensors
    D: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_VAL: tl.constexpr,
    BLOCK_GRP: tl.constexpr,
    VALUE_NUQ: tl.constexpr = 0,
    N_VAL_CENTROIDS: tl.constexpr = 8,
    N_OUTLIERS: tl.constexpr = 0,
):
    """Quantize values to VQB bits, pack, and store with scale/zero.

    Two value codecs share the same packed layout and side bytes:
      * uniform (VALUE_NUQ=0): per-vector min/max linear quantization; the
        side fp16 pair stores (scale, min).
      * non-uniform (VALUE_NUQ=1, KVQ-1): per-vector mean/std companding onto
        a Lloyd-Max N(0,1) codebook; the side fp16 pair stores (std, mean).
        Decoding maps the stored index through the value centroid table.
    """
    val_cache_offset = KPS

    if VQB == 3:
        val_vec = tl.load(Value_ptr + base + d_offs, mask=d_mask, other=0.0).to(
            tl.float32
        )
        if VALUE_NUQ:
            v_mean = tl.sum(tl.where(d_mask, val_vec, 0.0), axis=0) / D
            v_cen = tl.where(d_mask, val_vec - v_mean, 0.0)
            v_var = tl.sum(v_cen * v_cen, axis=0) / D
            v_scale = tl.sqrt(v_var)
            v_scale = tl.where(v_scale > 1e-8, v_scale, 1e-8)
            # Same fp16-storage hazard as the uniform branch below: (std, mean)
            # are the stored pair, so clamp them into the fp16 finite range
            # before companding, keeping the companding constants identical to
            # the ones written to the cache.
            v_scale = tl.minimum(v_scale, 65504.0)
            v_mean = tl.maximum(tl.minimum(v_mean, 65504.0), -65504.0)
            v_zero = v_mean
            z = (val_vec - v_mean) / v_scale
            q_vals = tl.zeros([BLOCK_D], dtype=tl.int32)
            for _m in range(N_VAL_CENTROIDS - 1):
                mid_m = tl.load(Val_midpoints_ptr + _m)
                q_vals += (z >= mid_m).to(tl.int32)
            q_vals = tl.minimum(tl.maximum(q_vals, 0), N_VAL_CENTROIDS - 1)
        else:
            val_min = tl.min(tl.where(d_mask, val_vec, float("inf")), axis=0)
            val_max = tl.max(tl.where(d_mask, val_vec, -float("inf")), axis=0)
            # The per-vector scale and zero point are stored as fp16 further
            # down. Clamp the observed range into fp16's finite limits *before*
            # deriving the scale and quantizing, so the zero point used to
            # quantize is the same one that is stored and later used to
            # reconstruct. A bf16 value outlier (an attention sink of
            # |min| > 65504 has been measured in real checkpoints) would
            # otherwise cast to -inf at store time and poison every element of
            # the vector on reconstruction.
            val_min = tl.maximum(val_min, -65504.0)
            val_max = tl.minimum(val_max, 65504.0)
            v_scale = (val_max - val_min) / 7.0
            v_scale = tl.where(v_scale > 1e-8, v_scale, 1e-8)
            v_zero = val_min
            q_vals = tl.minimum(
                tl.maximum(((val_vec - val_min) / v_scale + 0.5).to(tl.int32), 0), 7
            )

        grp_offs = tl.arange(0, BLOCK_GRP)
        grp_mask = grp_offs < (D // 8)
        q_grp = tl.reshape(q_vals, [BLOCK_GRP, 8])
        shifts_3bit = tl.arange(0, 8) * 3
        packed_24 = tl.sum(q_grp << shifts_3bit[None, :], axis=1)
        b0 = (packed_24 & 0xFF).to(tl.uint8)
        b1 = ((packed_24 >> 8) & 0xFF).to(tl.uint8)
        b2 = ((packed_24 >> 16) & 0xFF).to(tl.uint8)
        tl.store(
            KV_cache_ptr + slot_base + val_cache_offset + grp_offs * 3,
            b0,
            mask=grp_mask,
        )
        tl.store(
            KV_cache_ptr + slot_base + val_cache_offset + grp_offs * 3 + 1,
            b1,
            mask=grp_mask,
        )
        tl.store(
            KV_cache_ptr + slot_base + val_cache_offset + grp_offs * 3 + 2,
            b2,
            mask=grp_mask,
        )

        sc_offset = val_cache_offset + VAL_DATA_BYTES
        sc_f16 = v_scale.to(tl.float16)
        sc_u16 = sc_f16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + sc_offset, (sc_u16 & 0xFF).to(tl.uint8))
        tl.store(
            KV_cache_ptr + slot_base + sc_offset + 1,
            ((sc_u16 >> 8) & 0xFF).to(tl.uint8),
        )
        zr_f16 = v_zero.to(tl.float16)
        zr_u16 = zr_f16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + sc_offset + 2, (zr_u16 & 0xFF).to(tl.uint8))
        tl.store(
            KV_cache_ptr + slot_base + sc_offset + 3,
            ((zr_u16 >> 8) & 0xFF).to(tl.uint8),
        )

    else:  # VQB == 4
        val_vec = tl.load(Value_ptr + base + d_offs, mask=d_mask, other=0.0).to(
            tl.float32
        )
        val_min = tl.min(tl.where(d_mask, val_vec, float("inf")), axis=0)
        val_max = tl.max(tl.where(d_mask, val_vec, -float("inf")), axis=0)
        # See the VQB == 3 branch: clamp into the fp16 finite range before the
        # scale/zero are derived, since both are stored as fp16 below and the
        # zero point used to quantize must equal the one used to reconstruct.
        val_min = tl.maximum(val_min, -65504.0)
        val_max = tl.minimum(val_max, 65504.0)
        v_scale = (val_max - val_min) / 15.0
        v_scale = tl.where(v_scale > 1e-8, v_scale, 1e-8)

        # Quantize all D elements from register (no re-load)
        q_all = tl.minimum(
            tl.maximum(((val_vec - val_min) / v_scale + 0.5).to(tl.int32), 0), 15
        )
        # Reshape to pairs and pack two 4-bit values per byte
        q_pairs = tl.reshape(q_all, [BLOCK_D // 2, 2])
        shifts_4 = tl.arange(0, 2) * 4
        packed_val = tl.sum((q_pairs & 0xF) << shifts_4[None, :], axis=1).to(tl.uint8)
        val_offs = tl.arange(0, BLOCK_D // 2)
        val_mask = val_offs < VAL_DATA_BYTES
        tl.store(
            KV_cache_ptr + slot_base + val_cache_offset + val_offs,
            packed_val,
            mask=val_mask,
        )

        sc_offset = val_cache_offset + VAL_DATA_BYTES
        sc_f16 = v_scale.to(tl.float16)
        sc_u16 = sc_f16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + sc_offset, (sc_u16 & 0xFF).to(tl.uint8))
        tl.store(
            KV_cache_ptr + slot_base + sc_offset + 1,
            ((sc_u16 >> 8) & 0xFF).to(tl.uint8),
        )
        zr_f16 = val_min.to(tl.float16)
        zr_u16 = zr_f16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + sc_offset + 2, (zr_u16 & 0xFF).to(tl.uint8))
        tl.store(
            KV_cache_ptr + slot_base + sc_offset + 3,
            ((zr_u16 >> 8) & 0xFF).to(tl.uint8),
        )

    # ── OUTLIER SIDE-CHANNEL (KVQ-3) ──────────────────────────────────
    # Region layout after (scale, zero): [idx bytes (N_OUTLIERS) |
    # fp16 values (2*N_OUTLIERS)]. Indices/values are precomputed on host
    # (top-|v| per vector) and written verbatim here; decode scatter-gathers.
    if N_OUTLIERS > 0:
        out_off = val_cache_offset + VAL_DATA_BYTES + 4
        for _j in range(N_OUTLIERS):
            oi = tl.load(Outlier_idx_ptr + outlier_row * N_OUTLIERS + _j).to(tl.uint8)
            tl.store(KV_cache_ptr + slot_base + out_off + _j, oi)
            ov = tl.load(Outlier_val_ptr + outlier_row * N_OUTLIERS + _j)
            # Same fp16-storage hazard as the scale/zero pair: this channel
            # exists to carry the largest magnitudes in the vector, so it is the
            # most likely to leave the fp16 finite range. A saturated outlier is
            # inexact for |v| > 65504; a stored inf poisons the reconstructed
            # element and everything downstream of it.
            ov = tl.maximum(tl.minimum(ov, 65504.0), -65504.0)
            ov_u16 = ov.to(tl.float16).to(tl.uint16, bitcast=True)
            vpos = out_off + N_OUTLIERS + 2 * _j
            tl.store(KV_cache_ptr + slot_base + vpos, (ov_u16 & 0xFF).to(tl.uint8))
            tl.store(
                KV_cache_ptr + slot_base + vpos + 1, ((ov_u16 >> 8) & 0xFF).to(tl.uint8)
            )


# ═══════════════════════════════════════════════════════════════════════
# FP8 key store + value uniform quantization
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _tq_fused_store_fp8(
    Key_ptr,  # [NH, D] float16/bfloat16 — raw keys
    Value_ptr,  # [NH, D] float16/bfloat16 — raw values
    KV_cache_ptr,  # [total_bytes] uint8 (flattened view)
    Slot_mapping_ptr,  # [N] int32 — per-token slot indices
    Val_midpoints_ptr,  # [N_VAL_CENTROIDS-1] float32 (used only when VALUE_NUQ)
    Outlier_idx_ptr,  # [NH, N_OUTLIERS] int (used only when N_OUTLIERS>0)
    Outlier_val_ptr,  # [NH, N_OUTLIERS] float32 (used only when N_OUTLIERS>0)
    # Cache strides (for computing byte offsets)
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    # Dimensions
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    # TQ layout
    KPS: tl.constexpr,
    # Value quantization
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    # Packing block sizes
    BLOCK_VAL: tl.constexpr,
    BLOCK_GRP: tl.constexpr = 16,
    FP8_E4B15: tl.constexpr = 0,  # 1 = e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)
    VALUE_NUQ: tl.constexpr = 0,
    N_VAL_CENTROIDS: tl.constexpr = 8,
    N_OUTLIERS: tl.constexpr = 0,
):
    """FP8 key cast+scatter + value uniform quantization."""
    pid = tl.program_id(0)
    token_idx = pid // H
    head_idx = pid % H

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    blk = (slot // BLOCK_SIZE).to(tl.int64)
    off = (slot % BLOCK_SIZE).to(tl.int64)
    head_idx_i64 = tl.cast(head_idx, tl.int64)
    slot_base = (
        blk * stride_cache_block
        + off * stride_cache_pos
        + head_idx_i64 * stride_cache_head
    )

    base = pid * D

    # ── FP8 KEY: cast to FP8 in-kernel and store ─────────────────
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    k_vals = tl.load(Key_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    if FP8_E4B15:
        k_fp8 = k_vals.to(tl.float8e4b15)
    else:
        k_fp8 = k_vals.to(tl.float8e4nv)
    k_bytes = k_fp8.to(tl.uint8, bitcast=True)
    tl.store(KV_cache_ptr + slot_base + d_offs, k_bytes, mask=d_mask)

    # ── VALUE QUANTIZE + PACK ───────────────────────────────────────
    _store_quantized_value(
        Value_ptr,
        KV_cache_ptr,
        base,
        slot_base,
        d_offs,
        d_mask,
        Val_midpoints_ptr,
        Outlier_idx_ptr,
        Outlier_val_ptr,
        pid,
        D=D,
        KPS=KPS,
        VQB=VQB,
        VAL_DATA_BYTES=VAL_DATA_BYTES,
        BLOCK_D=BLOCK_D,
        BLOCK_VAL=BLOCK_VAL,
        BLOCK_GRP=BLOCK_GRP,
        VALUE_NUQ=VALUE_NUQ,
        N_VAL_CENTROIDS=N_VAL_CENTROIDS,
        N_OUTLIERS=N_OUTLIERS,
    )


# ═══════════════════════════════════════════════════════════════════════
# Fused MSE store: bucketize + MSE index pack + norm store + value pack
# (eliminates 4 PyTorch kernel launches per layer vs pack-only kernel)
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _tq_fused_store_mse(
    # Post-rotation inputs
    Y_ptr,  # [NH, D] float32 — rotated normalized keys (x_hat @ PiT)
    Norms_ptr,  # [NH] float32 — key vector norms (||k||)
    Value_ptr,  # [NH, D] float32 — raw values
    # Quantization tables
    Midpoints_ptr,  # [n_centroids-1] float32
    Val_midpoints_ptr,  # [N_VAL_CENTROIDS-1] float32 (used only when VALUE_NUQ)
    Outlier_idx_ptr,  # [NH, N_OUTLIERS] int (used only when N_OUTLIERS>0)
    Outlier_val_ptr,  # [NH, N_OUTLIERS] float32 (used only when N_OUTLIERS>0)
    # Cache and indexing
    KV_cache_ptr,  # [total_bytes] uint8 (flattened view)
    Slot_mapping_ptr,  # [N] int32 — per-token slot indices
    # Cache strides
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    # Dimensions
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    # TQ layout
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    # Value quantization
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    # Packing block sizes
    BLOCK_VAL: tl.constexpr,
    # MSE params
    MSE_BITS: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    BLOCK_GRP: tl.constexpr = 16,
    VALUE_NUQ: tl.constexpr = 0,
    N_VAL_CENTROIDS: tl.constexpr = 8,
    N_OUTLIERS: tl.constexpr = 0,
):
    """Fused MSE quantize + pack + store.

    Performs binary-search bucketize, MSE index packing, norm storage,
    and value quantization in one kernel.
    """
    pid = tl.program_id(0)
    token_idx = pid // H
    head_idx = pid % H

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    blk = (slot // BLOCK_SIZE).to(tl.int64)
    off = (slot % BLOCK_SIZE).to(tl.int64)
    head_idx_i64 = tl.cast(head_idx, tl.int64)
    slot_base = (
        blk * stride_cache_block
        + off * stride_cache_pos
        + head_idx_i64 * stride_cache_head
    )

    base = pid * D
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # ── 1. BINARY SEARCH BUCKETIZE ───────────────────────────────────
    # Midpoints are sorted (N_CENTROIDS-1 values); binary search finds
    # insertion point in MSE_BITS iterations vs N_CENTROIDS-1 for linear.
    y_vec = tl.load(Y_ptr + base + d_offs, mask=d_mask, other=0.0)
    lo = tl.zeros([BLOCK_D], dtype=tl.int32)
    hi = tl.full([BLOCK_D], N_CENTROIDS - 1, dtype=tl.int32)
    for _ in range(MSE_BITS):
        mid = (lo + hi) >> 1
        # Clamp to valid midpoint index [0, N_CENTROIDS-2] for load safety;
        # the search result (lo) is still correct since converged lanes
        # don't change.
        safe_mid = tl.minimum(mid, N_CENTROIDS - 2)
        mid_val = tl.load(Midpoints_ptr + safe_mid, mask=d_mask, other=0.0)
        lo = tl.where(y_vec >= mid_val, mid + 1, lo)
        hi = tl.where(y_vec >= mid_val, hi, mid)
    idx = tl.minimum(lo, N_CENTROIDS - 1)

    # ── 2. PACK MSE INDICES from register idx ─────────────────────────
    if MSE_BITS == 4:
        idx_pairs = tl.reshape(idx, [BLOCK_D // 2, 2])
        shifts_4 = tl.arange(0, 2) * 4
        packed = tl.sum((idx_pairs & 0xF) << shifts_4[None, :], axis=1).to(tl.uint8)
        mse_offs = tl.arange(0, BLOCK_D // 2)
        mse_mask = mse_offs < MSE_BYTES
        tl.store(KV_cache_ptr + slot_base + mse_offs, packed, mask=mse_mask)

    elif MSE_BITS == 3:
        grp_offs = tl.arange(0, BLOCK_GRP)
        grp_mask = grp_offs < (D // 8)
        idx_grp = tl.reshape(idx, [BLOCK_GRP, 8])
        shifts_3 = tl.arange(0, 8) * 3
        packed_24 = tl.sum((idx_grp & 0x7) << shifts_3[None, :], axis=1)
        b0 = (packed_24 & 0xFF).to(tl.uint8)
        b1 = ((packed_24 >> 8) & 0xFF).to(tl.uint8)
        b2 = ((packed_24 >> 16) & 0xFF).to(tl.uint8)
        tl.store(KV_cache_ptr + slot_base + grp_offs * 3, b0, mask=grp_mask)
        tl.store(KV_cache_ptr + slot_base + grp_offs * 3 + 1, b1, mask=grp_mask)
        tl.store(KV_cache_ptr + slot_base + grp_offs * 3 + 2, b2, mask=grp_mask)

    # ── 3. STORE vec_norm (fp16, 2 bytes) ─────────────────────────────
    norm_offset = MSE_BYTES

    vn_f16 = tl.load(Norms_ptr + pid).to(tl.float16)
    vn_u16 = vn_f16.to(tl.uint16, bitcast=True)
    tl.store(KV_cache_ptr + slot_base + norm_offset, (vn_u16 & 0xFF).to(tl.uint8))
    tl.store(
        KV_cache_ptr + slot_base + norm_offset + 1, ((vn_u16 >> 8) & 0xFF).to(tl.uint8)
    )

    # ── 4. VALUE QUANTIZE + PACK ──────────────────────────────────────
    _store_quantized_value(
        Value_ptr,
        KV_cache_ptr,
        base,
        slot_base,
        d_offs,
        d_mask,
        Val_midpoints_ptr,
        Outlier_idx_ptr,
        Outlier_val_ptr,
        pid,
        D=D,
        KPS=KPS,
        VQB=VQB,
        VAL_DATA_BYTES=VAL_DATA_BYTES,
        BLOCK_D=BLOCK_D,
        BLOCK_VAL=BLOCK_VAL,
        BLOCK_GRP=BLOCK_GRP,
        VALUE_NUQ=VALUE_NUQ,
        N_VAL_CENTROIDS=N_VAL_CENTROIDS,
        N_OUTLIERS=N_OUTLIERS,
    )


# ═══════════════════════════════════════════════════════════════════════
# Launcher
# ═══════════════════════════════════════════════════════════════════════


def triton_turboquant_store(
    key: torch.Tensor,  # [N, H, D] — raw keys (post-RoPE)
    value: torch.Tensor,  # [N, H, D] — raw values
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, padded_slot] uint8
    slot_mapping: torch.Tensor,  # [N] int32
    PiT: torch.Tensor,  # [D, D] float32
    midpoints: torch.Tensor,  # [n_centroids-1] float32
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_fp8: bool = False,
    value_nuq: bool = False,
    val_midpoints: torch.Tensor | None = None,  # [n_val_centroids-1] float32
    value_outliers: int = 0,  # KVQ-3: per-vector exact outliers (0 = disabled)
):
    """Launch TQ store kernel (FP8 or MSE path)."""
    N, H, D = key.shape
    NH = N * H
    block_size = kv_cache.shape[1]
    BLOCK_D = triton.next_power_of_2(D)
    mse_bytes = math.ceil(D * mse_bits / 8)
    n_centroids = 2**mse_bits

    val_data_bytes = math.ceil(D * value_quant_bits / 8)

    BLOCK_VAL = triton.next_power_of_2(val_data_bytes)

    # Non-uniform value codebook (KVQ-1). Midpoints are only dereferenced when
    # VALUE_NUQ=1; pass the key midpoints as a harmless placeholder otherwise so
    # the kernel arg is always a valid pointer.
    n_val_centroids = 2**value_quant_bits
    if value_nuq:
        # Loud-fail: bucketizing against the KEY midpoints (or a VQB the
        # kernel compiles as uniform) would silently write a wrong codec.
        if value_quant_bits != 3:
            raise ValueError(
                "value_nuq requires value_quant_bits == 3 "
                f"(got {value_quant_bits})"
            )
        if val_midpoints is None:
            raise ValueError("value_nuq=True but val_midpoints is None")
        val_mid = val_midpoints
    else:
        val_mid = midpoints
    value_nuq_flag = 1 if value_nuq else 0

    # KVQ-3 outlier side-channel: pick the top-|v| elements per (token, head)
    # vector on host (cheap, vectorized) and pass indices + signed fp16 values
    # for the kernel to write. Index is stored in 1 byte, so head_dim <= 256.
    n_outliers = int(value_outliers)
    if n_outliers > 0:
        assert D <= 256, "KVQ-3 outlier index is 1 byte; requires head_dim <= 256"
        v_abs = value.reshape(NH, D).abs()
        out_idx = v_abs.topk(n_outliers, dim=1).indices.to(torch.int32).contiguous()
        out_val = (
            torch.gather(value.reshape(NH, D), 1, out_idx.to(torch.int64))
            .to(torch.float32)
            .contiguous()
        )
    else:
        # Placeholders — never dereferenced when N_OUTLIERS=0.
        out_idx = slot_mapping
        out_val = val_mid

    # Cache strides (element_size=1 for uint8, so stride in bytes = stride())
    stride_block = kv_cache.stride(0)
    stride_pos = kv_cache.stride(1)
    stride_head = kv_cache.stride(2)

    block_grp = triton.next_power_of_2(D // 8) if D >= 8 else 1

    # ── FP8 PATH: in-kernel FP8 cast + scatter via fp8 kernel ──
    if key_fp8:
        k_flat = key.reshape(NH, D).contiguous()
        v_flat = value.reshape(NH, D).contiguous()

        fp8_e4b15 = _use_fp8_e4b15(key.device.index or 0)

        grid = (NH,)
        _tq_fused_store_fp8[grid](
            k_flat,
            v_flat,
            kv_cache,
            slot_mapping,
            val_mid,
            out_idx,
            out_val,
            stride_cache_block=stride_block,
            stride_cache_pos=stride_pos,
            stride_cache_head=stride_head,
            D=D,
            H=H,
            BLOCK_SIZE=block_size,
            BLOCK_D=BLOCK_D,
            KPS=key_packed_size,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            BLOCK_VAL=BLOCK_VAL,
            BLOCK_GRP=block_grp,
            FP8_E4B15=fp8_e4b15,
            VALUE_NUQ=value_nuq_flag,
            N_VAL_CENTROIDS=n_val_centroids,
            N_OUTLIERS=n_outliers,
            num_warps=4,
            num_stages=1,
        )
        return

    # ── MSE PATH: external GEMM + fused bucketize/pack kernel ──
    # Normalize + rotation GEMM externally (cuBLAS is faster than in-kernel)
    k_flat = key.float().reshape(NH, D)
    norms = k_flat.norm(dim=1, keepdim=True)
    x_hat = k_flat / (norms + 1e-8)
    y = x_hat @ PiT

    v_flat = value.float().reshape(NH, D)

    # Fused kernel: bucketize + MSE index pack + norm store + value pack
    grid = (NH,)
    _tq_fused_store_mse[grid](
        y,
        norms.squeeze(1),
        v_flat,
        midpoints,
        val_mid,
        out_idx,
        out_val,
        kv_cache,
        slot_mapping,
        stride_cache_block=stride_block,
        stride_cache_pos=stride_pos,
        stride_cache_head=stride_head,
        D=D,
        H=H,
        BLOCK_SIZE=block_size,
        BLOCK_D=BLOCK_D,
        MSE_BYTES=mse_bytes,
        KPS=key_packed_size,
        VQB=value_quant_bits,
        VAL_DATA_BYTES=val_data_bytes,
        BLOCK_VAL=BLOCK_VAL,
        MSE_BITS=mse_bits,
        N_CENTROIDS=n_centroids,
        BLOCK_GRP=block_grp,
        VALUE_NUQ=value_nuq_flag,
        N_VAL_CENTROIDS=n_val_centroids,
        N_OUTLIERS=n_outliers,
        num_warps=4,
        num_stages=1,
    )
