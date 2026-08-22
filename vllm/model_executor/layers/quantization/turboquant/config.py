# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant configuration."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = logging.getLogger(__name__)

# KVQ-4: per-layer bit allocation. A JSON object mapping GLOBAL model layer
# index (string) to a TQ preset name, e.g.
#   VLLM_TQ_LAYER_BITS='{"0":"turboquant_3bit_nc","4":"turboquant_3bit_nuqv"}'
# The key is the index parsed from the layer name (the position in
# config.layer_types for hybrids), NOT the rank among full-attention layers;
# keys that land on non-attention (e.g. GDN/Mamba) layers are silent no-ops.
# Unlisted layers fall back to the model-level --kv-cache-dtype preset. An empty
# / unset value means a uniform map (no behavior change).
TQ_LAYER_BITS_ENV = "VLLM_TQ_LAYER_BITS"

# Named TQ presets: each maps to frozen config parameters.
# key_quant_bits: 8 = FP8 keys, 3-4 = MSE (Lloyd-Max) quantized keys.
# value_quant_bits: 3-4 = uniform quantized values.
TQ_PRESETS: dict[str, dict] = {
    "turboquant_k8v4": {
        "key_quant_bits": 8,
        "value_quant_bits": 4,
        "norm_correction": False,
    },
    "turboquant_4bit_nc": {
        "key_quant_bits": 4,
        "value_quant_bits": 4,
        "norm_correction": True,
    },
    "turboquant_k3v4_nc": {
        "key_quant_bits": 3,
        "value_quant_bits": 4,
        "norm_correction": True,
    },
    "turboquant_3bit_nc": {
        "key_quant_bits": 3,
        "value_quant_bits": 3,
        "norm_correction": True,
    },
    # KVQ-1: non-uniform (Lloyd-Max) quantized values. Same 3-bit storage
    # and identical slot size as turboquant_3bit_nc, but values are encoded
    # against an analytic N(0,1) Lloyd-Max codebook (per-vector mean/std
    # companding) instead of naive uniform min/max. Value bits dominate the
    # quality loss, so this recovers PPL at zero extra storage.
    "turboquant_3bit_nuqv": {
        "key_quant_bits": 3,
        "value_quant_bits": 3,
        "norm_correction": True,
        "value_nuq": True,
    },
    # KVQ-2: attention-sink retention. Composes KVQ-1 with fp16 retention of
    # the first 32 token positions per sequence (K+V) in a small per-sequence
    # side buffer, read back at full precision during decode. Sinks carry
    # outsized attention mass; keeping them lossless is cheap (fixed 32-token
    # cost) and stabilizes long-context quality.
    "turboquant_3bit_nuqv_sink32": {
        "key_quant_bits": 3,
        "value_quant_bits": 3,
        "norm_correction": True,
        "value_nuq": True,
        "sink_tokens": 32,
    },
    # KVQ-3: per-value outlier side-channel. The top ~1% |elements| of each
    # value vector are stored exactly (index + fp16) in a small inline side
    # region and scatter-gathered back during decode. Heavy-tailed value
    # distributions lose most quality to a few large elements; keeping them
    # exact removes those errors. Composable with sink32.
    "turboquant_3bit_nuqv_out1": {
        "key_quant_bits": 3,
        "value_quant_bits": 3,
        "norm_correction": True,
        "value_nuq": True,
        "value_outlier_pct": 0.01,
    },
    "turboquant_3bit_nuqv_out1_sink32": {
        "key_quant_bits": 3,
        "value_quant_bits": 3,
        "norm_correction": True,
        "value_nuq": True,
        "value_outlier_pct": 0.01,
        "sink_tokens": 32,
    },
}


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV-cache quantization.

    Applies Hadamard rotation followed by per-coordinate Lloyd-Max scalar
    quantization for keys, and uniform quantization for values.

    Historical note: the core algorithmic pattern implemented for key
    quantization (Hadamard rotation followed by deterministic scalar
    quantization and re-normalization) was originally established in DRIVE
    (Vargaftik et al., NeurIPS 2021) and EDEN (Vargaftik et al., ICML
    2022). This formulation is also mathematically equivalent to the
    scalar case of the HIGGS quantization method (Malinovskii et al.,
    "Pushing the Limits of Large Language Model Quantization via the
    Linearity Theorem", NAACL 2025; preprint arXiv:2411.17525), which
    subsequently generalized these concepts.

    A first application of this approach to KV-cache compression is in
    "Cache Me If You Must: Adaptive Key-Value Quantization for Large
    Language Models" (Shutova et al., ICML 2025; preprint
    arXiv:2501.19392). All of these foundational and application
    references pre-date the TurboQuant paper (Zandieh et al., ICLR 2026).

    QJL is intentionally omitted: community consensus (5+ independent
    groups) found it hurts attention quality by amplifying variance
    through softmax.

    Named presets (use via --kv-cache-dtype):
        turboquant_k8v4:   FP8 keys + 4-bit values, 2.6x, +1.17% PPL
        turboquant_4bit_nc: 4-bit MSE keys + 4-bit values + NC, 3.8x, +2.71%
        turboquant_k3v4_nc: 3-bit MSE keys + 4-bit values + NC, ~3.5x, +10.63%
        turboquant_3bit_nc: 3-bit MSE keys + 3-bit values + NC, 4.9x, +20.59%
        turboquant_3bit_nuqv: 3-bit MSE keys + 3-bit NON-UNIFORM (Lloyd-Max)
            values + NC. Identical slot size to turboquant_3bit_nc; recovers
            value-side quality (value bits dominate PPL loss).
        turboquant_3bit_nuqv_sink32: KVQ-1 + fp16 retention of the first 32
            token positions per sequence (K+V) in a small side buffer.
        turboquant_3bit_nuqv_out1: KVQ-1 + top ~1% |value elements| stored
            exact (index + fp16) inline and gathered back at decode.
        turboquant_3bit_nuqv_out1_sink32: KVQ-1 + outliers + sink retention.

    Args:
        head_dim: Attention head dimension (e.g. 64, 96, 128).
        key_quant_bits: Bits for key quantization. 8 = FP8 keys (no
            rotation/MSE). 3-4 = Lloyd-Max MSE quantized keys.
        value_quant_bits: Bits per value dimension for uniform quantization.
            3 = 8 levels, 4 = 16 levels (default).
        norm_correction: Re-normalize centroid vectors to unit norm before
            inverse rotation during dequant. Fixes quantization-induced norm
            distortion, improving PPL by ~0.8% at 4-bit.
    """

    head_dim: int = 128
    key_quant_bits: int = 3  # 3-4 = MSE keys, 8 = FP8 keys
    value_quant_bits: int = 4  # 3-4 = uniform quantized values
    seed: int = 42  # kept for backward compatibility; no longer used internally
    norm_correction: bool = False
    value_nuq: bool = False  # KVQ-1: non-uniform (Lloyd-Max) value codebook
    sink_tokens: int = 0  # KVQ-2: first N positions kept fp16 in a side buffer
    value_outlier_pct: float = 0.0  # KVQ-3: fraction of value elements kept exact

    @property
    def key_fp8(self) -> bool:
        """Whether keys are stored as FP8 — no rotation/quantization needed."""
        return self.key_quant_bits == 8

    @property
    def mse_bits(self) -> int:
        """MSE quantizer bit-width (determines centroid count: 2^mse_bits).

        For MSE key modes, equals key_quant_bits.
        For FP8 key mode, falls back to value_quant_bits (centroids are still
        needed for continuation-prefill dequant and decode kernel params).
        """
        if self.key_fp8:
            return self.value_quant_bits
        return self.key_quant_bits

    @property
    def key_mse_bits(self) -> int:
        """MSE bits actually used for key quantization (0 if FP8 keys)."""
        if self.key_fp8:
            return 0
        return self.key_quant_bits

    @property
    def centroid_bits(self) -> int:
        """Bits for centroid generation — always non-zero."""
        return self.mse_bits

    @property
    def n_centroids(self) -> int:
        return 2**self.mse_bits

    @property
    def n_value_centroids(self) -> int:
        """Codebook size for non-uniform value quantization (2^value_bits)."""
        return 2**self.value_quant_bits

    @property
    def n_value_outliers(self) -> int:
        """Number of per-vector value elements kept exact (KVQ-3).

        ``round(head_dim * pct)`` with a floor of 1 when outliers are enabled,
        so a non-zero percentage always retains at least one element.
        """
        if self.value_outlier_pct <= 0.0:
            return 0
        return max(1, round(self.head_dim * self.value_outlier_pct))

    @property
    def value_outliers_enabled(self) -> bool:
        return self.n_value_outliers > 0

    @property
    def value_outlier_bytes(self) -> int:
        """Inline side-channel bytes: per outlier 1 index byte + 2 fp16 bytes."""
        return self.n_value_outliers * 3

    @property
    def sink_enabled(self) -> bool:
        """Whether attention-sink fp16 retention is active (KVQ-2)."""
        return self.sink_tokens > 0

    @property
    def sink_kv_bytes_per_token(self) -> int:
        """fp16 K+V bytes retained per sink token, per KV head.

        Sinks keep both key and value at full fp16 precision:
        head_dim * 2 bytes (K) + head_dim * 2 bytes (V).
        """
        return 2 * self.head_dim * 2

    def sink_side_bytes_per_seq(self, num_kv_heads: int) -> int:
        """Honest per-sequence side-buffer cost for sink retention (KVQ-2).

        A fixed cost independent of context length: only the first
        ``sink_tokens`` positions are retained, across all KV heads.
        """
        if not self.sink_enabled:
            return 0
        return self.sink_tokens * num_kv_heads * self.sink_kv_bytes_per_token

    @property
    def key_packed_size(self) -> int:
        """Packed bytes for a single KEY vector.

        FP8 mode (key_quant_bits=8):
          head_dim bytes (1 byte per element, no overhead).

        TQ mode:
          - MSE indices: ceil(head_dim * key_mse_bits / 8) bytes
          - vec_norm:     2 bytes (float16)
        """
        if self.key_fp8:
            return self.head_dim  # 1 byte per element
        mse_bytes = math.ceil(self.head_dim * self.key_mse_bits / 8)
        norm_bytes = 2  # vec_norm fp16
        return mse_bytes + norm_bytes

    @property
    def effective_value_quant_bits(self) -> int:
        """Actual bits used for value storage."""
        return self.value_quant_bits

    @property
    def value_packed_size(self) -> int:
        """Packed bytes for a single VALUE vector.

        Layout: [packed indices | scale(fp16) | zero(fp16) | outliers].
        Base = ceil(head_dim * bits / 8) + 4 bytes (scale + zero fp16). When
        the KVQ-3 outlier side-channel is enabled, appends
        ``n_value_outliers * 3`` bytes (index + fp16 value per outlier).
        """
        data_bytes = math.ceil(self.head_dim * self.value_quant_bits / 8)
        return data_bytes + 4 + self.value_outlier_bytes

    @property
    def value_outlier_offset(self) -> int:
        """Byte offset of the outlier region within a value vector (KVQ-3).

        Sits after the packed indices and the (scale, zero) fp16 pair.
        """
        data_bytes = math.ceil(self.head_dim * self.value_quant_bits / 8)
        return data_bytes + 4

    @property
    def slot_size(self) -> int:
        """Total packed bytes per head per position (key + value combined).

        Layout: [key_packed | value_packed]
        """
        return self.key_packed_size + self.value_packed_size

    @property
    def slot_size_aligned(self) -> int:
        """Slot size rounded up to next even number.

        Even-number is required so effective_head_size = slot_size_aligned // 2
        is integral.
        """
        s = self.slot_size
        return s + (s % 2)  # round up to even

    @staticmethod
    def get_boundary_skip_layers(
        model_config: ModelConfig,
        n: int = 2,
    ) -> list[str]:
        """Layer indices to skip TQ compression (boundary protection).

        For hybrid models (attention + Mamba/linear-attention), boundary
        protection is disabled — hybrids typically have only 8-12
        full-attention layers and a hard n=2 on each side would cover
        ~40 % of them.  The dense GSM8K baselines that motivate n=2
        don't apply to hybrids.

        For dense models, skips first N and last N attention layers.
        Empirically required for aggressive presets (k3v4_nc, 3bit_nc)
        — without it GSM8K drops ~30 points on Qwen3-4B.
        """
        if model_config.is_hybrid:
            attn_indices = _get_full_attention_layer_indices(model_config)
            if not attn_indices:
                raise NotImplementedError(
                    "TurboQuant KV cache requires identifiable "
                    "full-attention layers, but none were found in "
                    "the hybrid model config."
                )
            logger.info("TQ hybrid: full-attention layers %s", attn_indices)
            return []

        num_layers = model_config.hf_text_config.num_hidden_layers
        if n <= 0 or num_layers <= 0:
            return []
        n = min(n, num_layers // 2)  # don't skip more than half
        first = list(range(n))
        last = list(range(num_layers - n, num_layers))
        # Deduplicate (if num_layers <= 2*n)
        indices = sorted(set(first + last))
        return [str(i) for i in indices]

    @staticmethod
    def from_cache_dtype(cache_dtype: str, head_dim: int) -> TurboQuantConfig:
        """Create config from a named preset.

        Valid presets: turboquant_k8v4, turboquant_4bit_nc, etc.
        """
        if cache_dtype not in TQ_PRESETS:
            valid = ", ".join(TQ_PRESETS.keys())
            raise ValueError(
                f"Unknown TurboQuant cache dtype: {cache_dtype!r}. "
                f"Valid presets: {valid}"
            )
        preset = TQ_PRESETS[cache_dtype]
        return TurboQuantConfig(
            head_dim=head_dim,
            key_quant_bits=preset["key_quant_bits"],
            value_quant_bits=preset["value_quant_bits"],
            norm_correction=preset["norm_correction"],
            value_nuq=preset.get("value_nuq", False),
            sink_tokens=preset.get("sink_tokens", 0),
            value_outlier_pct=preset.get("value_outlier_pct", 0.0),
        )


@lru_cache(maxsize=8)
def _parse_tq_layer_bits(raw: str) -> tuple[tuple[int, str], ...]:
    """Parse and validate the per-layer bit map JSON (KVQ-4).

    Cached on the raw string so repeated resolution is cheap.

    Args:
        raw: JSON object string ``{"<layer_idx>": "<preset>"}`` (or empty).

    Returns:
        Tuple of ``(layer_idx, preset_name)`` pairs.

    Raises:
        ValueError: on malformed JSON, non-integer keys, or unknown presets.
    """
    if not raw:
        return ()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{TQ_LAYER_BITS_ENV} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(
            f"{TQ_LAYER_BITS_ENV} must be a JSON object mapping layer index to "
            f"preset name, got {type(data).__name__}."
        )
    items: list[tuple[int, str]] = []
    for k, v in data.items():
        try:
            idx = int(k)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"{TQ_LAYER_BITS_ENV} layer key {k!r} is not an integer."
            ) from e
        if v not in TQ_PRESETS:
            valid = ", ".join(TQ_PRESETS)
            raise ValueError(
                f"{TQ_LAYER_BITS_ENV} maps layer {idx} to unknown preset {v!r}. "
                f"Valid presets: {valid}"
            )
        items.append((idx, v))
    return tuple(items)


def get_tq_layer_bits_map() -> dict[int, str]:
    """Current per-layer preset override map from the environment (KVQ-4)."""
    raw = os.environ.get(TQ_LAYER_BITS_ENV, "").strip()
    return dict(_parse_tq_layer_bits(raw))


def resolve_tq_layer_preset(layer_idx: int, default_dtype: str) -> str:
    """Resolve the effective TQ preset for a layer (KVQ-4).

    ``layer_idx`` is the GLOBAL model layer index (``extract_layer_index`` of
    the layer name), matching the key basis of ``VLLM_TQ_LAYER_BITS``.
    Returns the per-layer override if the layer index is present in the map,
    otherwise ``default_dtype`` (the model-level ``--kv-cache-dtype`` preset).
    With no map set, this is always ``default_dtype`` — a uniform allocation
    with no behavior change.
    """
    return get_tq_layer_bits_map().get(layer_idx, default_dtype)


def _get_full_attention_layer_indices(model_config: ModelConfig) -> list[int]:
    """Global indices of full-attention layers in a hybrid model.

    Covers the conventions used across vLLM: ``layer_types`` (Qwen3.5/Next),
    ``layers_block_type`` (Jamba/Zamba2), ``attn_type_list`` (Minimax).
    """
    text_cfg = model_config.hf_text_config
    hf_cfg = model_config.hf_config

    layer_types = getattr(text_cfg, "layer_types", None)
    if layer_types is not None:
        return [
            i for i, t in enumerate(layer_types) if t in ("full_attention", "attention")
        ]

    layers_block_type = getattr(text_cfg, "layers_block_type", None)
    if layers_block_type is not None:
        return [
            i for i, t in enumerate(layers_block_type) if t in ("attention", "hybrid")
        ]

    attn_type_list = getattr(hf_cfg, "attn_type_list", None)
    if attn_type_list is not None:
        return [i for i, t in enumerate(attn_type_list) if t == 1]

    return []
