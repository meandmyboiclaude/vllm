# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up Triton kernels for hybrid Mamba2 models (e.g. NemotronH).

Extends the Qwen Triton warmup (#46750 / #47546) to hybrid models built on
``MambaMixer2``. Without this, the JIT monitor reports
``_causal_conv1d_fwd_kernel`` compiling during the first inference request:
the Mamba2 SSD warmup covers only the SSD chunk kernels, and it runs during
the profile pass before the conv cache exists, so the prefill conv kernel
cannot be warmed there.

Earlier revisions also warmed ``_zero_kv_blocks_kernel`` and
``_compute_slot_mapping_kernel``; upstream #49903 warms both natively
(``KVBlockZeroer`` and the runner-owned block-table warmup), so those legs
were dropped.
"""

import itertools
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.warmup.qwen_triton_warmup import (
    _synchronize_device,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def _iter_mamba_mixer2_layers(static_forward_context: object):
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2

    if not isinstance(static_forward_context, dict):
        return
    for module in static_forward_context.values():
        if isinstance(module, MambaMixer2):
            yield module


def _get_conv_state(layer: object) -> torch.Tensor | None:
    from vllm.model_executor.layers.mamba.mamba_utils import (
        is_conv_state_dim_first,
    )

    kv_cache = getattr(layer, "kv_cache", None)
    if not isinstance(kv_cache, (list, tuple)) or len(kv_cache) < 1:
        return None
    conv_cache = kv_cache[0]
    if not isinstance(conv_cache, torch.Tensor) or conv_cache.numel() == 0:
        return None
    return conv_cache if is_conv_state_dim_first() else conv_cache.transpose(-1, -2)


# In production the prefill tensors (query_start_loc_p, cache_indices_p,
# has_initial_state_p) are slices offset by num_decodes of the per-step
# batch tensors, so each pointer's 16-byte alignment varies per step and
# independently of each other. On this branch those pointers sit in the
# kernel's ``do_not_specialize_on_alignment`` list, so all eight
# aligned/unaligned combinations resolve to one JIT key and seven of the
# launches below are compile-cache hits. The sweep is kept as a cheap
# guard: if that list ever loses an entry, the warmup still covers every
# alignment variant instead of leaving first-request compiles behind.
_CONV1D_WARMUP_SLICE_OFFSETS = tuple(itertools.product((0, 1), repeat=3))


def _warm_mamba2_causal_conv1d_fwd(
    device: torch.device, layer: object, model_config: object
) -> bool:
    """Warm ``_causal_conv1d_fwd_kernel`` with the same JIT keys as a real
    prefill on ``layer``: real conv weights/bias/state (so the constexpr
    dims, dtypes and strides match) and a single dummy token routed to the
    null block (so no real cache line is written). With
    ``mamba_cache_mode="all"`` the mixer launches the ``IS_APC_ENABLED=True``
    specialization instead, so that variant is warmed too."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
    )
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    conv_state = _get_conv_state(layer)
    if conv_state is None:
        return False

    cache_config = getattr(layer, "cache_config", None)
    mamba_block_size = getattr(cache_config, "mamba_block_size", None)
    if mamba_block_size is None:
        return False

    # Channel-last single-token input, like production's transposed slice
    # of the projected-states buffer. A dense transpose keeps the output
    # token stride > 1 so Triton's ``== 1`` stride specialization matches
    # the production key.
    dim = layer.conv_weights.size(0)
    x = torch.zeros((1, dim), dtype=conv_state.dtype, device=device).t()

    for qsl_offset, ci_offset, bool_offset in _CONV1D_WARMUP_SLICE_OFFSETS:
        query_start_loc = torch.zeros(
            qsl_offset + 2, dtype=torch.int32, device=device
        )[qsl_offset:]
        query_start_loc[1] = 1
        cache_indices = torch.full(
            (ci_offset + 1,), NULL_BLOCK_ID, dtype=torch.int32, device=device
        )[ci_offset:]
        has_initial_state = torch.zeros(
            bool_offset + 1, dtype=torch.bool, device=device
        )[bool_offset:]

        causal_conv1d_fn(
            x,
            layer.conv_weights,
            layer.conv1d.bias,
            activation=layer.activation,
            conv_states=conv_state,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            block_size_to_align=mamba_block_size,
            metadata=None,
            query_start_loc=query_start_loc,
        )

    if getattr(cache_config, "mamba_cache_mode", None) == "all":
        # Prefix-cached prefill ("all" mode) launches the IS_APC_ENABLED=True
        # specialization: the mixer passes the four per-request block-index
        # tensors and a 2D block-table slice as cache_indices, whose row
        # stride enters the JIT key as a constexpr. Mirror that geometry with
        # a one-row table routed entirely to the null block; the kernel loads
        # the APC scalars and returns before touching any cache line. The row
        # width below is the block table's cdiv(max_model_len,
        # mamba_block_size) columns; if the runner ever pads its table the
        # only cost is one extra compiled variant.
        max_model_len = getattr(model_config, "max_model_len", None)
        if max_model_len is not None:
            num_cols = max(1, -(-max_model_len // mamba_block_size))
            apc_query_start_loc = torch.zeros(2, dtype=torch.int32, device=device)
            apc_query_start_loc[1] = 1
            apc_cache_indices = torch.full(
                (1, num_cols), NULL_BLOCK_ID, dtype=torch.int32, device=device
            )
            apc_zero = torch.zeros(1, dtype=torch.int32, device=device)
            causal_conv1d_fn(
                x,
                layer.conv_weights,
                layer.conv1d.bias,
                activation=layer.activation,
                conv_states=conv_state,
                has_initial_state=torch.zeros(1, dtype=torch.bool, device=device),
                cache_indices=apc_cache_indices,
                block_idx_first_scheduled_token=apc_zero,
                block_idx_last_scheduled_token=apc_zero,
                initial_state_idx=apc_zero,
                num_computed_tokens=apc_zero,
                block_size_to_align=mamba_block_size,
                metadata=None,
                query_start_loc=apc_query_start_loc,
            )
    return True


@torch.inference_mode()
def hybrid_mamba_triton_warmup(
    runner: "GPUModelRunner",
    model_config: object,
) -> None:
    """Warm hybrid-Mamba2 Triton kernels reported by the JIT monitor."""
    if runner.is_pooling_model:
        return

    compilation_config = getattr(runner, "compilation_config", None)
    static_forward_context = getattr(compilation_config, "static_forward_context", None)
    mixer_layers = list(_iter_mamba_mixer2_layers(static_forward_context))
    if not mixer_layers:
        return

    device = getattr(runner, "device", torch.device("cuda"))
    logger.info("Warming up hybrid Mamba2 Triton kernels.")

    # Prefill causal-conv1d kernel: warm one layer per distinct JIT key.
    seen_keys: set[tuple] = set()
    warmed_any = False
    for layer in mixer_layers:
        conv_state = _get_conv_state(layer)
        if conv_state is None:
            continue
        key = (
            layer.conv_weights.size(0),
            layer.conv_weights.size(1),
            conv_state.dtype,
            conv_state.size(0),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if _warm_mamba2_causal_conv1d_fwd(device, layer, model_config):
            warmed_any = True
    if not warmed_any:
        logger.info(
            "Skipping hybrid causal-conv1d warmup: no bound Mamba2 conv cache found."
        )

    _synchronize_device(device)
