# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up spec-decode rejection-sampler Triton kernels.

The rejection sampler kernels (``_compute_local_logits_stats_kernel``,
``_rejection_kernel``, ``_resample_kernel``) are JIT-compiled by Triton on
first use. Without warmup, the first spec-decode request pays a multi-second
compilation cost. This pre-compiles them with dummy data matching the
server's vocab size and speculative config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


@torch.inference_mode()
def spec_decode_rejection_warmup(worker: Worker) -> None:
    spec_config = worker.vllm_config.speculative_config
    if spec_config is None:
        return

    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )

    model_config = worker.vllm_config.model_config
    vocab_size = model_config.get_vocab_size()
    num_spec = spec_config.num_speculative_tokens
    if num_spec <= 0 or vocab_size <= 0:
        return

    # Mirror the constexpr-relevant flags the runtime uses.
    rejection_method = getattr(spec_config, "rejection_sample_method", None)
    use_block_verification = rejection_method == "block"
    use_synthetic = rejection_method == "synthetic"

    # HAS_DRAFT_LOGITS is a constexpr of _compute_local_logits_stats_kernel,
    # _rejection_kernel and _resample_kernel, so the two legs are separate
    # compiles. RejectionSampler is handed Speculator.draft_logits, which is
    # allocated only for draft_sample_method="probabilistic" and is None for
    # every other method (greedy DFlash included). Warm the served leg first,
    # then the other one as a fallback, so no first request can pay JIT.
    draft_sample_method = getattr(spec_config, "draft_sample_method", "greedy")
    serves_draft_logits = draft_sample_method == "probabilistic"

    # USE_FP64 is a constexpr of _resample_kernel and also switches the
    # resampled_local_max buffer to fp64. The runtime value comes from
    # Sampler(use_fp64_gumbel=model_config.use_fp64_gumbel); warming False
    # while the server runs True leaves the served kernel uncompiled.
    use_fp64 = bool(getattr(model_config, "use_fp64_gumbel", False))

    device = torch.device("cuda")
    num_reqs = 1
    tokens_per_req = num_spec + 1
    num_logits = num_reqs * tokens_per_req

    # Triton JIT-specializes on tensor dtypes. The target logits may be fp32
    # (apply_sampling_params copies to fp32 when processing is needed) or the
    # logits dtype (pass-through otherwise); draft logits carry the dtype
    # Speculator.draft_logits_spec() picks, which is model_config.head_dtype
    # (equal to the model dtype unless the lm_head is overridden). Warm every
    # (target, draft) combination the runtime can hit.
    model_dtype = model_config.dtype
    try:
        head_dtype = model_config.head_dtype
    except Exception:  # head_dtype validates and may reject on odd configs
        head_dtype = model_dtype
    logits_dtypes = {model_dtype, head_dtype}
    target_dtypes = logits_dtypes | {torch.float32}
    draft_dtypes = logits_dtypes | {torch.float32}

    # (has_draft_logits, target_dtype, draft_dtype); draft dtype is irrelevant
    # on the None leg, so that leg costs one compile set per target dtype.
    warmup_cases: list[tuple[bool, torch.dtype, torch.dtype | None]] = []
    legs = (True, False) if serves_draft_logits else (False, True)
    for has_draft_logits in legs:
        if has_draft_logits:
            warmup_cases += [
                (True, tgt, draft)
                for tgt in sorted(target_dtypes, key=str)
                for draft in sorted(draft_dtypes, key=str)
            ]
        else:
            warmup_cases += [
                (False, tgt, None) for tgt in sorted(target_dtypes, key=str)
            ]

    logger.info(
        "Warming up spec-decode rejection sampler kernels "
        "(vocab=%d, num_spec=%d, draft_sample_method=%s, cases=%s, "
        "use_fp64=%s, block_verify=%s).",
        vocab_size,
        num_spec,
        draft_sample_method,
        [(has, str(t), str(d)) for has, t, d in warmup_cases],
        use_fp64,
        use_block_verification,
    )
    for has_draft_logits, tgt_dtype, draft_dtype in warmup_cases:
        # Allocations stay inside the try: an OOM here must not abort
        # compile_or_warm_up_model, exactly like a kernel failure below.
        try:
            target_logits = torch.zeros(
                (num_logits, vocab_size), dtype=tgt_dtype, device=device
            )
            draft_logits = (
                torch.zeros(
                    (num_reqs, num_spec, vocab_size),
                    dtype=draft_dtype,
                    device=device,
                )
                if has_draft_logits
                else None
            )
            synthetic_rates = (
                torch.full((num_spec,), 0.5, dtype=torch.float32, device=device)
                if use_synthetic
                else None
            )
            rejection_sample(
                target_logits=target_logits,
                draft_logits=draft_logits,
                # draft_sampled is input_batch.input_ids[logits_indices]; the
                # input_ids buffer is int32, and the pointer dtype is part of
                # the Triton specialization key.
                draft_sampled=torch.zeros(num_logits, dtype=torch.int32, device=device),
                cu_num_logits=torch.tensor(
                    [0, num_logits], dtype=torch.int32, device=device
                ),
                # positions buffer is int64 (InputBuffers.positions).
                pos=torch.zeros(num_logits, dtype=torch.int64, device=device),
                # idx_mapping / expanded_idx_mapping are int64 on the served
                # MRV2 runner (#51210 widened them; buffers-1 restored the
                # pre-allocated buffer to int64), expanded_local_pos and
                # cu_num_logits are int32. An int32 idx_mapping here would
                # compile a Triton specialization no real batch uses.
                idx_mapping=torch.zeros(num_reqs, dtype=torch.int64, device=device),
                expanded_idx_mapping=torch.zeros(
                    num_logits, dtype=torch.int64, device=device
                ),
                expanded_local_pos=torch.arange(
                    num_logits, dtype=torch.int32, device=device
                ),
                temperature=torch.zeros(num_reqs, dtype=torch.float32, device=device),
                seed=torch.full((num_reqs,), 42, dtype=torch.int64, device=device),
                num_speculative_steps=num_spec,
                synthetic_conditional_rates=synthetic_rates,
                use_fp64=use_fp64,
                use_block_verification=use_block_verification,
            )
        except Exception:
            logger.warning(
                "Skipping spec-decode rejection sampler warmup "
                "(has_draft_logits=%s, target=%s, draft=%s).",
                has_draft_logits,
                tgt_dtype,
                draft_dtype,
                exc_info=True,
            )
            return
