# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Contains replacement functions to fallback Triton usages in CPU backend
"""

import ctypes
from collections.abc import Callable

import torch


class _FuncWrapper:
    def __init__(self, func: Callable) -> None:
        self.func = func

    def __getitem__(self, *args, **kwargs) -> Callable:
        return self.func


# For _compute_slot_mapping_kernel in vllm/v1/worker/block_table.py
def _compute_slot_mapping_kernel_impl(
    num_tokens: int,
    max_num_tokens: int,
    query_start_loc: torch.Tensor,  # [num_reqs + 1], int32
    positions: torch.Tensor,  # [num_tokens], int64
    block_table: torch.Tensor,  # [max_num_reqs, max_num_blocks_per_req], int32
    block_table_stride: int,  # max_num_blocks_per_req
    block_size: int,
    slot_mapping: torch.Tensor,  # [max_num_tokens], int64
    KV_CACHE_BLOCK_SIZE: int | None = None,
    BLOCKS_PER_KV_BLOCK: int = 1,
    TOTAL_CP_WORLD_SIZE: int = 1,
    TOTAL_CP_RANK: int = 0,
    CP_KV_CACHE_INTERLEAVE_SIZE: int = 1,
    PAD_ID: int = -1,
    BLOCK_SIZE: int = 1024,
) -> None:
    assert TOTAL_CP_WORLD_SIZE == 1, "Context Parallelism is not supported on CPU."
    if BLOCKS_PER_KV_BLOCK != 1:
        assert block_size * BLOCKS_PER_KV_BLOCK == KV_CACHE_BLOCK_SIZE
    torch.ops._C.compute_slot_mapping_kernel_impl(
        query_start_loc,
        positions,
        block_table,
        slot_mapping,
        block_size,
    )


compute_slot_mapping_kernel = _FuncWrapper(_compute_slot_mapping_kernel_impl)


def _ensure_int64(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.int64 else t.to(torch.int64)


def _eagle_prepare_inputs_padded_kernel_impl(
    cu_num_draft_tokens,
    valid_sampled_tokens_count,
    query_start_loc_gpu,
    token_indices_to_sample,
    num_rejected_tokens_gpu,
    num_reqs,
):
    # C++ expects int64 for cu_num_draft_tokens, valid_sampled_tokens_count,
    # and num_rejected_tokens_gpu, but Python allocates them as int32.
    orig_rejected_dtype = num_rejected_tokens_gpu.dtype
    rejected_i64 = (
        num_rejected_tokens_gpu
        if orig_rejected_dtype == torch.int64
        else num_rejected_tokens_gpu.to(torch.int64)
    )
    torch.ops._C.eagle_prepare_inputs_padded_kernel_impl(
        _ensure_int64(cu_num_draft_tokens),
        _ensure_int64(valid_sampled_tokens_count),
        query_start_loc_gpu,
        token_indices_to_sample,
        rejected_i64,
        num_reqs,
    )
    if orig_rejected_dtype != torch.int64:
        num_rejected_tokens_gpu.copy_(rejected_i64.to(orig_rejected_dtype))


def _eagle_prepare_next_token_padded_kernel_impl(
    sampled_token_ids,
    discard_request_mask,
    backup_next_token_ids,
    next_token_ids,
    valid_sampled_tokens_count,
    vocab_size,
    num_sampled_tokens_per_req,
    num_reqs,
    stride=None,
    BLOCK_SIZE_TOKENS=None,
):
    # C++ reads all integer tensors as int64_t*. Output tensors are written
    # in-place so we create int64 copies, call C++, and copy back.
    orig_next_dtype = next_token_ids.dtype
    orig_valid_dtype = valid_sampled_tokens_count.dtype
    next_i64 = _ensure_int64(next_token_ids)
    valid_i64 = _ensure_int64(valid_sampled_tokens_count)
    torch.ops._C.eagle_prepare_next_token_padded_kernel_impl(
        _ensure_int64(sampled_token_ids),
        discard_request_mask,
        _ensure_int64(backup_next_token_ids),
        next_i64,
        valid_i64,
        vocab_size,
        num_sampled_tokens_per_req,
        num_reqs,
    )
    if orig_next_dtype != torch.int64:
        next_token_ids.copy_(next_i64.to(orig_next_dtype))
    if orig_valid_dtype != torch.int64:
        valid_sampled_tokens_count.copy_(valid_i64.to(orig_valid_dtype))


def _eagle_step_slot_mapping_metadata_kernel_impl(
    positions,
    block_table,
    stride,
    seq_lens,
    out_clamped_positions,
    out_slot_mapping,
    block_size,
    max_model_len,
    n_blocks_per_req,
    PAD_ID,
    batch_size=None,
):
    assert batch_size is None or batch_size == positions.shape[0], (
        f"batch_size mismatch: {batch_size} vs positions.shape[0]={positions.shape[0]}"
    )
    torch.ops._C.eagle_step_slot_mapping_metadata_kernel_impl(
        positions,
        block_table,
        seq_lens,
        out_clamped_positions,
        out_slot_mapping,
        block_size,
        max_model_len,
        PAD_ID,
    )


def _copy_and_expand_eagle_inputs_kernel_impl(
    target_token_ids_ptr,
    target_positions_ptr,
    next_token_ids_ptr,
    out_input_ids_ptr,
    out_positions_ptr,
    out_is_rejected_token_mask_ptr,
    out_is_masked_token_mask_ptr,
    out_new_token_indices_ptr,
    out_hidden_state_mapping_ptr,
    query_start_loc_ptr,
    query_end_loc_ptr,
    padding_token_id,
    parallel_drafting_token_id,
    total_input_tokens,
    num_padding_slots_per_request,
    shift_input_ids,
    BLOCK_SIZE_TOKENS=None,
    BLOCK_SIZE_REQS=None,
):
    """Adapter between Triton kernel call convention and C++ implementation.

    The Triton kernel uses '_ptr' suffixed parameter names and compile-time
    constants (BLOCK_SIZE_TOKENS, BLOCK_SIZE_REQS) which are not needed by
    the C++ implementation. C++ reads token id tensors as int64_t*.
    Output tensors that are int32 need copy-back after C++ writes int64.
    """
    orig_ids_dtype = out_input_ids_ptr.dtype
    orig_pos_dtype = out_positions_ptr.dtype
    out_ids_i64 = _ensure_int64(out_input_ids_ptr)
    out_pos_i64 = _ensure_int64(out_positions_ptr)
    torch.ops._C.copy_and_expand_eagle_inputs_kernel_impl(
        _ensure_int64(target_token_ids_ptr),
        _ensure_int64(target_positions_ptr),
        _ensure_int64(next_token_ids_ptr),
        out_ids_i64,
        out_pos_i64,
        out_is_rejected_token_mask_ptr,
        out_is_masked_token_mask_ptr,
        out_new_token_indices_ptr,
        out_hidden_state_mapping_ptr,
        query_start_loc_ptr,
        query_end_loc_ptr,
        padding_token_id,
        parallel_drafting_token_id,
        total_input_tokens,
        num_padding_slots_per_request,
        shift_input_ids,
    )
    if orig_ids_dtype != torch.int64:
        out_input_ids_ptr.copy_(out_ids_i64.to(orig_ids_dtype))
    if orig_pos_dtype != torch.int64:
        out_positions_ptr.copy_(out_pos_i64.to(orig_pos_dtype))


def _copy_and_expand_dflash_inputs_kernel_impl(
    next_token_ids_ptr,
    target_positions_ptr,
    out_input_ids_ptr,
    out_context_positions_ptr,
    out_query_positions_ptr,
    out_context_slot_mapping_ptr,
    out_query_slot_mapping_ptr,
    out_token_indices_ptr,
    block_table_ptr,
    block_table_stride,
    query_start_loc_ptr,
    num_rejected_tokens_ptr,
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_tokens,
    total_input_tokens,
    BLOCK_SIZE=None,
    HAS_NUM_REJECTED=False,
):
    """Adapter between the DFlash Triton launch and the C++ CPU op."""
    assert block_table_stride == block_table_ptr.stride(0), (
        "block_table_stride mismatch: "
        f"{block_table_stride} vs {block_table_ptr.stride(0)}"
    )

    orig_ids_dtype = out_input_ids_ptr.dtype
    orig_context_positions_dtype = out_context_positions_ptr.dtype
    orig_query_positions_dtype = out_query_positions_ptr.dtype
    orig_context_slot_mapping_dtype = out_context_slot_mapping_ptr.dtype
    orig_query_slot_mapping_dtype = out_query_slot_mapping_ptr.dtype
    out_ids_i64 = _ensure_int64(out_input_ids_ptr)
    out_context_positions_i64 = _ensure_int64(out_context_positions_ptr)
    out_query_positions_i64 = _ensure_int64(out_query_positions_ptr)
    out_context_slot_mapping_i64 = _ensure_int64(out_context_slot_mapping_ptr)
    out_query_slot_mapping_i64 = _ensure_int64(out_query_slot_mapping_ptr)
    rejected_i64 = _ensure_int64(num_rejected_tokens_ptr) if HAS_NUM_REJECTED else None

    if hasattr(torch.ops._C, "copy_and_expand_dflash_inputs_kernel_impl"):
        torch.ops._C.copy_and_expand_dflash_inputs_kernel_impl(
            _ensure_int64(next_token_ids_ptr),
            _ensure_int64(target_positions_ptr),
            out_ids_i64,
            out_context_positions_i64,
            out_query_positions_i64,
            out_context_slot_mapping_i64,
            out_query_slot_mapping_i64,
            out_token_indices_ptr,
            block_table_ptr,
            query_start_loc_ptr,
            rejected_i64,
            parallel_drafting_token_id,
            block_size,
            num_query_per_req,
            num_speculative_tokens,
            total_input_tokens,
            HAS_NUM_REJECTED,
        )
    else:
        next_ids_i64 = _ensure_int64(next_token_ids_ptr)
        target_positions_i64 = _ensure_int64(target_positions_ptr)
        block_table_stride = block_table_ptr.stride(0)
        num_reqs = query_start_loc_ptr.shape[0] - 1

        for req_idx in range(num_reqs):
            ctx_start = int(query_start_loc_ptr[req_idx].item())
            ctx_end = int(query_start_loc_ptr[req_idx + 1].item())
            num_ctx = ctx_end - ctx_start
            valid_ctx_end = ctx_end
            if rejected_i64 is not None:
                valid_ctx_end -= int(rejected_i64[req_idx].item())
            # Guard against out-of-bounds: ensure valid_ctx_end > ctx_start.
            valid_ctx_end = max(valid_ctx_end, ctx_start + 1)

            last_pos = int(target_positions_i64[valid_ctx_end - 1].item())

            for j in range(num_ctx):
                ctx_idx = ctx_start + j
                ctx_pos_idx = min(ctx_idx, total_input_tokens - 1)
                position = int(target_positions_i64[ctx_pos_idx].item())
                block_num = min(position // block_size, block_table_stride - 1)
                block_id = int(block_table_ptr[req_idx, block_num].item())
                slot = block_id * block_size + (position % block_size)

                out_context_positions_i64[ctx_idx] = position
                out_context_slot_mapping_i64[ctx_idx] = slot

            for query_off in range(num_query_per_req):
                query_out = req_idx * num_query_per_req + query_off
                position = last_pos + 1 + query_off
                block_num = min(position // block_size, block_table_stride - 1)
                block_id = int(block_table_ptr[req_idx, block_num].item())
                slot = block_id * block_size + (position % block_size)

                out_query_positions_i64[query_out] = position
                out_query_slot_mapping_i64[query_out] = slot
                out_ids_i64[query_out] = (
                    int(next_ids_i64[req_idx].item())
                    if query_off == 0
                    else parallel_drafting_token_id
                )

                if query_off > 0:
                    sample_out_idx = req_idx * num_speculative_tokens + (query_off - 1)
                    out_token_indices_ptr[sample_out_idx] = query_out

    if orig_ids_dtype != torch.int64:
        out_input_ids_ptr.copy_(out_ids_i64.to(orig_ids_dtype))
    if orig_context_positions_dtype != torch.int64:
        out_context_positions_ptr.copy_(
            out_context_positions_i64.to(orig_context_positions_dtype)
        )
    if orig_query_positions_dtype != torch.int64:
        out_query_positions_ptr.copy_(
            out_query_positions_i64.to(orig_query_positions_dtype)
        )
    if orig_context_slot_mapping_dtype != torch.int64:
        out_context_slot_mapping_ptr.copy_(
            out_context_slot_mapping_i64.to(orig_context_slot_mapping_dtype)
        )
    if orig_query_slot_mapping_dtype != torch.int64:
        out_query_slot_mapping_ptr.copy_(
            out_query_slot_mapping_i64.to(orig_query_slot_mapping_dtype)
        )


# Keep in sync with the constants in vllm/v1/sample/rejection_sampler.py.
# Imported literally to avoid an import cycle through vllm.triton_utils.
_PLACEHOLDER_TOKEN_ID = -1
_NO_THINK_BOUNDARY_TOKEN_ID = -2
_FP32_RECOVERY_EPS = float.fromhex("0x1p-24")
_FP64_RECOVERY_EPS = float.fromhex("0x1p-64")


def _reconstruct_target_probs(target_logits, target_max, target_inv_sum):
    """Rebuild dense float32 target probabilities from online-softmax state.

    The Triton rejection kernels consume ``target_logits`` with per-row
    ``target_max``/``target_inv_sum`` normalization state; the C++ CPU
    kernels predate that and expect materialized probabilities.
    """
    return torch.exp(
        target_logits.float() - target_max.float().unsqueeze(-1)
    ) * target_inv_sum.float().unsqueeze(-1)


def _rejection_greedy_sample_kernel_impl(
    output_token_ids,
    cu_num_draft_tokens,
    draft_token_ids,
    target_argmax,
    bonus_token_ids,
    is_greedy,
    max_spec_len,
    think_start_token_id=_NO_THINK_BOUNDARY_TOKEN_ID,
    think_end_token_id=_NO_THINK_BOUNDARY_TOKEN_ID,
    uniform_probs=None,
    synthetic_conditional_rates=None,
    SYNTHETIC_MODE=False,
):
    # C++ kernel expects int64 for all integer tensors.
    # Note: uniform_probs, synthetic_conditional_rates, and SYNTHETIC_MODE are
    # passed by the rejection sampler for synthetic mode support, but are not
    # yet implemented in the C++ CPU kernel. We accept them here to maintain
    # compatibility with the kernel calling convention.
    assert not SYNTHETIC_MODE, "Synthetic acceptance not supported with CPU sampling"
    orig_dtype = output_token_ids.dtype
    output_token_ids_i64 = _ensure_int64(output_token_ids)
    cu_num_draft_tokens_i64 = _ensure_int64(cu_num_draft_tokens)
    draft_token_ids_i64 = _ensure_int64(draft_token_ids)
    target_argmax_i64 = _ensure_int64(target_argmax)
    torch.ops._C.rejection_greedy_sample_kernel_impl(
        output_token_ids_i64,
        cu_num_draft_tokens_i64,
        draft_token_ids_i64,
        target_argmax_i64,
        _ensure_int64(bonus_token_ids),
        is_greedy,
        max_spec_len,
    )
    if (
        think_start_token_id != _NO_THINK_BOUNDARY_TOKEN_ID
        or think_end_token_id != _NO_THINK_BOUNDARY_TOKEN_ID
    ):
        # The Triton kernel stops speculation after an accepted thinking
        # boundary token; the C++ kernel predates the feature, so apply the
        # same truncation to its output here.
        start_idx = 0
        for req_idx in range(cu_num_draft_tokens_i64.numel()):
            end_idx = int(cu_num_draft_tokens_i64[req_idx].item())
            if is_greedy is None or bool(is_greedy[req_idx].item()):
                for pos in range(end_idx - start_idx):
                    draft_id = int(draft_token_ids_i64[start_idx + pos].item())
                    if draft_id != int(target_argmax_i64[start_idx + pos].item()):
                        break
                    if draft_id in (think_start_token_id, think_end_token_id):
                        output_token_ids_i64[req_idx, pos + 1 :] = (
                            _PLACEHOLDER_TOKEN_ID
                        )
                        break
            start_idx = end_idx
    if orig_dtype != torch.int64:
        output_token_ids.copy_(output_token_ids_i64.to(orig_dtype))


def _relaxed_thinking_sample_kernel_impl(
    output_token_ids,
    cu_num_draft_tokens,
    draft_token_ids,
    target_argmax,
    topk_values,
    topk_indices,
    bonus_token_ids,
    is_greedy,
    max_spec_len,
    thinking_states,
    think_start_token_id,
    think_end_token_id,
    log_relax_ratio,
    K=None,
    num_warps=None,
):
    """Python emulation of relaxed_thinking_sample_kernel (no C++ counterpart).

    During the thinking phase a draft token is accepted when it appears in the
    target top-K above ``top1_logit + log_relax_ratio``; outside it, plain
    greedy equality applies. Speculation stops after an accepted thinking
    boundary token, matching the Triton kernel.
    """
    cu = cu_num_draft_tokens.tolist()
    draft_ids = draft_token_ids.tolist()
    argmax_ids = target_argmax.tolist()
    topk_vals = topk_values.tolist()
    topk_ids = topk_indices.tolist()
    thinking = thinking_states.tolist()
    start_idx = 0
    for req_idx, end_idx in enumerate(cu):
        num_draft = end_idx - start_idx
        if is_greedy is not None and not bool(is_greedy[req_idx].item()):
            start_idx = end_idx
            continue
        rejected = False
        for pos in range(num_draft):
            token_idx = start_idx + pos
            draft_id = draft_ids[token_idx]
            token_id = argmax_ids[token_idx]
            if thinking[req_idx]:
                logit_floor = topk_vals[token_idx][0] + log_relax_ratio
                accepted = any(
                    cand_id == draft_id and cand_logit >= logit_floor
                    for cand_logit, cand_id in zip(
                        topk_vals[token_idx], topk_ids[token_idx]
                    )
                )
            else:
                accepted = draft_id == token_id
            if accepted:
                token_id = draft_id
            output_token_ids[req_idx, pos] = token_id
            if not accepted:
                rejected = True
                break
            if draft_id in (think_start_token_id, think_end_token_id):
                rejected = True
                break
        if not rejected:
            output_token_ids[req_idx, num_draft] = int(
                bonus_token_ids[req_idx].item()
            )
        start_idx = end_idx


def _sample_lazy_recovered_tokens(
    draft_token_ids_i64,
    draft_probs,
    target_probs,
    recovery_seeds,
    vocab_size,
    no_draft_probs,
    use_fp64_gumbel,
):
    """Eagerly reproduce the Triton kernel's lazy exponential-race recovery.

    Scores every vocab entry as ``adjusted_prob / Exp(1)`` with noise drawn
    from the token's recovery seed and takes the argmax, falling back to the
    target argmax when every adjusted probability is zero. torch's CPU
    Philox stream differs from Triton's, so results match the recovery
    distribution per seed, not bit-for-bit.
    """
    num_tokens = target_probs.shape[0]
    recovered = torch.zeros(num_tokens, dtype=torch.int64)
    noise_dtype = torch.float64 if use_fp64_gumbel else torch.float32
    eps = _FP64_RECOVERY_EPS if use_fp64_gumbel else _FP32_RECOVERY_EPS
    generator = torch.Generator(device="cpu")
    for token_idx in range(num_tokens):
        token_target = target_probs[token_idx]
        if no_draft_probs:
            prob = token_target.clone()
            draft_id = int(draft_token_ids_i64[token_idx].item())
            if draft_id >= 0:
                # Padded (-1) ids are force-rejected by the caller; the Triton
                # kernel leaves their full target distribution intact.
                prob[draft_id] = 0.0
        else:
            prob = (token_target - draft_probs[token_idx].float()).clamp_(min=0.0)
        generator.manual_seed(int(recovery_seeds[token_idx].item()))
        u = torch.rand(vocab_size, dtype=noise_dtype, generator=generator)
        noise = (-torch.log1p(-u)).clamp_(min=eps)
        score = prob.to(noise_dtype) / noise
        score[prob <= 0.0] = 0.0
        best_id = int(score.argmax().item())
        if score[best_id] > 0.0:
            recovered[token_idx] = best_id
        else:
            recovered[token_idx] = token_target.argmax()
    return recovered


def _rejection_random_sample_kernel_impl(
    output_token_ids,
    cu_num_draft_tokens,
    draft_token_ids,
    draft_probs,
    target_logits,
    target_max,
    target_inv_sum,
    bonus_token_ids,
    recovery_seeds,
    uniform_probs,
    is_greedy,
    max_spec_len,
    vocab_size,
    synthetic_conditional_rates=None,
    BLOCK_SIZE=None,
    NO_DRAFT_PROBS=False,
    SYNTHETIC_MODE=False,
    USE_FP64_GUMBEL=False,
):
    # The C++ CPU kernel keeps the eager-recovery calling convention: dense
    # float32 target probabilities plus precomputed recovered token ids. The
    # Triton kernel consumes online-softmax state with lazy in-kernel
    # recovery, so rebuild both eager inputs before dispatching.
    # uniform_probs is intentionally float64 in Python to avoid exact-zero
    # samples; cast to float32 here for C++ compatibility.
    # Note: synthetic_conditional_rates and SYNTHETIC_MODE are passed by the
    # rejection sampler for synthetic mode support, but are not yet implemented
    # in the C++ CPU kernel. We accept them here to maintain compatibility with
    # the kernel calling convention.
    assert not SYNTHETIC_MODE, "Synthetic acceptance not supported with CPU sampling"
    orig_dtype = output_token_ids.dtype
    output_token_ids_i64 = _ensure_int64(output_token_ids)
    draft_token_ids_i64 = _ensure_int64(draft_token_ids)
    target_probs = _reconstruct_target_probs(target_logits, target_max, target_inv_sum)
    uniform_probs_f32 = uniform_probs.to(torch.float32)
    # Recovery must see the original ids: the Triton kernel never masks the
    # target probability of a padded (-1) draft id.
    recovered_token_ids = _sample_lazy_recovered_tokens(
        draft_token_ids_i64,
        draft_probs,
        target_probs,
        recovery_seeds,
        vocab_size,
        NO_DRAFT_PROBS,
        USE_FP64_GUMBEL,
    )
    padded = draft_token_ids_i64 < 0
    if bool(padded.any()):
        # The Triton kernel rejects padded (-1) draft ids outright; the C++
        # kernel would index target_probs with them, so clamp the id and
        # force the rejection through an unreachable uniform threshold.
        draft_token_ids_i64 = torch.where(
            padded, torch.zeros_like(draft_token_ids_i64), draft_token_ids_i64
        )
        if uniform_probs_f32 is uniform_probs:
            uniform_probs_f32 = uniform_probs_f32.clone()
        uniform_probs_f32[padded] = float("inf")
    torch.ops._C.rejection_random_sample_kernel_impl(
        output_token_ids_i64,
        _ensure_int64(cu_num_draft_tokens),
        draft_token_ids_i64,
        draft_probs,
        target_probs,
        _ensure_int64(bonus_token_ids),
        recovered_token_ids,
        uniform_probs_f32,
        is_greedy,
        max_spec_len,
        vocab_size,
        NO_DRAFT_PROBS,
    )
    if orig_dtype != torch.int64:
        output_token_ids.copy_(output_token_ids_i64.to(orig_dtype))


def _expand_kernel_impl(
    output,
    input_val,
    cu_num_tokens,
    replace_from,
    replace_to,
    MAX_NUM_TOKENS=None,
):
    torch.ops._C.expand_kernel_impl(
        _ensure_int64(output),
        _ensure_int64(input_val),
        _ensure_int64(cu_num_tokens),
        replace_from,
        replace_to,
    )


def _sample_recovered_tokens_kernel_impl(
    output_token_ids,
    cu_num_draft_tokens,
    draft_token_ids,
    draft_probs,
    target_logits,
    target_max,
    target_inv_sum,
    inv_q,
    vocab_size,
    BLOCK_SIZE=None,
    NO_DRAFT_PROBS=False,
    USE_FP64_GUMBEL=False,
):
    # USE_FP64_GUMBEL only controls the gumbel-noise precision, which the caller
    # has already applied to `inv_q` (fp64 vs fp32). The CPU kernel consumes
    # `inv_q` directly, so the flag is accepted for interface parity and the
    # value is read at its existing dtype.
    # C++ reads integer tensors as int64_t*; ensure correct dtype. The C++
    # kernel also predates the online-softmax state, so materialize dense
    # target probabilities from it.
    orig_dtype = output_token_ids.dtype
    output_i64 = _ensure_int64(output_token_ids)
    torch.ops._C.sample_recovered_tokens_kernel_impl(
        output_i64,
        _ensure_int64(cu_num_draft_tokens),
        _ensure_int64(draft_token_ids),
        draft_probs,
        _reconstruct_target_probs(target_logits, target_max, target_inv_sum),
        # C++ kernel reads inv_q as float32.
        inv_q.to(torch.float32),
        vocab_size,
        NO_DRAFT_PROBS,
    )
    if orig_dtype != torch.int64:
        output_token_ids.copy_(output_i64.to(orig_dtype))


eagle_prepare_inputs_padded_kernel = _FuncWrapper(
    _eagle_prepare_inputs_padded_kernel_impl
)
eagle_prepare_next_token_padded_kernel = _FuncWrapper(
    _eagle_prepare_next_token_padded_kernel_impl
)
copy_and_expand_eagle_inputs_kernel = _FuncWrapper(
    _copy_and_expand_eagle_inputs_kernel_impl
)
copy_and_expand_dflash_inputs_kernel = _FuncWrapper(
    _copy_and_expand_dflash_inputs_kernel_impl
)
eagle_step_slot_mapping_metadata_kernel = _FuncWrapper(
    _eagle_step_slot_mapping_metadata_kernel_impl
)
rejection_greedy_sample_kernel = _FuncWrapper(_rejection_greedy_sample_kernel_impl)
rejection_random_sample_kernel = _FuncWrapper(_rejection_random_sample_kernel_impl)
relaxed_thinking_sample_kernel = _FuncWrapper(_relaxed_thinking_sample_kernel_impl)
expand_kernel = _FuncWrapper(_expand_kernel_impl)
sample_recovered_tokens_kernel = _FuncWrapper(_sample_recovered_tokens_kernel_impl)


def _batch_memcpy_impl(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=None):
    # BLOCK_SIZE is unused; kept for signature parity with the Triton kernel.
    for src, dst, size in zip(src_ptrs.tolist(), dst_ptrs.tolist(), sizes.tolist()):
        ctypes.memmove(dst, src, size)


batch_memcpy_kernel = _FuncWrapper(_batch_memcpy_impl)
