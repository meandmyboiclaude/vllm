# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

import math
from dataclasses import dataclass, replace
from typing import Literal

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointBuilder,
    MambaPrefillCheckpointMetadata,
)
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu.attn_utils import compute_common_gdn_attn_metadata

# FlashInfer's GDN prefill emits states every N tokens, N a multiple of its
# 64-token chunk, so checkpoint offsets must be multiples of 64.
FLASHINFER_GDN_CHECKPOINT_ALIGNMENT = 64


def plan_flashinfer_checkpoints(
    offsets: list[int], query_lens: list[int]
) -> tuple[int, list[int], list[int]]:
    """Lay out internal checkpoints as FlashInfer GDN prefill emits them.

    Args:
        offsets: Checkpoint offset into each checkpoint row's query, 0 for
            none. The last ``len(query_lens)`` rows are the prefill kernel's.
        query_lens: Query length of each sequence the prefill kernel runs.

    Returns:
        ``(every_n_tokens, cu_starts, rows)``: the stride shared by every
        checkpoint in the batch, cumulative per-sequence checkpoint counts,
        and the index of each checkpoint row's state among the emitted ones.

    """
    every_n_tokens = math.gcd(*offsets)
    cu_starts = [0]
    for query_len in query_lens:
        cu_starts.append(cu_starts[-1] + query_len // every_n_tokens)
    first = len(offsets) - len(query_lens)
    rows = [0] * len(offsets)
    for i, offset in enumerate(offsets[first:]):
        if offset:
            rows[first + i] = cu_starts[i] + offset // every_n_tokens - 1
    return every_n_tokens, cu_starts, rows


@dataclass
class GDNFlashInferCheckpointPlan:
    every_n_tokens: int
    num_checkpoints: int
    cu_starts: torch.Tensor
    # Per checkpoint row, its state among the ``num_checkpoints`` emitted.
    rows: torch.Tensor


class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    checkpoint: MambaPrefillCheckpointMetadata | None = None
    flashinfer_checkpoint_plan: GDNFlashInferCheckpointPlan | None = None
    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_sequence_masks_cpu: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # Per-decode-row ring write position for the cached decode kernel.
    # shape: [num_decodes]; None unless use_replayssm is enabled.
    write_pos_d: torch.Tensor | None = None

    # Cached-SPEC decode cursors: persistent, block-keyed (full (num_blocks,)
    # fixed-address buffers indexed by spec_state_indices_tensor[:, 0]), advanced
    # once per step by commit_gdn_replayssm_spec. None unless
    # use_replayssm_spec is enabled.
    spec_write_pos_d: torch.Tensor | None = None
    spec_cache_base_d: torch.Tensor | None = None
    spec_is_flush_d: torch.Tensor | None = None

    # Batch-level spec-row state of this step, carried so sibling GDN groups
    # (update_block_table) null and reset their own state ids exactly as the
    # source group's build() does: the stale zero-accept rows (bool,
    # [num_spec_decodes]) and the first-decode flags of the ReplaySSM-spec
    # cursor reset (int8, [num_spec_decodes]; None when no reset ran).
    stale_spec_reqs: torch.Tensor | None = None
    spec_first_decode_d: torch.Tensor | None = None
    # True when build() sliced the spec rows (they lead the batch, padding
    # trails); update_block_table() takes the same path without re-checking.
    spec_rows_lead: bool = False

    uniform_spec_sequence_length: int | None = None  # None for ragged batches

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None
    # Chunk-kernel inputs for prefill
    prefill_query_start_loc: torch.Tensor | None = None
    prefill_state_indices: torch.Tensor | None = None
    prefill_has_initial_state: torch.Tensor | None = None
    aiter_prefill_metadata: object | None = None

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    kv_cache_spec: MambaSpec
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.checkpoint_builder = MambaPrefillCheckpointBuilder(
            vllm_config, kv_cache_spec
        )
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            _resolve_gdn_prefill_backend,
        )

        self.gdn_prefill_backend: Literal[
            "triton", "flashinfer", "cutedsl", "aiter_flydsl"
        ]
        _, self.gdn_prefill_backend = _resolve_gdn_prefill_backend(vllm_config)
        self._check_chunk_metadata_override(type(self), self.gdn_prefill_backend)
        # Only the Qwen GDN layer opts in on FlashInfer; KDA layers sharing this
        # backend export checkpoints their own way.
        checkpoint_alignment = kv_cache_spec.prefill_checkpoint_alignment
        self.plan_flashinfer_checkpoints = (
            self.gdn_prefill_backend == "flashinfer"
            and checkpoint_alignment is not None
            and checkpoint_alignment % FLASHINFER_GDN_CHECKPOINT_ALIGNMENT == 0
        )

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode: bool = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.use_full_cuda_graph: bool = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        # update_block_table() keeps the source group's batch-level FULL graph
        # buffers, so only MRV2, which also reuses metadata at capture, may use
        # it.
        self.supports_update_block_table = (
            vllm_config.use_v2_model_runner
            and device.type == "cuda"
            # Not isinstance: KDA's RecoverSSM/checkpoint metadata is per group.
            and type(self) is GDNAttentionMetadataBuilder
        )
        if self.supports_update_block_table:
            # Opts into MRV2's CUDA-only aligned-index precompute.
            self.mamba_aligned_state_indices: torch.Tensor | None = None

        self.decode_cudagraph_max_bs: int = (
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        self.spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_sequence_masks: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.bool,
            device=device,
        )
        self.spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

        # Cached decode kernel: persistent per-decode-row ring write position.
        # write_pos is derived per request each step (decode_step % max_cache_len)
        # so recycled paged blocks need no zero-init.
        self.use_cached_kernel: bool = vllm_config.cache_config.use_replayssm
        self.max_cache_len: int = vllm_config.cache_config.replayssm_buffer_len
        if self.use_cached_kernel:
            # zeros, not empty: capture runs before any real decode batch has
            # staged positions, and the captured kernel executes once with
            # whatever this buffer holds.
            self.decode_write_pos_d: torch.Tensor = torch.zeros(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )

        # Cached-SPEC decode: block-keyed cursors (sized num_gpu_blocks),
        # allocated lazily on first build (num_gpu_blocks is unknown here), and
        # advanced once per step by commit_gdn_replayssm_spec.
        self.use_cache_spec_kernel: bool = vllm_config.cache_config.use_replayssm_spec
        self.max_spec_len: int = 1 + self.num_spec
        # L = B + max_spec_len history window; physical pow2 ring = next_pow2(L).
        self.spec_flush_threshold = self.max_cache_len + self.max_spec_len
        self.spec_cache_buf_len = 1 << (self.spec_flush_threshold - 1).bit_length()
        self.cursor_device = device
        self.spec_write_pos: torch.Tensor | None = None
        self.spec_cache_base: torch.Tensor | None = None
        self.spec_is_flush: torch.Tensor | None = None

        # build() is shared with cudagraph capture, which carries no CPU
        # prompt/computed-token counts (the runner populates them off the
        # capture path only) and never replays a capture-time write position.
        self._capturing: bool = False

    @staticmethod
    def _check_chunk_metadata_override(
        builder_cls: type["GDNAttentionMetadataBuilder"], backend: str
    ) -> None:
        """Reject a subclass whose chunk metadata the AITER path would skip.

        AITER brings its own varlen prefill metadata and never calls
        ``_build_chunk_metadata``, so a builder that overrides it would lose
        that override without a word. No in-tree subclass can get here --
        ``_resolve_gdn_prefill_backend`` only selects this backend for models
        with 128-dim GDN key and value heads, which the KDA builders are not --
        but a future one should be told rather than quietly ignored.
        """
        if backend != "aiter_flydsl":
            return
        if (
            builder_cls._build_chunk_metadata
            is GDNAttentionMetadataBuilder._build_chunk_metadata
        ):
            return
        raise RuntimeError(
            f"{builder_cls.__name__} builds its own FLA chunk metadata, which "
            "the 'aiter_flydsl' GDN prefill backend does not use. Select a "
            "different gdn_prefill_backend for this model."
        )

    def _build_chunk_metadata(
        self,
        prefill_query_start_loc: torch.Tensor,
        prefill_query_start_loc_cpu: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

        if self.gdn_prefill_backend == "cutedsl":
            from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                prepare_metadata_cutedsl,
            )

            assert prefill_query_start_loc is not None
            assert prefill_query_start_loc_cpu is not None
            total_tokens = int(prefill_query_start_loc_cpu[-1].item())
            return prepare_metadata_cutedsl(
                prefill_query_start_loc,
                total_tokens,
                FLA_CHUNK_SIZE,
            )

        # Only prefill batches use FLA chunk ops.
        # Pre-compute on CPU and async-copy to GPU to avoid
        # GPU→CPU sync (.tolist()) in prepare_chunk_indices.
        from vllm.third_party.flash_linear_attention.ops.index import (
            prepare_chunk_indices,
            prepare_chunk_offsets,
        )

        assert prefill_query_start_loc_cpu is not None
        return (
            async_tensor_h2d(
                prepare_chunk_indices(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                device=device,
            ),
            async_tensor_h2d(
                prepare_chunk_offsets(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                device=device,
            ),
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
        num_prefills: int = 0,
        num_prefill_tokens: int = 0,
        num_decodes: int = 0,
        num_decode_tokens: int = 0,
        num_spec_decodes: int = 0,
        num_spec_decode_tokens: int = 0,
        spec_query_start_loc: torch.Tensor | None = None,
        non_spec_query_start_loc: torch.Tensor | None = None,
        non_spec_query_start_loc_cpu: torch.Tensor | None = None,
        spec_sequence_masks_cpu: torch.Tensor | None = None,
        spec_sequence_masks: torch.Tensor | None = None,
        non_spec_sequence_masks_cpu: torch.Tensor | None = None,
        spec_token_indx: torch.Tensor | None = None,
        non_spec_token_indx: torch.Tensor | None = None,
        stale_spec_reqs: torch.Tensor | None = None,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        aligned_state_indices = getattr(self, "mamba_aligned_state_indices", None)
        if aligned_state_indices is not None:
            block_table_tensor = aligned_state_indices[: m.num_reqs]
        else:
            block_table_tensor = mamba_get_block_table_tensor(
                m.block_table_tensor,
                m.seq_lens,
                self.kv_cache_spec,
                self.vllm_config.cache_config.mamba_cache_mode,
            )

        uniform_spec_sequence_length = None
        spec_rows_lead = False
        if (
            spec_sequence_masks is None
            and non_spec_query_start_loc is None
            and num_decode_draft_tokens_cpu is not None
            and (self.use_spec_decode or self.use_cache_spec_kernel)
        ):
            # V1-runner callers pass only the raw draft/accepted counts; the
            # V2 runner (and build_for_cudagraph_capture) precompute the spec
            # split once per step and always hand over non_spec_query_start_loc,
            # so this fallback never fires there. Without it every spec-verify
            # row would silently take the non-spec path.
            (
                num_prefills,
                num_prefill_tokens,
                num_decodes,
                num_decode_tokens,
                num_spec_decodes,
                num_spec_decode_tokens,
                spec_query_start_loc,
                non_spec_query_start_loc,
                non_spec_query_start_loc_cpu,
                spec_sequence_masks_cpu,
                spec_sequence_masks,
                non_spec_sequence_masks_cpu,
                spec_token_indx,
                non_spec_token_indx,
                num_accepted_tokens,
                stale_spec_reqs,
            ) = compute_common_gdn_attn_metadata(
                num_decode_draft_tokens_cpu,
                num_accepted_tokens,
                query_start_loc,
                query_start_loc_cpu,
                self.num_spec,
                use_cache_spec_kernel=self.use_cache_spec_kernel,
                is_prefilling_cpu=m.is_prefilling,
            )

        if spec_sequence_masks is None:
            query_lens_cpu = query_start_loc_cpu.diff()
            if self.use_cached_kernel:
                # ReplaySSM cached decode: single-token prefill-as-decode rows
                # route to prefill (step-continuity assumption of the ring).
                split_m = m
            else:
                # V2 already excludes prefills from full decode graphs via
                # has_prefill. Classify first chunks as prefills to mask
                # recycled state; resumed one-token chunks can still use the
                # decode kernels.
                assert m.seq_lens_cpu_upper_bound is not None
                no_prior_state = (query_lens_cpu > 0) & (
                    m.seq_lens_cpu_upper_bound <= query_lens_cpu
                )
                # Capture batches also have seq_len == query_len, but are not
                # prefills.
                if m.is_prefilling is not None:
                    no_prior_state &= m.is_prefilling
                else:
                    no_prior_state = torch.zeros_like(no_prior_state)
                split_m = m.replace(is_prefilling=no_prior_state)
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(
                    split_m,
                    decode_threshold=1,
                    treat_short_extends_as_decodes=False,
                )
            )
            # Exclude trailing padding from both prefill counts.
            if num_prefills:
                num_prefills -= int((query_lens_cpu[num_decodes:] == 0).sum())
                num_prefill_tokens = (
                    int(query_start_loc_cpu[num_decodes + num_prefills])
                    - num_decode_tokens
                )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            assert spec_sequence_masks_cpu is not None
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            spec_query_lens_cpu = query_lens_cpu[spec_sequence_masks_cpu]
            if spec_query_lens_cpu.numel() > 0:
                first_spec_sequence_length = int(spec_query_lens_cpu[0])
                if first_spec_sequence_length > 0 and bool(
                    torch.all(spec_query_lens_cpu == first_spec_sequence_length)
                ):
                    uniform_spec_sequence_length = first_spec_sequence_length

            if num_prefills == 0 and num_decodes == 0:
                # #58763 slices on the premise that padded sequences trail the
                # spec decodes; check it on the host mask (CPU only, no sync), as
                # the hoist does for num_accepted_tokens, so the state ids and the
                # accepted counts can never disagree.
                spec_rows_lead = bool(spec_sequence_masks_cpu[:num_spec_decodes].all())
                if spec_rows_lead:
                    # Padded sequences trail the spec decodes, so slice them off
                    # rather than gather with the host mask (an H2D copy + a kernel).
                    spec_state_indices_tensor = block_table_tensor[
                        :num_spec_decodes, : self.num_spec + 1
                    ]
                    if stale_spec_reqs is not None:
                        # The slice is a view of the runner's block table and the
                        # stale null below fills in place: take a copy first.
                        spec_state_indices_tensor = spec_state_indices_tensor.clone()
                else:
                    # A zero-length row sits before a spec row: gather by mask.
                    spec_state_indices_tensor = block_table_tensor[
                        spec_sequence_masks_cpu, : self.num_spec + 1
                    ]
                non_spec_state_indices_tensor = None
            else:
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = block_table_tensor[
                    non_spec_sequence_masks_cpu, 0
                ]
            if stale_spec_reqs is not None:
                # Stale zero-accept rows (async-scheduling discard) have no
                # valid spec state; null their slots so the kernels skip both
                # the initial-state read and the final-state write. Mask and
                # clamped counts come from compute_common_gdn_attn_metadata
                # (carried #51508-subset + #40756 guard, relocated from the
                # pre-#52297 per-group build).
                spec_state_indices_tensor.masked_fill_(
                    stale_spec_reqs.unsqueeze(-1), NULL_BLOCK_ID
                )

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        prefill_query_start_loc: torch.Tensor | None = None
        prefill_state_indices: torch.Tensor | None = None
        prefill_has_initial_state: torch.Tensor | None = None
        aiter_prefill_metadata: object | None = None
        if num_prefills > 0:
            # In a mixed non-spec batch, decodes are peeled off to the recurrent
            # kernel (decode-first front slice), so build chunk metadata from the
            # rebased prefill-only cu_seqlens; otherwise use the full non-spec one.
            # _forward_core keys off the same condition, so they agree.
            if spec_sequence_masks is None and num_decodes > 0:
                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                assert non_spec_state_indices_tensor is not None
                prefill_query_start_loc = (
                    non_spec_query_start_loc[num_decodes:] - num_decode_tokens
                )
                prefill_query_start_loc_cpu = (
                    non_spec_query_start_loc_cpu[num_decodes:] - num_decode_tokens
                )
                prefill_state_indices = non_spec_state_indices_tensor[num_decodes:]
            else:
                prefill_query_start_loc = non_spec_query_start_loc
                prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu
                prefill_state_indices = non_spec_state_indices_tensor

            if self.gdn_prefill_backend == "aiter_flydsl":
                # AITER carries its own reusable varlen metadata and has no use
                # for FLA's chunk indices, so it replaces them rather than
                # extending what _build_chunk_metadata returns.
                assert prefill_query_start_loc_cpu is not None
                aiter_prefill_metadata = (
                    rocm_aiter_ops.build_gdn_flydsl_prefill_metadata(
                        torch.diff(prefill_query_start_loc_cpu).tolist(),
                        cu_seqlens=prefill_query_start_loc,
                    )
                )
            elif self.gdn_prefill_backend != "flashinfer":
                # FlashInfer's chunk kernel consumes cu_seqlens directly and
                # never reads chunk_indices/chunk_offsets, so skip the FLA
                # metadata (CPU compute + H2D copies) entirely.
                chunk_indices, chunk_offsets = self._build_chunk_metadata(
                    prefill_query_start_loc,
                    prefill_query_start_loc_cpu,
                    query_start_loc.device,
                )

        # The ReplaySSM cursor/ring blocks below consume context_lens_tensor
        # on pure-(spec-)decode batches too — including cudagraph capture —
        # so compute it unconditionally (pre-#51674 behaviour) instead of
        # only for prefill batches.
        context_lens_tensor = m.compute_num_computed_tokens()
        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks_cpu is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks_cpu]
                assert non_spec_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
            if spec_sequence_masks is None and num_decodes > 0:
                prefill_has_initial_state = has_initial_state[num_decodes:]
            else:
                prefill_has_initial_state = has_initial_state
        else:
            has_initial_state = None

        checkpoint: MambaPrefillCheckpointMetadata | None = None
        if num_prefills > 0:
            request_rows = list(range(m.num_reqs))
            if spec_sequence_masks_cpu is not None:
                request_rows = (~spec_sequence_masks_cpu).nonzero().flatten().tolist()
            checkpoint = self.checkpoint_builder.build(m, request_rows)

        flashinfer_checkpoint_plan: GDNFlashInferCheckpointPlan | None = None
        if checkpoint is not None and self.plan_flashinfer_checkpoints:
            assert checkpoint.offsets is not None
            assert checkpoint.query_lens is not None
            # The prefill kernel runs the trailing checkpoint rows.
            num_prefill_rows = prefill_query_start_loc_cpu.shape[0] - 1
            every_n_tokens, cu_starts, rows = plan_flashinfer_checkpoints(
                checkpoint.offsets, checkpoint.query_lens[-num_prefill_rows:]
            )
            plan = async_tensor_h2d(
                cu_starts + rows, query_start_loc.device, torch.int64
            )
            flashinfer_checkpoint_plan = GDNFlashInferCheckpointPlan(
                every_n_tokens=every_n_tokens,
                num_checkpoints=cu_starts[-1],
                cu_starts=plan[: len(cu_starts)],
                rows=plan[len(cu_starts) :],
            )

        # Function code counted on either presency non-spec decode or spec decode,
        # but not both.
        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

        # Cached decode kernel: derive the per-request ring write position
        # (write_pos = decode_step % max_cache_len). Only the non-spec decode
        # path runs the cached kernel.
        write_pos_d = None
        spec_write_pos_d = None
        spec_cache_base_d = None
        spec_is_flush_d = None
        spec_first_decode_d = None
        # Capture is skipped: the graph only records the staged buffer's
        # address, and every replay rebuilds the positions from a real batch.
        if (
            self.use_cached_kernel
            and spec_sequence_masks is None
            and num_decodes > 0
            and not self._capturing
        ):
            num_prompt_tokens_cpu = m.num_prompt_tokens_cpu
            num_computed_tokens_cpu = m._num_computed_tokens_cpu
            if num_prompt_tokens_cpu is None or num_computed_tokens_cpu is None:
                raise ValueError(
                    "use_replayssm requires CPU prompt and "
                    "computed-token counts to derive decode write positions"
                )
            decode_steps_cpu = (
                num_computed_tokens_cpu[:num_decodes]
                - num_prompt_tokens_cpu[:num_decodes]
            )
            query_lens_cpu = (
                query_start_loc_cpu[1 : num_decodes + 1]
                - query_start_loc_cpu[:num_decodes]
            )
            valid_decode_rows = query_lens_cpu > 0
            if torch.any(decode_steps_cpu[valid_decode_rows] < 0).item():
                raise ValueError(
                    "use_replayssm requires decode-step counts that "
                    "exclude prompt tokens and start at zero"
                )
            decode_steps_cpu = torch.where(
                valid_decode_rows,
                decode_steps_cpu,
                torch.zeros_like(decode_steps_cpu),
            )
            write_pos_cpu = torch.remainder(decode_steps_cpu, self.max_cache_len)
            write_pos_d = async_tensor_h2d(
                write_pos_cpu.to(torch.int32).tolist(),
                dtype=torch.int32,
                device=query_start_loc.device,
            )

        # Cached-SPEC decode: advance the block-keyed cursors once per step
        # (commit-at-start, using the previous step's num_accepted), then reset
        # first-decode rows. Runs on the UNPADDED spec tensors (block-keyed
        # cursors are skipped for null blocks, so cudagraph padding is fine).
        # The commit/reset run in build() (eager, not in the captured region);
        # the cursors are full (num_gpu_blocks,) fixed-address buffers read by
        # the captured verify kernel.
        # INVARIANT (cudagraph capture): the commit below also runs during
        # capture builds, where the runner's dummy batch keeps zeroed block
        # tables — every row then keys NULL_BLOCK_ID and the commit is a
        # per-row no-op, so capture never advances real cursors. The reset is
        # skipped during capture (num_prompt_tokens_cpu is None there).
        if self.use_cache_spec_kernel and num_spec_decodes > 0:
            from vllm.third_party.flash_linear_attention.ops.gdn_replayssm_spec_decode import (  # noqa: E501
                commit_gdn_replayssm_spec,
                reset_gdn_replayssm_spec_cursors,
            )

            assert spec_state_indices_tensor is not None
            assert num_accepted_tokens is not None
            # non-None whenever num_spec_decodes > 0 (set together above)
            assert spec_sequence_masks_cpu is not None
            self._ensure_spec_cursors()
            sbi = spec_state_indices_tensor[:, 0]
            commit_gdn_replayssm_spec(
                self.spec_write_pos,
                self.spec_cache_base,
                self.spec_is_flush,
                num_accepted_tokens.to(torch.int32),
                sbi,
                max_cache_len=self.spec_flush_threshold,
                max_spec_len=self.max_spec_len,
                cache_buf_len=self.spec_cache_buf_len,
                null_block_id=NULL_BLOCK_ID,
            )
            # prefill->decode reset for first-decode rows (cursors only; conv
            # context lives in conv_state). A request's first spec verify has
            # num_computed_tokens == num_prompt_tokens; that resets its (possibly
            # recycled) block's cursors to write_pos=0.
            num_prompt_tokens_cpu = m.num_prompt_tokens_cpu
            if num_prompt_tokens_cpu is not None:
                num_prompt_d = num_prompt_tokens_cpu.to(
                    context_lens_tensor.device, non_blocking=True
                )
                first_decode_full = (context_lens_tensor == num_prompt_d).to(torch.int8)
                spec_row_idx = spec_sequence_masks_cpu.nonzero(as_tuple=True)[0].to(
                    query_start_loc.device, non_blocking=True
                )
                first_decode_d = first_decode_full.index_select(0, spec_row_idx)
                reset_gdn_replayssm_spec_cursors(
                    self.spec_write_pos,
                    self.spec_cache_base,
                    self.spec_is_flush,
                    first_decode_d,
                    sbi,
                    max_cache_len=self.spec_flush_threshold,
                    max_spec_len=self.max_spec_len,
                    null_block_id=NULL_BLOCK_ID,
                )
                spec_first_decode_d = first_decode_d
            spec_write_pos_d = self.spec_write_pos
            spec_cache_base_d = self.spec_cache_base
            spec_is_flush_d = self.spec_is_flush

        # Prepare per-request tensors for cudagraph. m.num_actual_tokens is
        # token-padded for FULL graph replay, but the GDN state/query/accepted
        # metadata below is indexed by request.
        batch_size = m.num_reqs

        if self._stage_spec_decode(
            num_prefills, num_decodes, num_spec_decodes, num_spec_decode_tokens
        ):
            # The slices below are taken over batch_size, which may exceed
            # num_spec_decodes when zero-length padded requests are present.
            assert batch_size <= self.decode_cudagraph_max_bs
            assert spec_sequence_masks is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if self._stage_decode(num_prefills, num_decodes, num_spec_decodes):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(NULL_BLOCK_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

            if self.use_cached_kernel:
                # write_pos_d is None when the batch has no live decode rows
                # (num_decodes == 0: an all-padding batch); stage an all-zero
                # buffer in that case.
                if write_pos_d is not None:
                    self.decode_write_pos_d[:num_decodes].copy_(
                        write_pos_d, non_blocking=True
                    )
                write_pos_d = self.decode_write_pos_d[:batch_size]
                # The cached decode kernel only runs on all-decode batches
                # (num_prefills == 0, so num_decodes == batch_size); the rows
                # past num_decodes are never read, zero just keeps the staged
                # buffer deterministic.
                write_pos_d[num_decodes:].fill_(0)

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            checkpoint=checkpoint,
            flashinfer_checkpoint_plan=flashinfer_checkpoint_plan,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_state_indices=prefill_state_indices,
            prefill_has_initial_state=prefill_has_initial_state,
            aiter_prefill_metadata=aiter_prefill_metadata,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_sequence_masks_cpu=spec_sequence_masks_cpu,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            write_pos_d=write_pos_d,
            spec_write_pos_d=spec_write_pos_d,
            spec_cache_base_d=spec_cache_base_d,
            spec_is_flush_d=spec_is_flush_d,
            stale_spec_reqs=stale_spec_reqs,
            spec_first_decode_d=spec_first_decode_d,
            spec_rows_lead=spec_rows_lead,
            uniform_spec_sequence_length=uniform_spec_sequence_length,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        return attn_metadata

    def _stage_spec_decode(
        self,
        num_prefills: int,
        num_decodes: int,
        num_spec_decodes: int,
        num_spec_decode_tokens: int,
    ) -> bool:
        """Whether spec-decode metadata goes into the FULL cudagraph buffers."""
        return (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        )

    def _stage_decode(
        self, num_prefills: int, num_decodes: int, num_spec_decodes: int
    ) -> bool:
        """Whether decode metadata goes into the FULL cudagraph buffers."""
        return (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        )

    def _ensure_spec_cursors(self) -> None:
        """Allocate this builder's ReplaySSM-spec block-keyed cursor buffers on
        first use (num_gpu_blocks is unknown at __init__)."""
        if self.spec_write_pos is None:
            n_blocks = self.vllm_config.cache_config.num_gpu_blocks
            assert n_blocks is not None and n_blocks > 0, (
                "--use-replayssm-spec needs num_gpu_blocks at "
                "build time to size the block-keyed cursor buffers"
            )
            self.spec_write_pos = torch.zeros(
                n_blocks, dtype=torch.int32, device=self.cursor_device
            )
            self.spec_cache_base = torch.zeros(
                n_blocks, dtype=torch.int32, device=self.cursor_device
            )
            self.spec_is_flush = torch.zeros(
                n_blocks, dtype=torch.int8, device=self.cursor_device
            )

    def update_block_table(
        self,
        metadata: GDNAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> GDNAttentionMetadata:
        """Re-gather this group's state indices. The other fields are
        batch-level and stay shared with ``metadata``.

        The per-group side effects of build() run here as well, on this
        group's own state ids and buffers, from the batch-level values the
        source group's build() carried in ``metadata``: the stale zero-accept
        null (``stale_spec_reqs``) and, with use_replayssm_spec, the cursor
        commit (``num_accepted_tokens``) and first-decode reset
        (``spec_first_decode_d``) on this builder's block-keyed cursors, which
        the returned metadata then points at.
        """
        m = metadata
        checkpoint = (
            m.checkpoint.regather_state_indices(blk_table)
            if m.checkpoint is not None
            else None
        )
        if self.vllm_config.cache_config.mamba_cache_mode == "align":
            assert self.mamba_aligned_state_indices is not None
            blk_table = self.mamba_aligned_state_indices
        masks = m.spec_sequence_masks_cpu
        spec_indices = non_spec_indices = prefill_indices = None
        if masks is None:
            non_spec_indices = blk_table[:, 0]
            if m.num_prefills > 0:
                prefill_indices = non_spec_indices[m.num_decodes :]
        elif m.num_prefills == 0 and m.spec_rows_lead:
            # Same as build(): padded sequences trail the spec decodes.
            spec_indices = blk_table[: m.num_spec_decodes, : self.num_spec + 1]
        else:
            spec_indices = blk_table[masks, : self.num_spec + 1]
            if m.num_prefills > 0:
                non_spec_indices = prefill_indices = blk_table[~masks, 0]

        spec_reset_indices = None
        if spec_indices is not None:
            # The cursor reset reads the ids from before the stale null, as
            # build() does once BUG-252's reset-window fix is applied: a row
            # that is stale on its first decode keeps its only reset.
            spec_reset_indices = spec_indices[:, 0]
            if m.stale_spec_reqs is not None:
                # Same null as build(), out of place: the slice above is a view
                # of the runner's block table.
                spec_indices = spec_indices.masked_fill(
                    m.stale_spec_reqs.unsqueeze(-1), NULL_BLOCK_ID
                )

        # Local names differ from build()'s on purpose: patchers in the fixes/
        # chain anchor build()'s cursor lines and require them exactly once.
        cursors = (m.spec_write_pos_d, m.spec_cache_base_d, m.spec_is_flush_d)
        if self.use_cache_spec_kernel and m.num_spec_decodes > 0:
            from vllm.third_party.flash_linear_attention.ops.gdn_replayssm_spec_decode import (  # noqa: E501
                commit_gdn_replayssm_spec as commit_spec_cursors,
                reset_gdn_replayssm_spec_cursors as reset_spec_cursors,
            )

            assert spec_indices is not None
            assert m.num_accepted_tokens is not None
            # The cursors are keyed by this group's own block ids, so they live
            # in this builder's buffers, advanced once per step like build().
            self._ensure_spec_cursors()
            cursors = (self.spec_write_pos, self.spec_cache_base, self.spec_is_flush)
            sbi = spec_indices[:, 0]
            commit_spec_cursors(
                *cursors,
                m.num_accepted_tokens[: m.num_spec_decodes].to(torch.int32),
                sbi,
                max_cache_len=self.spec_flush_threshold,
                max_spec_len=self.max_spec_len,
                cache_buf_len=self.spec_cache_buf_len,
                null_block_id=NULL_BLOCK_ID,
            )
            if m.spec_first_decode_d is not None:
                reset_spec_cursors(
                    *cursors,
                    m.spec_first_decode_d,
                    sbi if m.stale_spec_reqs is None else spec_reset_indices,
                    max_cache_len=self.spec_flush_threshold,
                    max_spec_len=self.max_spec_len,
                    null_block_id=NULL_BLOCK_ID,
                )

        if self._stage_spec_decode(
            m.num_prefills, m.num_decodes, m.num_spec_decodes, m.num_spec_decode_tokens
        ):
            assert m.spec_state_indices_tensor is not None
            batch_size = m.spec_state_indices_tensor.shape[0]
            self.spec_state_indices_tensor[: m.num_spec_decodes].copy_(
                spec_indices, non_blocking=True
            )
            spec_indices = self.spec_state_indices_tensor[:batch_size]
            spec_indices[m.num_spec_decodes :].fill_(NULL_BLOCK_ID)

        if self._stage_decode(m.num_prefills, m.num_decodes, m.num_spec_decodes):
            assert m.non_spec_state_indices_tensor is not None
            batch_size = m.non_spec_state_indices_tensor.shape[0]
            self.non_spec_state_indices_tensor[: m.num_decodes].copy_(
                non_spec_indices, non_blocking=True
            )
            non_spec_indices = self.non_spec_state_indices_tensor[:batch_size]
            non_spec_indices[m.num_decodes :].fill_(NULL_BLOCK_ID)

        return replace(
            m,
            spec_state_indices_tensor=spec_indices,
            non_spec_state_indices_tensor=non_spec_indices,
            prefill_state_indices=prefill_indices,
            checkpoint=checkpoint,
            spec_write_pos_d=cursors[0],
            spec_cache_base_d=cursors[1],
            spec_is_flush_d=cursors[2],
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = torch.diff(m.query_start_loc_cpu).sub_(1)
        assert num_decode_draft_tokens_cpu.shape == num_accepted_tokens.shape

        (
            num_prefills,
            num_prefill_tokens,
            num_decodes,
            num_decode_tokens,
            num_spec_decodes,
            num_spec_decode_tokens,
            spec_query_start_loc,
            non_spec_query_start_loc,
            non_spec_query_start_loc_cpu,
            spec_sequence_masks_cpu,
            spec_sequence_masks,
            non_spec_sequence_masks_cpu,
            spec_token_indx,
            non_spec_token_indx,
            num_accepted_tokens,
            stale_spec_reqs,
        ) = compute_common_gdn_attn_metadata(
            num_decode_draft_tokens_cpu,
            num_accepted_tokens,
            m.query_start_loc,
            m.query_start_loc_cpu,
            self.num_spec,
            # ReplaySSM: capture must route through the spec kernel even for
            # q=1 (draft-less T=1 window) batches; the synthetic draft counts
            # alone would send those to the non-spec path.
            use_cache_spec_kernel=self.use_cache_spec_kernel,
            is_prefilling_cpu=m.is_prefilling,
        )

        self._capturing = True
        try:
            return self.build(
                0,
                m,
                num_accepted_tokens,
                num_decode_draft_tokens_cpu,
                num_prefills=num_prefills,
                num_prefill_tokens=num_prefill_tokens,
                num_decodes=num_decodes,
                num_decode_tokens=num_decode_tokens,
                num_spec_decodes=num_spec_decodes,
                num_spec_decode_tokens=num_spec_decode_tokens,
                spec_query_start_loc=spec_query_start_loc,
                non_spec_query_start_loc=non_spec_query_start_loc,
                non_spec_query_start_loc_cpu=non_spec_query_start_loc_cpu,
                spec_sequence_masks_cpu=spec_sequence_masks_cpu,
                spec_sequence_masks=spec_sequence_masks,
                non_spec_sequence_masks_cpu=non_spec_sequence_masks_cpu,
                spec_token_indx=spec_token_indx,
                non_spec_token_indx=non_spec_token_indx,
                stale_spec_reqs=stale_spec_reqs,
            )
        finally:
            self._capturing = False
