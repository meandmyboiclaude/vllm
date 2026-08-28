# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GDNAttentionMetadataBuilder.build() — specifically the
reclassification of non-spec decodes as prefills when spec decodes exist.
Covers the fix for https://github.com/vllm-project/vllm/issues/34845.
"""

from dataclasses import dataclass

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu.attn_utils import compute_common_gdn_attn_metadata

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


@dataclass
class GDNBuildTestCase:
    """Specification for a GDN metadata builder classification test."""

    seq_lens: list[int]
    query_lens: list[int]
    num_decode_draft_tokens: list[int] | None  # None = no spec config
    num_speculative_tokens: int
    expected_num_decodes: int
    expected_num_prefills: int
    expected_num_prefill_tokens: int
    expected_num_spec_decodes: int


GDN_BUILD_TEST_CASES = {
    # The original #34845 crash: non-spec query_len=1 + spec decode
    "mixed_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[65, 20],
        query_lens=[1, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
    # All requests are spec decodes — no reclassification needed
    "pure_spec_decode": GDNBuildTestCase(
        seq_lens=[50, 30],
        query_lens=[3, 3],
        num_decode_draft_tokens=[2, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=2,
    ),
    # No speculative config at all — standard decode path
    "pure_regular_decode": GDNBuildTestCase(
        seq_lens=[40, 30, 20],
        query_lens=[1, 1, 1],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=3,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=0,
    ),
    # Multi-token prefill alongside spec decode — no decode to reclassify
    "spec_decode_with_real_prefill": GDNBuildTestCase(
        seq_lens=[100, 20],
        query_lens=[50, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=1,
    ),
    # All three types in one batch — decode gets reclassified
    "prefill_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[100, 65, 20],
        query_lens=[50, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=2,
        expected_num_prefill_tokens=51,
        expected_num_spec_decodes=1,
    ),
    # Multiple non-spec query_len=1 requests all reclassified
    "multiple_decodes_reclassified": GDNBuildTestCase(
        seq_lens=[40, 50, 60, 20],
        query_lens=[1, 1, 1, 3],
        num_decode_draft_tokens=[-1, -1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=3,
        expected_num_prefill_tokens=3,
        expected_num_spec_decodes=1,
    ),
    # Zero-length padded sequence excluded from counts
    "zero_length_padding_with_spec": GDNBuildTestCase(
        seq_lens=[16, 65, 20],
        query_lens=[0, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
}


def _create_gdn_builder(
    num_speculative_tokens: int = 0,
    full_cuda_graph: bool = False,
) -> GDNAttentionMetadataBuilder:
    """Create a GDNAttentionMetadataBuilder with minimal config."""
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
    )
    if full_cuda_graph:
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    if num_speculative_tokens > 0:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
    )
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _build(
    builder: GDNAttentionMetadataBuilder,
    batch_spec: BatchSpec,
    num_decode_draft_tokens: list[int] | None = None,
    num_accepted_tokens: list[int] | None = None,
    is_prefilling: list[bool] | None = None,
    common: CommonAttentionMetadata | None = None,
) -> GDNAttentionMetadata:
    """Build GDN attention metadata, optionally with spec-decode kwargs."""
    if common is None:
        common = create_common_attn_metadata(batch_spec, BLOCK_SIZE, DEVICE)
    if is_prefilling is not None:
        common = common.replace(
            is_prefilling=torch.tensor(is_prefilling, dtype=torch.bool)
        )
    kwargs: dict = {}
    if num_decode_draft_tokens is not None:
        num_decode_draft_tokens_cpu = torch.tensor(
            num_decode_draft_tokens, dtype=torch.int32
        )
        if num_accepted_tokens is None:
            num_accepted_tokens = torch.ones(
                batch_spec.batch_size, dtype=torch.int32, device=DEVICE
            )
        else:
            num_accepted_tokens = torch.tensor(
                num_accepted_tokens, dtype=torch.int32, device=DEVICE
            )
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
            common.query_start_loc,
            common.query_start_loc_cpu,
            builder.num_spec,
        )
        kwargs["num_decode_draft_tokens_cpu"] = num_decode_draft_tokens_cpu
        kwargs["num_accepted_tokens"] = num_accepted_tokens
        kwargs["num_prefills"] = num_prefills
        kwargs["num_prefill_tokens"] = num_prefill_tokens
        kwargs["num_decodes"] = num_decodes
        kwargs["num_decode_tokens"] = num_decode_tokens
        kwargs["num_spec_decodes"] = num_spec_decodes
        kwargs["num_spec_decode_tokens"] = num_spec_decode_tokens
        kwargs["spec_query_start_loc"] = spec_query_start_loc
        kwargs["non_spec_query_start_loc"] = non_spec_query_start_loc
        kwargs["non_spec_query_start_loc_cpu"] = non_spec_query_start_loc_cpu
        kwargs["spec_sequence_masks_cpu"] = spec_sequence_masks_cpu
        kwargs["spec_sequence_masks"] = spec_sequence_masks
        kwargs["non_spec_sequence_masks_cpu"] = non_spec_sequence_masks_cpu
        kwargs["spec_token_indx"] = spec_token_indx
        kwargs["non_spec_token_indx"] = non_spec_token_indx
        kwargs["stale_spec_reqs"] = stale_spec_reqs
    return builder.build(common_prefix_len=0, common_attn_metadata=common, **kwargs)


@pytest.mark.parametrize(
    "test_case", GDN_BUILD_TEST_CASES.values(), ids=GDN_BUILD_TEST_CASES.keys()
)
def test_gdn_build_classification(test_case: GDNBuildTestCase):
    """Test that GDN metadata builder classifies requests correctly."""
    builder = _create_gdn_builder(test_case.num_speculative_tokens)
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    meta = _build(builder, batch, test_case.num_decode_draft_tokens)

    assert meta.num_decodes == test_case.expected_num_decodes
    assert meta.num_prefills == test_case.expected_num_prefills
    assert meta.num_prefill_tokens == test_case.expected_num_prefill_tokens
    assert meta.num_spec_decodes == test_case.expected_num_spec_decodes


def test_has_initial_state_after_reclassification():
    """After reclassification, num_prefills > 0 so the prefill kernel path
    should compute has_initial_state. For the reclassified request with
    context_lens > 0, the corresponding entry must be True."""
    builder = _create_gdn_builder(num_speculative_tokens=2)
    batch = BatchSpec(seq_lens=[65, 20], query_lens=[1, 3])
    meta = _build(builder, batch, num_decode_draft_tokens=[-1, 2])

    assert meta.num_prefills > 0, "reclassification should produce prefills"
    assert meta.has_initial_state is not None
    # req0 has context_lens = 65 - 1 = 64 > 0, so has_initial_state[0] = True
    assert meta.has_initial_state[0].item() is True


@pytest.mark.parametrize("full_cuda_graph", [False, True])
def test_zero_accepted_tokens_nulls_state_slots(full_cuda_graph: bool):
    """A stale row (num_accepted_tokens == 0, produced when async scheduling
    discards a step's sampled tokens) must have its whole state-index row set
    to NULL_BLOCK_ID so the kernels skip both the initial-state read and the
    final-state write, and must have its count clamped into range.

    Nulling — rather than only clamping — is what stops the kernel from
    advancing the recurrent state of a request whose step was discarded.
    Live rows in the same batch must be left untouched.
    """
    num_speculative_tokens = 3
    builder = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens,
        full_cuda_graph=full_cuda_graph,
    )
    batch = BatchSpec(seq_lens=[80, 96], query_lens=[4, 4])
    # NULL_BLOCK_ID is 0 and the helper draws random block ids from [0, 1000),
    # so a live row can contain a genuine 0; pin the block table to non-null
    # ids to keep the "live row untouched" check deterministic.
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE).replace(
        block_table_tensor=torch.arange(1, 13, dtype=torch.int32).reshape(2, 6)
    )
    # Row 0 is stale (its sampled tokens were discarded); row 1 is live.
    meta = _build(
        builder,
        batch,
        num_decode_draft_tokens=[3, 3],
        num_accepted_tokens=[0, 2],
        common=common,
    )

    assert meta.spec_state_indices_tensor is not None
    assert meta.num_accepted_tokens is not None

    stale_row = meta.spec_state_indices_tensor[0]
    live_row = meta.spec_state_indices_tensor[1]

    assert (stale_row == NULL_BLOCK_ID).all(), (
        "every state slot of a stale row must be nulled so the kernel's "
        f"state_idx <= 0 guard skips it, got {stale_row.tolist()}"
    )
    assert not (live_row == NULL_BLOCK_ID).any(), (
        f"a live row must keep its state slots, got {live_row.tolist()}"
    )

    # The clamp keeps the slot index (num_accepted_tokens - 1) in bounds; the
    # live row's real count must survive it.
    assert meta.num_accepted_tokens[0].item() == 1
    assert meta.num_accepted_tokens[1].item() == 2
    assert (meta.num_accepted_tokens >= 1).all()


def test_zero_accepted_tokens_does_not_corrupt_block_table():
    """Nulling stale rows must not write NULL_BLOCK_ID back into the block
    table the runner reuses across steps."""
    builder = _create_gdn_builder(num_speculative_tokens=3)
    batch = BatchSpec(seq_lens=[80, 96], query_lens=[4, 4])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    block_table_before = common.block_table_tensor.clone()

    # Go through _build: since #52297 the spec split is precomputed by the
    # caller, so a bare build() with only the two spec kwargs leaves
    # spec_sequence_masks None and never reaches the nulling code at all.
    _build(
        builder,
        batch,
        num_decode_draft_tokens=[3, 3],
        num_accepted_tokens=[0, 2],
        common=common,
    )

    torch.testing.assert_close(common.block_table_tensor, block_table_before)


def test_one_token_first_chunk_is_prefill():
    """A first chunk has no recurrent state for the decode path to read."""
    builder = _create_gdn_builder()
    batch = BatchSpec(seq_lens=[100, 50, 1], query_lens=[1, 1, 1])
    meta = _build(builder, batch, is_prefilling=[False, False, True])

    assert meta.num_decodes == 2
    assert meta.num_decode_tokens == 2
    assert meta.num_prefills == 1
    assert meta.num_prefill_tokens == 1
    assert meta.has_initial_state is not None
    assert meta.has_initial_state.tolist() == [True, True, False]
    assert meta.prefill_has_initial_state is not None
    assert meta.prefill_has_initial_state.tolist() == [False]


def test_one_token_resumed_prefill_stays_decode():
    """A one-token chunk with prior recurrent state can use the decode path."""
    builder = _create_gdn_builder()
    batch = BatchSpec(seq_lens=[100, 50, 65], query_lens=[1, 1, 1])
    meta = _build(builder, batch, is_prefilling=[False, False, True])

    assert meta.num_decodes == 3
    assert meta.num_prefills == 0
    assert meta.has_initial_state is None


def test_cudagraph_capture_batch_stays_decode_only():
    """Capture dummies look stateless but are not real prefill requests."""
    builder = _create_gdn_builder(full_cuda_graph=True)
    batch = BatchSpec(seq_lens=[1] * 4, query_lens=[1] * 4)
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE).replace(
        is_prefilling=torch.zeros(4, dtype=torch.bool)
    )
    meta = builder.build_for_cudagraph_capture(common)

    assert meta.num_decodes == 4
    assert meta.num_prefills == 0
    assert meta.has_initial_state is None


@pytest.mark.parametrize(
    ("seq_lens", "is_prefilling", "expected_num_prefills"),
    [
        pytest.param([100, 50, 1], [False, False, True], 1, id="mixed-with-decodes"),
        pytest.param([1, 1, 1], [True, True, True], 3, id="all-stateless"),
    ],
)
def test_one_token_prefill_batch_stages_cudagraph_metadata(
    seq_lens: list[int],
    is_prefilling: list[bool],
    expected_num_prefills: int,
):
    """FULL-graph dispatch is shape-based, so uniform metadata must be staged."""
    builder = _create_gdn_builder(full_cuda_graph=True)
    batch = BatchSpec(seq_lens=seq_lens, query_lens=[1, 1, 1])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE).replace(
        is_prefilling=torch.tensor(is_prefilling)
    )
    meta = builder.build(0, common)

    assert meta.num_prefills == expected_num_prefills
    assert meta.non_spec_state_indices_tensor is not None
    assert meta.non_spec_query_start_loc is not None
    assert (
        meta.non_spec_state_indices_tensor.data_ptr()
        == builder.non_spec_state_indices_tensor.data_ptr()
    )
    assert (
        meta.non_spec_query_start_loc.data_ptr()
        == builder.non_spec_query_start_loc.data_ptr()
    )
    torch.testing.assert_close(meta.non_spec_query_start_loc, common.query_start_loc)


def test_one_token_prefill_excludes_cudagraph_padding():
    """Padding rows must not enter the prefill chunk metadata."""
    builder = _create_gdn_builder(full_cuda_graph=True)
    batch = BatchSpec(seq_lens=[100, 50, 1, 0], query_lens=[1, 1, 1, 0])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE).replace(
        is_prefilling=torch.tensor([False, False, True, False]),
        num_actual_tokens=4,
    )
    meta = builder.build(0, common)

    assert meta.num_decodes == 2
    assert meta.num_decode_tokens == 2
    assert meta.num_prefills == 1
    assert meta.num_prefill_tokens == 1
    assert meta.has_initial_state is not None
    # has_initial_state is batch-indexed and rides the staged batch_size-wide
    # arrays into causal_conv1d_fn, which asserts size == (padded_batch,);
    # the padding row is state-less, hence the trailing False.
    assert meta.has_initial_state.tolist() == [True, True, False, False]
    # It is read inside the captured region, so it must ride the builder's
    # persistent staging buffer, not a per-step allocation (capture#2).
    assert meta.has_initial_state.data_ptr() == builder.has_initial_state_buf.data_ptr()
    assert meta.prefill_query_start_loc is not None
    assert meta.prefill_query_start_loc.tolist() == [0, 1]
    assert meta.prefill_state_indices is not None
    assert meta.prefill_state_indices.shape == (1,)
    assert meta.prefill_has_initial_state is not None
    assert meta.prefill_has_initial_state.tolist() == [False]
    assert meta.non_spec_state_indices_tensor is not None
    assert meta.non_spec_state_indices_tensor.shape == (4,)
    torch.testing.assert_close(
        meta.non_spec_state_indices_tensor, common.block_table_tensor[:, 0]
    )
    assert meta.non_spec_query_start_loc is not None
    torch.testing.assert_close(meta.non_spec_query_start_loc, common.query_start_loc)


def test_multi_token_prefill_batch_does_not_stage_cudagraph_metadata():
    """A non-uniform batch cannot replay the captured uniform-decode graph."""
    builder = _create_gdn_builder(full_cuda_graph=True)
    batch = BatchSpec(seq_lens=[100, 50], query_lens=[1, 2])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE).replace(
        is_prefilling=torch.tensor([False, True])
    )
    meta = builder.build(0, common)

    assert meta.non_spec_state_indices_tensor is not None
    assert (
        meta.non_spec_state_indices_tensor.data_ptr()
        != builder.non_spec_state_indices_tensor.data_ptr()
    )


def test_first_chunk_without_prefill_flag_keeps_length_classification():
    """Without the runner flag, capture-shaped metadata is ambiguous."""
    builder = _create_gdn_builder()
    batch = BatchSpec(seq_lens=[1], query_lens=[1])
    meta = _build(builder, batch)

    assert meta.num_decodes == 1
    assert meta.num_prefills == 0
    assert meta.has_initial_state is None


def test_full_cudagraph_spec_metadata_uses_request_count():
    """FULL cudagraph token padding must not pad request-indexed metadata."""
    num_speculative_tokens = 3
    builder = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens,
        full_cuda_graph=True,
    )
    batch = BatchSpec(seq_lens=[80, 96], query_lens=[4, 4])
    meta = _build(builder, batch, num_decode_draft_tokens=[3, 3])

    assert meta.num_spec_decodes == batch.batch_size
    assert meta.num_spec_decode_tokens == batch.compute_num_tokens()
    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.shape == (
        batch.batch_size,
        num_speculative_tokens + 1,
    )
    assert meta.spec_sequence_masks is not None
    assert meta.spec_sequence_masks.shape == (batch.batch_size,)
    assert meta.spec_query_start_loc is not None
    assert meta.spec_query_start_loc.shape == (batch.batch_size + 1,)
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.shape == (batch.batch_size,)
