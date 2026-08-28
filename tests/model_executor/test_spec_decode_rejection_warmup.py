# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the spec-decode rejection-sampler Triton warmup.

The warmup must (a) call the MRV2 ``rejection_sample`` entry point with the
same dtypes the runner (``vllm/v1/worker/gpu/``) uses at serve time and
(b) actually compile on a real GPU for the served configuration, so no
first request pays the JIT cost.
"""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.warmup import spec_decode_rejection_warmup as warm_mod


def _make_worker(
    *,
    num_spec: int = 5,
    rejection_sample_method: str = "standard",
    draft_sample_method: str = "greedy",
    dtype: torch.dtype = torch.bfloat16,
    head_dtype: torch.dtype | None = None,
    use_fp64_gumbel: bool = False,
    vocab_size: int = 151936,
):
    spec_config = SimpleNamespace(
        num_speculative_tokens=num_spec,
        rejection_sample_method=rejection_sample_method,
        draft_sample_method=draft_sample_method,
    )
    model_config = SimpleNamespace(
        get_vocab_size=lambda: vocab_size,
        dtype=dtype,
        head_dtype=dtype if head_dtype is None else head_dtype,
        use_fp64_gumbel=use_fp64_gumbel,
    )
    vllm_config = SimpleNamespace(
        speculative_config=spec_config, model_config=model_config
    )
    return SimpleNamespace(vllm_config=vllm_config)


def test_no_spec_config_is_noop():
    worker = SimpleNamespace(vllm_config=SimpleNamespace(speculative_config=None))
    with patch.object(warm_mod, "torch") as fake_torch:
        warm_mod.spec_decode_rejection_warmup(worker)
        fake_torch.zeros.assert_not_called()


def test_dtypes_match_mrv2_runner_buffers():
    """The Triton specialization key includes pointer dtypes; the warmup must
    pass what the MRV2 runner passes (input_ids int32, positions int64,
    idx_mapping/expanded_idx_mapping int64 per input_batch.py, local_pos
    int32, temperature fp32, seeds int64)."""
    calls: list[dict] = []

    def fake_rejection_sample(**kw):
        calls.append(kw)
        return None, None

    worker = _make_worker(draft_sample_method="greedy")
    with (
        patch(
            "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils.rejection_sample",
            fake_rejection_sample,
        ),
        # Allocations go to "cuda" by name; redirect to CPU for the dtype check.
        patch.object(warm_mod.torch, "device", lambda *_: "cpu"),
    ):
        warm_mod.spec_decode_rejection_warmup(worker)

    assert calls, "warmup did not reach rejection_sample"
    # Greedy DFlash serves draft_logits=None first, then the fallback leg.
    assert calls[0]["draft_logits"] is None
    assert any(c["draft_logits"] is not None for c in calls)
    for kw in calls:
        assert kw["draft_sampled"].dtype == torch.int32
        assert kw["pos"].dtype == torch.int64
        assert kw["idx_mapping"].dtype == torch.int64
        assert kw["expanded_idx_mapping"].dtype == torch.int64
        assert kw["expanded_local_pos"].dtype == torch.int32
        assert kw["temperature"].dtype == torch.float32
        assert kw["seed"].dtype == torch.int64
        assert kw["cu_num_logits"].dtype == torch.int32
        assert kw["num_speculative_steps"] == 5
        assert kw["target_logits"].shape == (6, 151936)
        assert kw["use_block_verification"] is False
        assert kw["synthetic_conditional_rates"] is None
    target_dtypes = {c["target_logits"].dtype for c in calls}
    assert target_dtypes == {torch.bfloat16, torch.float32}


def test_block_and_synthetic_flags_forwarded():
    calls: list[dict] = []

    def fake_rejection_sample(**kw):
        calls.append(kw)
        return None, None

    worker = _make_worker(rejection_sample_method="block", use_fp64_gumbel=True)
    with (
        patch(
            "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils.rejection_sample",
            fake_rejection_sample,
        ),
        patch.object(warm_mod.torch, "device", lambda *_: "cpu"),
    ):
        warm_mod.spec_decode_rejection_warmup(worker)
    assert calls and all(c["use_block_verification"] for c in calls)
    assert all(c["use_fp64"] for c in calls)

    calls.clear()
    worker = _make_worker(rejection_sample_method="synthetic")
    with (
        patch(
            "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils.rejection_sample",
            fake_rejection_sample,
        ),
        patch.object(warm_mod.torch, "device", lambda *_: "cpu"),
    ):
        warm_mod.spec_decode_rejection_warmup(worker)
    assert calls
    for c in calls:
        assert c["synthetic_conditional_rates"] is not None
        assert c["synthetic_conditional_rates"].shape == (5,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("draft_sample_method", ["greedy", "probabilistic"])
@pytest.mark.parametrize("rejection_sample_method", ["standard", "block"])
def test_warmup_compiles_on_gpu(
    caplog, draft_sample_method: str, rejection_sample_method: str
):
    """End-to-end: the served DFlash K=5 configuration must compile every
    leg without hitting the warning/early-return path."""
    worker = _make_worker(
        draft_sample_method=draft_sample_method,
        rejection_sample_method=rejection_sample_method,
    )
    with caplog.at_level(logging.WARNING, logger=warm_mod.logger.name):
        warm_mod.spec_decode_rejection_warmup(worker)
    torch.cuda.synchronize()
    skipped = [r for r in caplog.records if "Skipping" in r.getMessage()]
    assert not skipped, [r.getMessage() for r in skipped]
