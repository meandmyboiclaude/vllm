# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coverage of the spec-decode rejection-sampler warmup.

The warmup exists so no first request pays Triton JIT cost, which only holds
if it compiles the kernels the server actually runs: the HAS_DRAFT_LOGITS leg
implied by ``draft_sample_method``, the runtime tensor dtypes, and the
``use_fp64_gumbel`` the sampler is constructed with. It must also never abort
``compile_or_warm_up_model``, allocation failures included.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils as rs_utils
from vllm.model_executor.warmup.spec_decode_rejection_warmup import (
    spec_decode_rejection_warmup,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")

VOCAB_SIZE = 256
NUM_SPEC = 5


def _make_worker(
    *,
    draft_sample_method: str = "greedy",
    rejection_sample_method: str = "standard",
    use_fp64_gumbel: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    head_dtype: torch.dtype | None = None,
) -> SimpleNamespace:
    model_config = SimpleNamespace(
        dtype=dtype,
        head_dtype=head_dtype if head_dtype is not None else dtype,
        use_fp64_gumbel=use_fp64_gumbel,
        get_vocab_size=lambda: VOCAB_SIZE,
    )
    spec_config = SimpleNamespace(
        num_speculative_tokens=NUM_SPEC,
        draft_sample_method=draft_sample_method,
        rejection_sample_method=rejection_sample_method,
    )
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=model_config, speculative_config=spec_config
        )
    )


def _record_calls(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    def fake_rejection_sample(**kwargs):
        calls.append(kwargs)
        return None, None

    monkeypatch.setattr(rs_utils, "rejection_sample", fake_rejection_sample)
    return calls


def test_greedy_warms_the_none_leg_first(monkeypatch):
    """DFlash K=5 runs greedy, so Speculator.draft_logits is None: the
    HAS_DRAFT_LOGITS=False kernels are the ones the first request needs."""
    calls = _record_calls(monkeypatch)
    spec_decode_rejection_warmup(_make_worker(draft_sample_method="greedy"))

    assert calls, "warmup made no rejection_sample calls"
    assert calls[0]["draft_logits"] is None
    legs = {c["draft_logits"] is not None for c in calls}
    assert legs == {True, False}, "both HAS_DRAFT_LOGITS legs must be warmed"


def test_probabilistic_warms_the_draft_logits_leg_first(monkeypatch):
    calls = _record_calls(monkeypatch)
    spec_decode_rejection_warmup(_make_worker(draft_sample_method="probabilistic"))

    assert calls[0]["draft_logits"] is not None
    legs = {c["draft_logits"] is not None for c in calls}
    assert legs == {True, False}


def test_runtime_dtypes_are_matched(monkeypatch):
    """Every index tensor must carry the dtype the GPU runner passes, since
    Triton specializes on the pointer dtype (input_ids int32, positions int64,
    cu_num_logits/idx_mapping/expanded_* int32, temperature fp32, seed int64)."""
    calls = _record_calls(monkeypatch)
    spec_decode_rejection_warmup(_make_worker())

    expected = {
        "draft_sampled": torch.int32,
        "cu_num_logits": torch.int32,
        "pos": torch.int64,
        "idx_mapping": torch.int32,
        "expanded_idx_mapping": torch.int32,
        "expanded_local_pos": torch.int32,
        "temperature": torch.float32,
        "seed": torch.int64,
    }
    for call in calls:
        for name, dtype in expected.items():
            assert call[name].dtype is dtype, f"{name}: {call[name].dtype} != {dtype}"


@pytest.mark.parametrize("use_fp64_gumbel", [False, True])
def test_use_fp64_follows_model_config(monkeypatch, use_fp64_gumbel: bool):
    calls = _record_calls(monkeypatch)
    spec_decode_rejection_warmup(_make_worker(use_fp64_gumbel=use_fp64_gumbel))

    assert calls
    assert all(c["use_fp64"] is use_fp64_gumbel for c in calls)


def test_head_dtype_is_covered(monkeypatch):
    """draft_logits carry Speculator.draft_logits_spec() -> head_dtype."""
    calls = _record_calls(monkeypatch)
    spec_decode_rejection_warmup(
        _make_worker(
            draft_sample_method="probabilistic",
            dtype=torch.bfloat16,
            head_dtype=torch.float16,
        )
    )

    draft_dtypes = {
        c["draft_logits"].dtype for c in calls if c["draft_logits"] is not None
    }
    assert torch.float16 in draft_dtypes
    target_dtypes = {c["target_logits"].dtype for c in calls}
    assert {torch.float16, torch.bfloat16, torch.float32} <= target_dtypes


def test_allocation_failure_does_not_propagate(monkeypatch):
    """Tensor allocation sits inside the try: an OOM here must not abort
    compile_or_warm_up_model."""
    _record_calls(monkeypatch)
    real_zeros = torch.zeros

    def boom(*args, **kwargs):
        if kwargs.get("device") is not None:
            raise torch.cuda.OutOfMemoryError("simulated")
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", boom)
    spec_decode_rejection_warmup(_make_worker())


def test_kernel_failure_does_not_propagate(monkeypatch):
    def fake_rejection_sample(**kwargs):
        raise RuntimeError("simulated Triton failure")

    monkeypatch.setattr(rs_utils, "rejection_sample", fake_rejection_sample)
    spec_decode_rejection_warmup(_make_worker())


@pytest.mark.parametrize("draft_sample_method", ["greedy", "probabilistic"])
def test_warmup_compiles_real_kernels(caplog, draft_sample_method: str):
    """End-to-end: the real Triton kernels must accept every warmed case.
    A silent bail (the warning path) would leave the JIT cost on the first
    request, so treat any warning as a failure."""
    import logging

    with caplog.at_level(logging.WARNING):
        spec_decode_rejection_warmup(
            _make_worker(draft_sample_method=draft_sample_method)
        )
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warnings, [r.getMessage() for r in warnings]


def test_no_speculative_config_is_a_noop(monkeypatch):
    calls = _record_calls(monkeypatch)
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(model_config=None, speculative_config=None)
    )
    spec_decode_rejection_warmup(worker)
    assert calls == []
