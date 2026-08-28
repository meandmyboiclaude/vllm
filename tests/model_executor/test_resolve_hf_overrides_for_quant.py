# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coverage for ``resolve_hf_overrides_for_quant`` (upstream #53881).

The first five cases are upstream's; the rest pin the house requirement that a
callable ``hf_overrides`` is never allowed to mutate the live ``hf_config`` and
that a config-transform return value is reported as "no overrides" instead of
raising.
"""

import pytest

from vllm.model_executor.model_loader.weight_utils import (
    resolve_hf_overrides_for_quant,
)


def test_dict_passthrough():
    assert resolve_hf_overrides_for_quant({"a": 1}) == {"a": 1}


def test_none_becomes_empty():
    assert resolve_hf_overrides_for_quant(None) == {}


def test_callable_with_hf_config():
    def compose(cfg):
        return {"quantization_config_dict_json": cfg}

    assert resolve_hf_overrides_for_quant(compose, hf_config="draft") == {
        "quantization_config_dict_json": "draft"
    }


def test_callable_zero_arg():
    assert resolve_hf_overrides_for_quant(lambda: {"k": 1}) == {"k": 1}


def test_invalid_type_still_raises():
    with pytest.raises(ValueError, match="must be a dict"):
        resolve_hf_overrides_for_quant(["not-a-dict"])


class _FakeConfig:
    """Stand-in for a PretrainedConfig that a transform rewrites in place."""

    def __init__(self, model_type: str):
        self.model_type = model_type


def test_config_transform_does_not_mutate_live_config():
    live = _FakeConfig("qwen3")

    def transform(cfg):
        # What SpeculativeConfig.hf_config_override does: mutate and return.
        cfg.model_type = "qwen3_mtp"
        return cfg

    assert resolve_hf_overrides_for_quant(transform, hf_config=live) == {}
    assert live.model_type == "qwen3", "live hf_config was mutated by a quant lookup"


def test_config_transform_return_is_not_an_error():
    # A PretrainedConfig-shaped return is a transform, not an override map.
    assert resolve_hf_overrides_for_quant(lambda cfg: _FakeConfig("x"), hf_config=None) == {}


def test_one_arg_callable_raising_typeerror_is_called_once():
    calls = []

    def transform(cfg):
        calls.append(cfg)
        raise TypeError("internal failure, not an arity mismatch")

    assert resolve_hf_overrides_for_quant(transform, hf_config=_FakeConfig("q")) == {}
    assert len(calls) == 1, "a one-arg callable was retried as zero-arg"


def test_real_compose_draft_hf_overrides_leaves_the_live_config_alone():
    """The production callable shape: a mutating config transform."""
    from transformers import PretrainedConfig

    from vllm.config.speculative import SpeculativeConfig

    live = PretrainedConfig(
        model_type="deepseek_v3",
        architectures=["DeepseekV3ForCausalLM"],
        num_nextn_predict_layers=1,
    )
    composed = SpeculativeConfig.compose_draft_hf_overrides(None)

    assert resolve_hf_overrides_for_quant(composed, hf_config=live) == {}
    assert live.model_type == "deepseek_v3"
    assert live.architectures == ["DeepseekV3ForCausalLM"]


def test_uncopyable_config_skips_the_call():
    class _Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("nope")

    called = []

    def transform(cfg):
        called.append(cfg)
        return {"a": 1}

    assert resolve_hf_overrides_for_quant(transform, hf_config=_Uncopyable()) == {}
    assert called == []
