# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""load_dflash_model's transient-vocabulary post-condition.

A DFlash variant whose checkpoint carries no vocabulary weights (DFlash2,
`transient_vocab_size = 1`) builds one-row embedding / lm_head placeholders
and depends on load_dflash_model aliasing the target's modules over them.
Every aliasing branch is conditional, so a placeholder can survive — with no
error, and with the drafter embedding every token as row 0.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch.nn as nn

import vllm.compilation.backends as compilation_backends
from vllm.model_executor.models import qwen3_dflash
from vllm.v1.worker.gpu.spec_decode.dflash import utils as dflash_utils


class _VocabModule(nn.Module):
    def __init__(self, num_embeddings: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings


def _make_draft(transient_vocab_size: int | None) -> nn.Module:
    inner = nn.Module()
    inner.transient_vocab_size = transient_vocab_size
    inner.embed_tokens = _VocabModule(1 if transient_vocab_size else 1000)
    draft = nn.Module()
    draft.model = inner
    draft.lm_head = _VocabModule(1 if transient_vocab_size else 1000)
    return draft


def _make_target(*, with_embed: bool = True, with_lm_head: bool = True) -> nn.Module:
    inner = nn.Module()
    if with_embed:
        inner.embed_tokens = _VocabModule(151936)
    target = nn.Module()
    target.model = inner
    if with_lm_head:
        target.lm_head = _VocabModule(151936)
    return target


@pytest.fixture
def patched_loader(monkeypatch):
    def _apply(draft: nn.Module):
        monkeypatch.setattr(dflash_utils, "get_model", lambda **_: draft)
        monkeypatch.setattr(
            dflash_utils, "get_pp_group", lambda: SimpleNamespace(world_size=1)
        )
        monkeypatch.setattr(dflash_utils, "replace", lambda obj, **_: obj)
        monkeypatch.setattr(
            compilation_backends, "set_model_tag", lambda _: nullcontext()
        )
        monkeypatch.setattr(qwen3_dflash, "dflash_has_any_non_causal", lambda _: False)
        monkeypatch.setattr(
            qwen3_dflash, "dflash_target_rope_is_neox_style", lambda _: None
        )
        return SimpleNamespace(
            speculative_config=SimpleNamespace(
                draft_model_config=SimpleNamespace(hf_config=SimpleNamespace()),
                attention_backend=None,
                kv_cache_dtype=None,
            ),
            attention_config=SimpleNamespace(),
            cache_config=SimpleNamespace(),
        )

    return _apply


def test_transient_vocab_placeholders_are_replaced(patched_loader):
    draft = _make_draft(transient_vocab_size=1)
    vllm_config = patched_loader(draft)
    target = _make_target()

    loaded = dflash_utils.load_dflash_model(target, vllm_config)

    assert loaded.model.embed_tokens is target.model.embed_tokens
    assert loaded.lm_head is target.lm_head


def test_unreplaced_embedding_placeholder_fails_loudly(patched_loader):
    # A target that exposes no embedding under either name: the aliasing
    # branch is skipped and the one-row placeholder would survive.
    draft = _make_draft(transient_vocab_size=1)
    vllm_config = patched_loader(draft)
    target = _make_target(with_embed=False)

    with pytest.raises(AssertionError, match="embedding placeholder"):
        dflash_utils.load_dflash_model(target, vllm_config)


def test_unreplaced_lm_head_placeholder_fails_loudly(patched_loader):
    draft = _make_draft(transient_vocab_size=1)
    vllm_config = patched_loader(draft)
    target = _make_target(with_lm_head=False)

    with pytest.raises(AssertionError, match="lm_head placeholder"):
        dflash_utils.load_dflash_model(target, vllm_config)


def test_full_vocab_drafter_is_not_checked(patched_loader):
    # transient_vocab_size is None for DFlash/DSpark: those checkpoints carry
    # their own vocabulary, so a declined share is legitimate.
    draft = _make_draft(transient_vocab_size=None)
    vllm_config = patched_loader(draft)
    target = _make_target(with_embed=False, with_lm_head=False)

    loaded = dflash_utils.load_dflash_model(target, vllm_config)

    assert loaded.model.embed_tokens.num_embeddings == 1000
    assert loaded.lm_head.num_embeddings == 1000
