# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
)


def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import (
        dflash_has_any_non_causal,
        dflash_target_rope_is_neox_style,
    )

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    # The drafter must rotate Q/K the way its target does. Take that from the
    # built target before super() constructs the draft.
    is_neox_style = dflash_target_rope_is_neox_style(target_model)
    if is_neox_style is not None:
        draft_model_config.hf_config.is_neox_style = is_neox_style
    # Select an attention backend that supports the drafter's attention: mixing
    # a non-causal layer onto a causal-only backend would fail.
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=speculative_config.attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
    )
    with set_model_tag("dflash_head"):
        dflash_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    # MuseGlimmerForCausalLM marks its inner MuseGlimmerModel as the language
    # model, so get_language_model() already returns the inner module and has
    # no .model of its own.
    target_inner = getattr(target_language_model, "model", target_language_model)
    draft_inner = dflash_model.model

    # A DFlash variant whose checkpoint ships no vocabulary weights builds
    # one-row placeholders instead of a full [vocab, hidden] table (DFlash2's
    # `transient_vocab_size = 1`, #53662) and relies on the aliasing below to
    # replace them. Remember whether that is the case so the handoff can be
    # checked afterwards.
    transient_vocab_size = getattr(draft_inner, "transient_vocab_size", None)

    # Skip embedding sharing under PP — each rank owns its own embedding.
    if get_pp_group().world_size == 1:
        target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
            target_inner, "embedding", None
        )
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if target_embed is not None and _should_share(
            dflash_model, "has_own_embed_tokens", draft_embed, target_embed
        ):
            if draft_embed is not None:
                del draft_inner.embed_tokens
            draft_inner.embed_tokens = target_embed

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(dflash_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        dflash_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del dflash_model.lm_head
        dflash_model.lm_head = target_lm_head

    # Post-condition for the transient-vocabulary drafters: every branch above
    # is conditional (no target module found, _should_share declining on a
    # quantized or mismatched copy, PP > 1), and a placeholder that survives
    # one of them is a silent wrong-answer bug — the drafter would embed every
    # token as row 0 and score against a one-token head. Fail loudly here
    # instead. Sizes are checked rather than identity: a variant may legally
    # alias to something other than these exact modules, as long as what it
    # ends up with is not the one-row stub.
    if transient_vocab_size is not None:
        embed = getattr(draft_inner, "embed_tokens", None)
        num_embeddings = getattr(embed, "num_embeddings", None)
        assert num_embeddings is None or num_embeddings > transient_vocab_size, (
            f"{type(dflash_model).__name__} built a {transient_vocab_size}-row "
            "embedding placeholder and nothing replaced it: the target's "
            "embedding was not shared with the drafter."
        )
        head = getattr(dflash_model, "lm_head", None)
        head_rows = getattr(head, "num_embeddings", None)
        assert head_rows is None or head_rows > transient_vocab_size, (
            f"{type(dflash_model).__name__} built a {transient_vocab_size}-row "
            "lm_head placeholder and nothing replaced it: the target's lm_head "
            "was not shared with the drafter."
        )

    return dflash_model
