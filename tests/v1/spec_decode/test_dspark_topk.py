# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead


def _markov_head(weight: torch.Tensor) -> DSparkMarkovHead:
    head = DSparkMarkovHead.__new__(DSparkMarkovHead)
    nn.Module.__init__(head)
    head.markov_w2 = nn.Linear(
        weight.shape[1], weight.shape[0], bias=False, dtype=weight.dtype
    )
    head.markov_w2.weight.data.copy_(weight)
    return head


def test_gathered_markov_bias_overwrites_dense_logits():
    weight = torch.arange(21, dtype=torch.float32).view(7, 3) / 10
    markov_embed = torch.tensor([[0.5, -1.0, 0.25], [1.0, 0.5, -0.5]])
    logits = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        ]
    )
    values, index = logits.topk(3, dim=-1)
    values = torch.stack((values, torch.zeros_like(values)), dim=1)[:, 0]
    expected = values + torch.bmm(weight[index], markov_embed.unsqueeze(-1)).squeeze(-1)
    logits.fill_(float("-inf"))

    result = _markov_head(weight).apply_bias_gathered(
        markov_embed, logits, values, index
    )

    assert result is logits
    torch.testing.assert_close(result.gather(1, index), expected)
    selected = torch.zeros_like(result, dtype=torch.bool).scatter_(1, index, True)
    assert torch.isneginf(result.masked_select(~selected)).all()


def test_gathered_markov_bias_matches_dense_at_full_vocab():
    weight = torch.arange(15, dtype=torch.float32).view(5, 3) / 10
    markov_embed = torch.tensor([[0.5, -1.0, 0.25]])
    logits = torch.tensor([[0.1, 0.4, -0.2, 0.3, 0.0]])
    original = logits.clone()
    values, index = logits.topk(logits.shape[-1], dim=-1)
    scale = 0.5
    logits.fill_(float("-inf"))

    result = _markov_head(weight).apply_bias_gathered(
        markov_embed, logits, values, index, scale
    )

    expected = original + markov_embed @ weight.T * scale
    torch.testing.assert_close(result, expected)


def test_markov_embed_clamps_warmup_sentinel_ids():
    """The anchor read can carry -1 during warmup / cudagraph capture."""
    head = DSparkMarkovHead.__new__(DSparkMarkovHead)
    nn.Module.__init__(head)
    head.markov_w1 = nn.Embedding(4, 3)

    # In range: an identity.
    ids = torch.tensor([0, 2, 3])
    torch.testing.assert_close(head.embed(ids), head.markov_w1(ids))

    # Out of range in both directions: clamped onto the edge rows instead of
    # gathering off the end of the codebook.
    out_of_range = head.embed(torch.tensor([-1, 9]))
    torch.testing.assert_close(out_of_range, head.markov_w1(torch.tensor([0, 3])))


def test_skip_drafts_reset_clears_stale_confidences():
    """The skip-drafts return samples nothing, but the runner still records."""
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    speculator = DSparkSpeculator.__new__(DSparkSpeculator)
    speculator.enable_adaptive_verification = True
    speculator.draft_token_confidence_probs = torch.full((4, 3), 0.25)

    speculator._reset_draft_side_outputs(2)

    # Only the active rows are reset; 1.0 is the no-measurement value.
    assert speculator.draft_token_confidence_probs[:2].eq(1.0).all()
    assert speculator.draft_token_confidence_probs[2:].eq(0.25).all()

    # Adaptive verification off: nothing consumes the buffer, nothing to do.
    speculator.enable_adaptive_verification = False
    speculator.draft_token_confidence_probs.fill_(0.25)
    speculator._reset_draft_side_outputs(2)
    assert speculator.draft_token_confidence_probs.eq(0.25).all()

    # The base speculator publishes no such buffer.
    base = DFlashSpeculator.__new__(DFlashSpeculator)
    assert base._reset_draft_side_outputs(2) is None
