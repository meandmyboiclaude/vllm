# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A held Qwen3 ``<tool_call>`` opener through the served composition.

``vllm serve --reasoning-parser qwen3 --tool-call-parser qwen3_coder`` holds
a ``<tool_call>`` until ``<function=`` shows it opens a real call. These cases
pin what reaches the client when that hold spans deltas: an implicit-end call
must not leak its opener into content, and an opener cut off by EOS is a
truncated call, dropped.
"""

import json
from unittest.mock import MagicMock

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.engine.registered_adapters import (
    Qwen3ParserReasoningAdapter,
    Qwen3ParserToolAdapter,
)

_SPECIAL = {
    "<think>": 50,
    "</think>": 51,
    "<tool_call>": 60,
    "</tool_call>": 61,
    "<|im_start|>": 70,
    "<|im_end|>": 71,
}
# One token per character for ordinary text, as in a served token-ID stream.
_CHARS = {chr(c): 1000 + c for c in (10, *range(32, 127))}
_PROMPT = [_SPECIAL["<|im_start|>"]]


class _Qwen3DelegatingParser(DelegatingParser):
    reasoning_parser_cls = Qwen3ParserReasoningAdapter
    tool_parser_cls = Qwen3ParserToolAdapter


_TOOLS = [
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    )
]


def _request():
    req = MagicMock(spec=ChatCompletionRequest)
    req.tools = _TOOLS
    req.tool_choice = "auto"
    req.include_reasoning = True
    return req


def _tokens(text: str) -> list[str]:
    out, i = [], 0
    while i < len(text):
        special = next((t for t in _SPECIAL if text.startswith(t, i)), None)
        out.append(special or text[i])
        i += len(out[-1])
    return out


def _parser():
    tokenizer = make_mock_tokenizer(
        {**_SPECIAL, **_CHARS}, special_tokens=list(_SPECIAL)
    )
    return _Qwen3DelegatingParser(tokenizer)


def _parse(text: str):
    parser = _parser()
    parser.set_prompt_token_ids(_PROMPT)
    reasoning, content, calls = parser.parse(text, _request(), enable_auto_tools=True)
    return reasoning, content, [(c.name, json.loads(c.arguments)) for c in calls or []]


def _stream(text: str, chunk_size: int, with_token_ids: bool):
    """Deltas of ``chunk_size`` tokens (special tokens never split), then EOS."""
    parser = _parser()
    request = _request()
    tokens = _tokens(text)
    chunks = [
        "".join(tokens[i : i + chunk_size]) for i in range(0, len(tokens), chunk_size)
    ]
    chunks.append("<|im_end|>")
    reasoning, content, calls = "", "", {}
    for i, chunk in enumerate(chunks):
        ids = [_SPECIAL.get(t) or _CHARS[t] for t in _tokens(chunk)]
        delta = parser.parse_delta(
            chunk,
            ids if with_token_ids else [],
            request,
            prompt_token_ids=_PROMPT,
            finished=i == len(chunks) - 1,
        )
        if delta is None:
            continue
        reasoning += delta.reasoning or ""
        content += delta.content or ""
        for tc in delta.tool_calls or []:
            slot = calls.setdefault(tc.index, ["", ""])
            slot[0] += (tc.function and tc.function.name) or ""
            slot[1] += (tc.function and tc.function.arguments) or ""
    return reasoning, content, [(n, json.loads(a)) for n, a in calls.values()]


_CHUNK_SIZES = [1, 2, 3, 4, 5, 6, 7, 1000]
_CALL = (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n"
    "</parameter>\n</function>\n</tool_call>"
)


@pytest.mark.parametrize("chunk_size", _CHUNK_SIZES)
@pytest.mark.parametrize("with_token_ids", [True, False])
def test_implicit_end_call_does_not_leak_its_opener(chunk_size, with_token_ids):
    """The opener and ``<function=`` arrive in different deltas for every
    chunk size below the call length."""
    text = "Let me check.\n" + _CALL
    reasoning, content, calls = _stream(text, chunk_size, with_token_ids)
    assert reasoning == "Let me check.\n"
    assert content == ""
    assert calls == [("get_weather", {"city": "Tokyo"})]

    reasoning, content, calls = _parse(text)
    assert reasoning == "Let me check.\n"
    assert content is None
    assert calls == [("get_weather", {"city": "Tokyo"})]


@pytest.mark.parametrize("chunk_size", _CHUNK_SIZES)
@pytest.mark.parametrize("with_token_ids", [True, False])
def test_opener_cut_off_by_eos_is_dropped(chunk_size, with_token_ids):
    text = "Done.</think>DONE\n<tool_call>\n"
    reasoning, content, calls = _stream(text, chunk_size, with_token_ids)
    assert reasoning == "Done."
    assert content.strip() == "DONE"
    assert calls == []

    reasoning, content, calls = _parse(text)
    assert reasoning == "Done."
    assert content.strip() == "DONE"
    assert calls == []


@pytest.mark.parametrize("chunk_size", _CHUNK_SIZES)
@pytest.mark.parametrize("with_token_ids", [True, False])
def test_quoted_opener_in_reasoning_keeps_text_and_answer(chunk_size, with_token_ids):
    text = "I should mention `<tool_call>` literally.</think>The answer is 42."
    reasoning, content, calls = _stream(text, chunk_size, with_token_ids)
    assert reasoning == "I should mention `<tool_call>` literally."
    assert content == "The answer is 42."
    assert calls == []


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("Thinking.\n<tool_call>\n", id="opener-cut-off"),
        pytest.param(
            "Thinking.\n<tool_call>\n<function", id="partial-function-cut-off"
        ),
        pytest.param("Thinking.\n<tool_call>\n</tool_call>", id="empty-block"),
    ],
)
@pytest.mark.parametrize("chunk_size", _CHUNK_SIZES)
@pytest.mark.parametrize("with_token_ids", [True, False])
def test_truncated_or_empty_implicit_end_block_does_not_leak(
    text, chunk_size, with_token_ids
):
    """An opener met in reasoning that never reaches a function name is no
    call and none of its markup reaches the content. (Streaming keeps a cut-off
    opener in reasoning, non-streaming drops it: unchanged from before.)"""
    reasoning, content, calls = _stream(text, chunk_size, with_token_ids)
    assert reasoning.startswith("Thinking.\n")
    assert content == ""
    assert calls == []

    reasoning, content, calls = _parse(text)
    assert reasoning.startswith("Thinking.\n")
    assert content is None
    assert calls == []


# Stacked on #59821: with the prompt ending in an open <think>, a <tool_call>
# met in reasoning is held by #59821's reasoning holder, not by the #56661
# holder above, so the EOS rule never resolves it.
_OPEN_THINK_PROMPT = [_SPECIAL["<|im_start|>"], _SPECIAL["<think>"]]
_QUOTED_CUT = (
    "Example: <tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n"
    "</parameter>\n</function>\n</tool_call> is the format, and then"
)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(_QUOTED_CUT, id="quoted-call-then-cut"),
        pytest.param("Thinking.\n<tool_call>\n", id="opener-then-cut"),
    ],
)
@pytest.mark.parametrize("chunk_size", _CHUNK_SIZES)
@pytest.mark.parametrize("with_token_ids", [True, False])
def test_armed_reasoning_span_cut_by_eos_stays_reasoning(
    text, chunk_size, with_token_ids, monkeypatch
):
    """A quoted call in thinking, then EOS without </think>: no call, no
    markup in content, the whole span stays reasoning."""
    monkeypatch.setattr(
        "tests.parser.engine.test_qwen3_tool_opener_handoff._PROMPT",
        _OPEN_THINK_PROMPT,
    )
    reasoning, content, calls = _stream(text, chunk_size, with_token_ids)
    assert (reasoning, content, calls) == (text, "", [])

    reasoning, content, calls = _parse(text)
    assert (reasoning, content, calls) == (text, None, [])
