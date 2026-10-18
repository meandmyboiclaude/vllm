# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Markup quoted inside a Qwen3 tool argument survives the reasoning pass.

``vllm serve --reasoning-parser qwen3 --tool-call-parser qwen3_coder`` runs a
reasoning pass (``skip_tool_parsing``) before the tool pass. The tool span it
forwards must reach the tool pass byte for byte, whatever markup a parameter
value quotes, in non-streaming and in streaming of any delta size.
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
_OPEN_THINK_PROMPT = [_SPECIAL["<|im_start|>"], _SPECIAL["<think>"]]


class _Qwen3DelegatingParser(DelegatingParser):
    reasoning_parser_cls = Qwen3ParserReasoningAdapter
    tool_parser_cls = Qwen3ParserToolAdapter


def _tool(name: str, *params: str) -> ChatCompletionToolsParam:
    return ChatCompletionToolsParam(
        type="function",
        function={
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {p: {"type": "string"} for p in params},
            },
        },
    )


_TOOLS = [
    _tool("echo_payload", "text"),
    _tool("write_file", "path", "content"),
    _tool("get_weather", "city"),
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
    parser.set_prompt_token_ids(_OPEN_THINK_PROMPT)
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
            prompt_token_ids=_OPEN_THINK_PROMPT,
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


_PAYLOAD = "alpha <think>literal</think> omega"
_THINK_MARKERS_IN_VALUE = (
    "Reason.\n</think>\n<tool_call>\n<function=echo_payload>\n"
    f"<parameter=text>\n{_PAYLOAD}\n</parameter>\n</function>\n</tool_call>"
)
# An odd number of fences inside one value, then a second call.
_FENCE_IN_VALUE = (
    "Reason.\n</think>\n<tool_call>\n<function=write_file>\n"
    "<parameter=path>\na.md\n</parameter>\n<parameter=content>\nx ``` y\n"
    "</parameter>\n</function>\n</tool_call>\n<tool_call>\n"
    "<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n"
    "</function>\n</tool_call>"
)
_CASES = [
    pytest.param(
        _THINK_MARKERS_IN_VALUE,
        [("echo_payload", {"text": _PAYLOAD})],
        id="think-markers-in-value",
    ),
    pytest.param(
        _FENCE_IN_VALUE,
        [
            ("write_file", {"path": "a.md", "content": "x ``` y"}),
            ("get_weather", {"city": "Tokyo"}),
        ],
        id="odd-fence-in-value-then-call",
    ),
]


@pytest.mark.parametrize("text, expected_calls", _CASES)
def test_non_streaming(text, expected_calls):
    reasoning, content, calls = _parse(text)
    assert reasoning == "Reason.\n"
    assert content is None
    assert calls == expected_calls


_CHUNK_SIZES = [1, 2, 3, 4, 5, 6, 7, 1000]


@pytest.mark.parametrize("text, expected_calls", _CASES)
@pytest.mark.parametrize("chunk_size", _CHUNK_SIZES)
@pytest.mark.parametrize("with_token_ids", [True, False])
def test_streaming(text, expected_calls, chunk_size, with_token_ids):
    reasoning, _, calls = _stream(text, chunk_size, with_token_ids)
    assert reasoning == "Reason.\n"
    assert calls == expected_calls


def _content_params():
    for case_id, text in (
        ("think-markers-in-value", _THINK_MARKERS_IN_VALUE),
        ("odd-fence-in-value-then-call", _FENCE_IN_VALUE),
    ):
        for chunk_size in _CHUNK_SIZES:
            for with_token_ids in (True, False):
                marks = ()
                if text is _THINK_MARKERS_IN_VALUE and (chunk_size, with_token_ids) == (
                    1000,
                    True,
                ):
                    marks = pytest.mark.xfail(
                        strict=True,
                        reason="separate defect: the tool pass gets the token ids "
                        "after the LAST </think> of the delta (the quoted one), "
                        "so its strict token-id mode demotes <tool_call> to content",
                    )
                yield pytest.param(
                    text,
                    chunk_size,
                    with_token_ids,
                    marks=marks,
                    id=f"{case_id}-{chunk_size}-{'ids' if with_token_ids else 'text'}",
                )


@pytest.mark.parametrize("text, chunk_size, with_token_ids", _content_params())
def test_streaming_content(text, chunk_size, with_token_ids):
    _, content, _ = _stream(text, chunk_size, with_token_ids)
    assert content.strip() == ""
