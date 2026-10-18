# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tool markup inside Qwen3 reasoning, through the served composition.

``vllm serve --reasoning-parser qwen3 --tool-call-parser qwen3_coder``, with
the prompt ending in an open ``<think>`` block, holds a ``<tool_call>`` met in
reasoning. After ``</think>`` the next non-whitespace item decides: an answer
or another call means the span was quoted and stays reasoning; nothing but
whitespace up to EOS means it was the turn's call (#39056). Without
``</think>``, a span that ends in a closed call is an implicit-end call.
"""

import json
import logging
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


def _tool(name: str, param: str) -> ChatCompletionToolsParam:
    return ChatCompletionToolsParam(
        type="function",
        function={
            "name": name,
            "parameters": {"type": "object", "properties": {param: {"type": "string"}}},
        },
    )


# The quoted name is offered, so name validation alone cannot hide a call.
_TOOLS = [_tool("terminal", "command"), _tool("read_file", "path")]


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
    calls = [(c.name, json.loads(c.arguments)) for c in calls or []]
    return reasoning or "", (content or "").strip(), calls


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
    calls = [(n, json.loads(a)) for n, a in calls.values()]
    return reasoning, content.strip(), calls


_QUOTED = (
    "Example: <tool_call>\n<function=terminal>\n<parameter=command>\n"
    "rm -rf /tmp/build\n</parameter>\n</function>\n</tool_call> is the format."
)
_CALL = (
    "<tool_call>\n<function=read_file>\n<parameter=path>\n/etc/hosts\n"
    "</parameter>\n</function>\n</tool_call>"
)
_READ = ("read_file", {"path": "/etc/hosts"})

_CASES = [
    pytest.param(
        _QUOTED + "</think>Done.", _QUOTED, "Done.", [], id="quoted-then-answer"
    ),
    pytest.param(
        _QUOTED + "</think>\n" + _CALL, _QUOTED, "", [_READ], id="quoted-then-real-call"
    ),
    pytest.param(
        "Reading it.\n" + _CALL + "</think>",
        "Reading it.\n",
        "",
        [_READ],
        id="call-then-think-end-39056",
    ),
    pytest.param(
        "Reading it.\n" + _CALL + "\n</think>\n\n",
        "Reading it.\n",
        "",
        [_READ],
        id="call-then-think-end-and-whitespace",
    ),
    pytest.param(
        "Reading it.\n" + _CALL, "Reading it.\n", "", [_READ], id="implicit-end-call"
    ),
    pytest.param(
        _QUOTED + " More thinking",
        _QUOTED + " More thinking",
        "",
        [],
        id="quoted-then-cut-off",
    ),
    pytest.param(
        "Thinking.\n<tool_call>\n<function=terminal>\n<parameter=command>\nrm",
        "Thinking.\n<tool_call>\n<function=terminal>\n<parameter=command>\nrm",
        "",
        [],
        id="call-cut-off-in-reasoning",
    ),
]


@pytest.mark.parametrize("text, reasoning, content, calls", _CASES)
def test_non_streaming(text, reasoning, content, calls):
    assert _parse(text) == (reasoning, content, calls)


@pytest.mark.parametrize("text, reasoning, content, calls", _CASES)
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 5, 6, 7, 1000])
@pytest.mark.parametrize("with_token_ids", [True, False])
def test_streaming(text, reasoning, content, calls, chunk_size, with_token_ids):
    assert _stream(text, chunk_size, with_token_ids) == (reasoning, content, calls)


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_guard_logs_one_info_line_per_quoted_span():
    engine_logger = logging.getLogger("vllm.parser.engine.streaming_parser_engine")
    handler = _Records()
    previous_level = engine_logger.level
    engine_logger.addHandler(handler)
    engine_logger.setLevel(logging.INFO)
    try:
        _stream(_QUOTED + "</think>Done.", 3, True)
        _stream("Reading it.\n" + _CALL + "</think>", 3, True)
    finally:
        engine_logger.removeHandler(handler)
        engine_logger.setLevel(previous_level)
    guard = [r for r in handler.records if "[PC2]" in r.getMessage()]
    assert len(guard) == 1
    assert guard[0].levelno == logging.INFO
    assert guard[0].getMessage().startswith(
        "[PC2] quoted tool span held as reasoning, no tool call emitted"
    )
    assert "\n" not in guard[0].getMessage()


def test_prompt_without_open_think_leaves_streaming_unchanged():
    """The hold is armed only by a prompt ending in an open think block."""
    parser = _parser()
    request = _request()
    delta = parser.parse_delta(
        "Thinking.\n<tool_call>",
        [*(_CHARS[c] for c in "Thinking.\n"), _SPECIAL["<tool_call>"]],
        request,
        prompt_token_ids=[_SPECIAL["<|im_start|>"]],
        finished=False,
    )
    reasoning_engine = parser._reasoning_parser._parser_engine._engine
    assert reasoning_engine._pending_reasoning_tool_tokens is None
    assert delta is not None and delta.reasoning == "Thinking.\n"
