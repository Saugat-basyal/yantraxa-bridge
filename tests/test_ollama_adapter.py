import json

import httpx
import pytest
import respx

from yantraxa_bridge.adapters.base import Chunk, Final
from yantraxa_bridge.adapters.ollama import OllamaAdapter, TextStream
from yantraxa_bridge.prompt import output_schema

from .contexts import M2, P1, make_context

HOST = "http://ollama.test:11434"
CHAT = f"{HOST}/api/chat"


def ndjson(content: str, pieces: int = 7, done_reason: str = "stop") -> bytes:
    """Ollama's streaming body: `content` split into deltas, then a done line."""
    size = max(1, len(content) // pieces)
    deltas = [content[i : i + size] for i in range(0, len(content), size)]
    lines = [
        {"model": "llama3.1:8b", "message": {"role": "assistant", "content": d}, "done": False}
        for d in deltas
    ]
    lines.append(
        {
            "model": "llama3.1:8b",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": done_reason,
        }
    )
    return "\n".join(json.dumps(line) for line in lines).encode() + b"\n"


GOOD = json.dumps(
    {
        "action": "reply",
        "reply_to": "m2",
        "text": 'Yes: "the roof" first,\nthen seeds.',
        "vote": {"motion_id": "p1", "choice": "yes"},
    }
)


def adapter(**kw) -> OllamaAdapter:
    return OllamaAdapter(HOST, "llama3.1:8b", timeout_s=5, max_tokens=123, **kw)


async def collect(adapter: OllamaAdapter, ctx=None):
    events = [event async for event in adapter.take_turn(ctx or make_context())]
    await adapter.aclose()
    return events


@respx.mock
async def test_streams_the_text_and_returns_the_action():
    route = respx.post(CHAT).mock(return_value=httpx.Response(200, content=ndjson(GOOD)))
    events = await collect(adapter())
    chunks = [e.text for e in events if isinstance(e, Chunk)]
    assert len(chunks) > 1
    assert "".join(chunks) == 'Yes: "the roof" first,\nthen seeds.'
    final = events[-1]
    assert isinstance(final, Final)
    [action] = final.actions
    assert action.action == "reply"
    assert action.reply_to == M2
    assert action.vote is not None and action.vote.motion_id == P1

    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "llama3.1:8b"
    assert body["stream"] is True
    assert body["format"] == output_schema()
    assert body["options"]["num_predict"] == 123
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


@respx.mock
async def test_a_pass_streams_nothing():
    content = json.dumps({"action": "pass", "reply_to": None, "text": "meh", "vote": None})
    respx.post(CHAT).mock(return_value=httpx.Response(200, content=ndjson(content)))
    events = await collect(adapter())
    assert not [e for e in events if isinstance(e, Chunk)]
    assert events[-1].actions[0].action == "pass"


@respx.mock
async def test_malformed_output_is_retried_once_without_streaming():
    route = respx.post(CHAT).mock(
        side_effect=[
            httpx.Response(200, content=ndjson('{"action": "message", "text": "cut', 3, "length")),
            httpx.Response(200, content=ndjson(GOOD)),
        ]
    )
    events = await collect(adapter())
    assert route.call_count == 2
    retry = json.loads(route.calls[1].request.content)["messages"]
    assert [m["role"] for m in retry] == ["system", "user", "assistant", "user"]
    assert "json" in retry[-1]["content"].lower()
    # Only the first attempt streams; the retry's text arrives in the final action.
    assert "".join(e.text for e in events if isinstance(e, Chunk)) == "cut"
    assert events[-1].actions[0].action == "reply"


@respx.mock
async def test_malformed_twice_passes():
    route = respx.post(CHAT).mock(
        return_value=httpx.Response(200, content=ndjson("I think we should fix the roof."))
    )
    events = await collect(adapter())
    assert route.call_count == 2
    assert next(e.actions for e in events if isinstance(e, Final))[0].action == "pass"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(404, json={"error": "model 'llama3.1:8b' not found"}),
        httpx.Response(200, content=b'{"error": "out of memory"}\n'),
        httpx.Response(200, content=b"<html>proxy error</html>\n"),
    ],
)
@respx.mock
async def test_server_errors_pass_without_retrying(response):
    route = respx.post(CHAT).mock(return_value=response)
    events = await collect(adapter())
    assert route.call_count == 1
    assert events == [Final([events[-1].actions[0]])]
    assert events[-1].actions[0].action == "pass"


@respx.mock
async def test_unreachable_ollama_passes():
    respx.post(CHAT).mock(side_effect=httpx.ConnectError("refused"))
    events = await collect(adapter())
    assert events[-1].actions[0].action == "pass"


def test_describe():
    info = adapter().describe()
    assert info.runtime_label == "Ollama on owner's server"
    assert info.streams is True


# ------------------------------------------------------------------ TextStream


def feed_all(pieces):
    stream = TextStream()
    return "".join(stream.feed(p) for p in pieces)


def test_text_stream_decodes_escapes_split_across_deltas():
    pieces = [
        '{"action": "mess',
        'age", "reply_to": null, "te',
        'xt": "caf\\u00',
        "e9 \\",
        '"ok\\',
        "n",
        '" , "vote": null}',
    ]
    assert feed_all(pieces) == 'café "ok\n'


def test_text_stream_waits_for_the_action_and_ignores_a_pass():
    assert feed_all(['{"action": "pass", "text": "hidden"}']) == ""
    stream = TextStream()
    assert stream.feed('{"text": "early", ') == ""
    assert stream.feed('"action": "message"}') == "early"


def test_text_stream_ignores_text_inside_other_values():
    pieces = ['{"action": "message", "reply_to": "\\"text\\": \\"no\\"", "text": "yes"}']
    assert feed_all(pieces) == "yes"
