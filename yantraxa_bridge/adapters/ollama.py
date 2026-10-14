"""The `ollama` adapter: a local model behind Ollama's `/api/chat`, streaming.

The request constrains the answer to `prompt.output_schema()` through `format`, which
asks for an optional advisory `vote` with a `support`/`oppose`/`abstain` choice and a
reason. Deltas
of the answer's `text` field stream as chunks (decoded, without the JSON around them);
the whole answer is parsed at the end. Malformed output gets one retry, which doesn't
stream, and then the turn passes. Server errors and an unreachable Ollama pass at once.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..prompt import build_messages, output_schema, parse_output
from .base import AdapterInfo, Chunk, Final, TurnContext, TurnEvent, pass_turn

log = logging.getLogger("yantraxa_bridge.adapters.ollama")

RUNTIME_LABEL = "Ollama on owner's server"
RETRY_PROMPT = (
    "That answer was not one valid JSON object in the required shape. "
    "Answer again with only the JSON object."
)


class OllamaError(Exception):
    """Ollama answered with an error; retrying won't help."""


class OllamaAdapter:
    def __init__(
        self,
        host: str,
        model: str,
        *,
        timeout_s: float,
        max_tokens: int,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.url = host.rstrip("/") + "/api/chat"
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self._http = http

    def describe(self) -> AdapterInfo:
        return AdapterInfo(
            RUNTIME_LABEL, ("message", "question", "reply", "proposal", "vote"), streams=True
        )

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_s, connect=min(10.0, self.timeout_s))
            )
        return self._http

    async def take_turn(self, ctx: TurnContext) -> AsyncIterator[TurnEvent]:
        messages, refs = build_messages(ctx)
        for attempt in (1, 2):
            text = TextStream() if attempt == 1 else None
            raw = ""
            try:
                async for delta in self._chat(messages):
                    raw += delta
                    piece = text.feed(delta) if text else ""
                    if piece:
                        yield Chunk(piece)
            except (OllamaError, httpx.HTTPError) as error:
                log.warning("ollama turn failed (%s); passing", _describe(error))
                yield pass_turn()
                return
            action = parse_output(raw, ctx, refs)
            if action is not None:
                yield Final([action])
                return
            log.warning("ollama answer %d was not valid JSON for the schema", attempt)
            messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": RETRY_PROMPT},
            ]
        yield pass_turn()

    async def _chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "format": output_schema(),
            "options": {"num_predict": self.max_tokens},
        }
        async with self._client().stream("POST", self.url, json=body) as response:
            if response.status_code != 200:
                await response.aread()
                raise OllamaError(f"HTTP {response.status_code}: {_error_text(response.text)}")
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    raise OllamaError("the response is not Ollama's NDJSON") from None
                if not isinstance(data, dict):
                    raise OllamaError("the response is not Ollama's NDJSON")
                if "error" in data:
                    raise OllamaError(str(data["error"])[:200])
                message = data.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    yield message["content"]
                if data.get("done"):
                    if data.get("done_reason") not in (None, "stop"):
                        log.info("ollama stopped early: %s", data.get("done_reason"))
                    return


def _error_text(body: str) -> str:
    try:
        data = json.loads(body)
    except ValueError:
        return body[:200]
    return str(data.get("error", body))[:200] if isinstance(data, dict) else body[:200]


def _describe(error: Exception) -> str:
    if isinstance(error, httpx.TimeoutException):
        return "timed out"
    return str(error) or type(error).__name__


_INCOMPLETE_ESCAPE = re.compile(r"(?<!\\)(?:\\\\)*\\(?:u[0-9a-fA-F]{0,3})?$")
_HIGH_SURROGATE = re.compile(r"(?<!\\)(?:\\\\)*\\u[dD][89abAB][0-9a-fA-F]{2}$")


class TextStream:
    """Decodes the top-level `"text"` string of a JSON object as it streams in.

    `feed` returns the newly decoded text. Nothing comes out until `action` is known
    and isn't `pass`, so a pass never shows as typing.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._i = 0
        self._depth = 0
        self._in_string = False
        self._escaped = False
        self._string_start = 0
        self._is_key = False
        self._expect_key = False
        self._key: str | None = None
        self._text_start: int | None = None
        self._text_end: int | None = None
        self._emitted = 0
        self.action: str | None = None

    def feed(self, delta: str) -> str:
        self._buf += delta
        self._scan()
        return self._emit()

    def _scan(self) -> None:
        buf = self._buf
        while self._i < len(buf):
            c = buf[self._i]
            if self._in_string:
                if self._escaped:
                    self._escaped = False
                elif c == "\\":
                    self._escaped = True
                elif c == '"':
                    self._in_string = False
                    self._string_done(buf[self._string_start : self._i])
            elif c == '"':
                self._in_string = True
                self._string_start = self._i + 1
                self._is_key = self._depth == 1 and self._expect_key
                if (
                    self._depth == 1
                    and not self._is_key
                    and self._key == "text"
                    and self._text_start is None
                ):
                    self._text_start = self._i + 1
            elif c in "{[":
                self._depth += 1
                self._expect_key = c == "{" and self._depth == 1
            elif c in "}]":
                self._depth -= 1
            elif self._depth == 1 and c == ":":
                self._expect_key = False
            elif self._depth == 1 and c == ",":
                self._expect_key = True
            self._i += 1

    def _string_done(self, raw: str) -> None:
        if self._depth != 1:
            return
        if self._is_key:
            self._key = _decode(raw)
        elif self._key == "action" and self.action is None:
            self.action = _decode(raw)
        elif self._key == "text" and self._text_end is None and self._text_start is not None:
            self._text_end = self._i

    def _emit(self) -> str:
        if self.action is None or self.action == "pass" or self._text_start is None:
            return ""
        end = self._text_end if self._text_end is not None else len(self._buf)
        segment = self._buf[self._text_start : end]
        if self._text_end is None:
            segment = _HIGH_SURROGATE.sub("", segment)
            segment = _INCOMPLETE_ESCAPE.sub("", segment)
        decoded = _decode(segment)
        if decoded is None:
            return ""
        piece = decoded[self._emitted :]
        self._emitted = len(decoded)
        return piece


def _decode(raw: str) -> Any:
    try:
        return json.loads(f'"{raw}"', strict=False)
    except ValueError:
        return None
