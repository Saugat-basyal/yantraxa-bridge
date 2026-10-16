"""Turns through an adapter: activity states, streamed chunks, redaction, pass on failure.

For each `turn_request` the handler sends `activity reading_agenda`, builds the context,
sends `preparing`, then runs the adapter. `responding` goes out with the first chunk, or
just before a non-pass answer from an adapter that doesn't stream; a pass never claims
to be responding. Timeouts and exceptions are the client's (`client._run_turn` passes).

Streamed text is redacted too. Chunks are coalesced (one per `chunk_interval_s` at
most) and only text up to the last whitespace goes out, so a key split across deltas
is whole when the redactor sees it. A PEM header, which contains spaces, is held back
until it is complete. Every chunk sent is a prefix of the redacted text; if that ever
stops being true (a custom pattern spanning words), streaming stops for the turn and
the final answer still arrives in full.

An action the turn doesn't allow (`TurnContext.allowed_actions`) never goes out, whatever
the model answered: it becomes a `message` with the same text when messages are allowed,
and a pass otherwise. `downgraded` is told why, for the transparency log.

A project request (0.2.0) runs the same adapter on a project context, without streaming
(there is no chunk frame for projects): the first spoken action's text, redacted and cut
to the request's `max_chars`, is the reply. A pass says why: the model chose it
(`declined`), the owner's caps stopped it (`limit`), or the adapter failed (`error`).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from ..client import ProjectAnswer, TurnIO
from ..protocol import ProjectRequestPayload, TurnAction, TurnRequestPayload
from ..redaction import Redactor
from .base import MAX_TEXT_CHARS, Adapter, Chunk, Final, TurnContext
from .context import build_context, build_project_context

log = logging.getLogger("yantraxa_bridge.adapters.handler")

PASS = TurnAction(action="pass")
MAX_ACTIONS = 5
_PEM_START = "-----BEGIN"
_PEM_HEADER = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")


class RedactedStream:
    """Raw streamed text in, redacted deltas out."""

    def __init__(self, redactor: Redactor, limit: int) -> None:
        self.redactor = redactor
        self.limit = limit
        self.raw = ""
        self.sent = ""
        self.broken = False

    def add(self, text: str) -> None:
        self.raw = (self.raw + text)[: self.limit]

    def flush(self, final: bool = False) -> str:
        if self.broken:
            return ""
        end = len(self.raw) if final else self._safe_end()
        redacted = self.redactor.redact(self.raw[:end])
        if not redacted.startswith(self.sent):
            log.info("streamed text no longer matches its redaction; streaming stopped")
            self.broken = True
            return ""
        delta = redacted[len(self.sent) :]
        self.sent = redacted
        return delta

    def _safe_end(self) -> int:
        raw = self.raw
        end = max(raw.rfind(" "), raw.rfind("\n"), raw.rfind("\t")) + 1
        pem = raw.rfind(_PEM_START, 0, end)
        if pem != -1 and not _PEM_HEADER.match(raw, pem):
            end = pem
        return end


class AdapterTurnHandler:
    def __init__(
        self,
        adapter: Adapter,
        redactor: Redactor,
        *,
        timeout_s: float,
        chunk_interval_s: float = 0.75,
    ) -> None:
        self.adapter = adapter
        self.redactor = redactor
        self.timeout_s = timeout_s
        self.chunk_interval_s = chunk_interval_s

    async def aclose(self) -> None:
        await self.adapter.aclose()

    async def take_turn(self, request: TurnRequestPayload, io: TurnIO) -> list[TurnAction]:
        if io.meeting is None:
            log.warning(
                "turn %s is for a meeting this bridge hasn't joined; passing", request.turn_id
            )
            return [PASS]
        await io.activity("reading_agenda")
        ctx = build_context(request, io.meeting, io.client, timeout_s=self.timeout_s)
        await io.activity("preparing")

        loop = asyncio.get_running_loop()
        stream = RedactedStream(self.redactor, ctx.max_output_chars)
        last_flush = loop.time()
        responding = False
        answer: list[TurnAction] | None = None
        events = self.adapter.take_turn(ctx)
        try:
            async for event in events:
                if isinstance(event, Final):
                    answer = event.actions
                    break
                if isinstance(event, Chunk) and event.text:
                    if not responding:
                        await io.activity("responding")
                        responding = True
                    stream.add(event.text)
                    if loop.time() - last_flush >= self.chunk_interval_s:
                        last_flush = loop.time()
                        if piece := stream.flush():
                            await io.chunk(piece)
        finally:
            aclose = getattr(events, "aclose", None)
            if aclose is not None:
                await aclose()

        actions = self._finish(self._allowed(answer, ctx), ctx.max_output_chars)
        if actions[0].action != "pass":
            if not responding:
                await io.activity("responding")
            elif piece := stream.flush(final=True):
                await io.chunk(piece)
        return actions

    async def take_project_turn(
        self, request: ProjectRequestPayload, agent: dict[str, Any] | None
    ) -> ProjectAnswer:
        ctx = build_project_context(request, agent, timeout_s=self.timeout_s)
        final: Final | None = None
        events = self.adapter.take_turn(ctx)
        try:
            async for event in events:
                if isinstance(event, Final):
                    final = event
                    break
        finally:
            aclose = getattr(events, "aclose", None)
            if aclose is not None:
                await aclose()
        if final is None:
            return ProjectAnswer(reason="error")
        actions = self._finish(self._allowed(final.actions, ctx), ctx.max_output_chars)
        if actions[0].action == "pass":
            return ProjectAnswer(reason=final.pass_reason or "declined")
        return ProjectAnswer(text=actions[0].text)

    def _allowed(self, answer: list[TurnAction] | None, ctx: TurnContext) -> list[TurnAction]:
        """`answer` with every action the turn doesn't allow made a message or a pass."""
        allowed = ctx.allowed_actions
        kept = []
        for action in answer or []:
            if action.action in allowed:
                kept.append(action)
                continue
            if "message" in allowed:
                kept.append(action.model_copy(update={"action": "message", "reply_to": None}))
                reason = f"{action.action} isn't allowed this turn; sent as a message"
            else:
                reason = f"{action.action} isn't allowed this turn; passed"
            log.info("turn %s: %s", ctx.turn_id, reason)
            self.downgraded(ctx, action, reason)
        return kept

    def downgraded(self, ctx: TurnContext, action: TurnAction, reason: str) -> None:
        """Called for each action `_allowed` changed; the logged handler records it."""

    def _finish(self, answer: list[TurnAction] | None, limit: int) -> list[TurnAction]:
        """Redacted, non-empty actions; a pass when nothing usable is left."""
        spoken = []
        for action in (answer or [])[:MAX_ACTIONS]:
            if action.action == "pass":
                continue
            # Redacted before the cut, so a secret the cut would split is still found.
            text = self.redactor.redact((action.text or "").strip())[:limit]
            if not text.strip():
                continue
            spoken.append(action.model_copy(update={"text": text[:MAX_TEXT_CHARS]}))
        return spoken or [PASS]
