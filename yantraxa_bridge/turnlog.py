"""The transparency log (spec §1.6): `$YANTRAXA_HOME/log/turns.jsonl`, one JSON line per event.

Two kinds of line:

- `model`, written by the CLI adapters: the meeting, the turn, the adapter and model, the
  prompt the model was given (the system text and the fenced meeting data), its raw output,
  the validated actions, timings and usage, and why a turn passed.
- `sent`, written for every adapter's turn: the actions the bridge sent after redaction,
  or that the turn failed or ran out of time (the client then passes).
- `downgraded`, when the model answered with an action the turn doesn't allow (a
  proposal with proposals off): what it chose, and that it went out as a message or a pass.

Append-only, 0600 in a 0700 directory, rotated at 20 MB with five old files kept. Every
line goes through the secret filter, so the agent secret never lands in it.
`yantraxa-bridge log` prints it readably; `--tail` follows it. Control characters are
stripped from the stored model output and from everything printed, so a hostile answer
can't send escape sequences to the owner's terminal.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
from collections import deque
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from .adapters.base import TurnContext
from .adapters.handler import AdapterTurnHandler
from .client import TurnIO
from .config import ensure_private_dir
from .protocol import TurnAction, TurnRequestPayload
from .redaction import secret_filter

LOG_DIR = "log"
LOG_FILE = "turns.jsonl"
MAX_BYTES = 20 * 1024 * 1024
BACKUPS = 5
# C0 and C1 control characters, and DEL, except tab and newline: an ESC or BEL in a model's
# output would otherwise reach the owner's terminal through `yantraxa-bridge log`.
_CONTROLS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def printable(text: str) -> str:
    """`text` without control characters other than newline and tab."""
    return _CONTROLS.sub("", text)


def log_path(home: Path) -> Path:
    return home / LOG_DIR / LOG_FILE


class TurnLog:
    def __init__(self, home: Path, *, max_bytes: int = MAX_BYTES, backups: int = BACKUPS) -> None:
        self.path = log_path(home)
        self.max_bytes = max_bytes
        self.backups = backups

    def record(self, event: str, **fields: Any) -> None:
        """Appends one event; fields that are None are left out. Never raises."""
        entry = {"ts": datetime.now().astimezone().isoformat(timespec="milliseconds")}
        entry["event"] = event
        entry.update({k: v for k, v in fields.items() if v is not None})
        entry = secret_filter.scrub_values(entry)
        line = secret_filter.scrub(json.dumps(entry, ensure_ascii=False, default=str)) + "\n"
        with contextlib.suppress(OSError):  # the log must never cost a turn
            self._append(line.encode("utf-8"))

    def _append(self, data: bytes) -> None:
        directory = ensure_private_dir(self.path.parent)
        os.chmod(directory, 0o700)
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            size = 0
        if size and size + len(data) > self.max_bytes:
            self._rotate()
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            os.write(fd, data)
        finally:
            os.close(fd)

    def _rotate(self) -> None:
        oldest = self.path.with_name(f"{LOG_FILE}.{self.backups}")
        oldest.unlink(missing_ok=True)
        for n in range(self.backups - 1, 0, -1):
            older = self.path.with_name(f"{LOG_FILE}.{n}")
            if older.exists():
                older.replace(self.path.with_name(f"{LOG_FILE}.{n + 1}"))
        self.path.replace(self.path.with_name(f"{LOG_FILE}.1"))


def _actions(actions: list[TurnAction]) -> list[dict[str, Any]]:
    return [a.model_dump(mode="json", exclude_none=True) for a in actions]


class LoggedTurnHandler(AdapterTurnHandler):
    """An `AdapterTurnHandler` that writes a `sent` line for every turn it answers."""

    def __init__(self, *args: Any, turnlog: TurnLog, adapter_name: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.turnlog = turnlog
        self.adapter_name = adapter_name

    def downgraded(self, ctx: TurnContext, action: TurnAction, reason: str) -> None:
        self.turnlog.record(
            "downgraded",
            meeting_id=ctx.meeting.id,
            meeting_code=ctx.meeting.code,
            turn_id=ctx.turn_id,
            adapter=self.adapter_name,
            chose=action.action,
            outcome=reason,
        )

    async def take_turn(self, request: TurnRequestPayload, io: TurnIO) -> list[TurnAction]:
        started = time.monotonic()
        fields = {
            "meeting_id": request.meeting_id,
            "meeting_code": io.meeting.code if io.meeting is not None else None,
            "turn_id": request.turn_id,
            "adapter": self.adapter_name,
        }
        try:
            actions = await super().take_turn(request, io)
        except asyncio.CancelledError:
            self.turnlog.record(
                "sent",
                **fields,
                outcome="ran out of time; the bridge passed",
                duration_s=round(time.monotonic() - started, 2),
            )
            raise
        except Exception as error:
            self.turnlog.record(
                "sent",
                **fields,
                outcome=f"failed ({type(error).__name__}); the bridge passed",
                duration_s=round(time.monotonic() - started, 2),
            )
            raise
        self.turnlog.record(
            "sent",
            **fields,
            actions=_actions(actions),
            duration_s=round(time.monotonic() - started, 2),
        )
        return actions


# ------------------------------------------------------------------ reading it back


def read_events(home: Path, last: int = 20) -> list[dict[str, Any]]:
    """The newest `last` events in the current file, oldest first."""
    path = log_path(home)
    try:
        with path.open(encoding="utf-8") as handle:
            lines = deque(handle, maxlen=last)
    except FileNotFoundError:
        return []
    return [event for line in lines if (event := _parse(line)) is not None]


def _parse(line: str) -> dict[str, Any] | None:
    try:
        event = json.loads(line)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def follow(home: Path, *, poll_s: float = 0.5) -> Iterator[dict[str, Any]]:
    """New events as they are written, across rotations. Runs until interrupted."""
    path = log_path(home)
    handle = None
    inode = None
    while True:
        if handle is None:
            try:
                handle = path.open(encoding="utf-8")
                inode = os.fstat(handle.fileno()).st_ino
                handle.seek(0, os.SEEK_END)
            except FileNotFoundError:
                time.sleep(poll_s)
                continue
        line = handle.readline()
        if line:
            if (event := _parse(line)) is not None:
                yield event
            continue
        try:
            rotated = path.stat().st_ino != inode
        except FileNotFoundError:
            rotated = True
        if rotated:
            handle.close()
            handle = None
            try:
                handle = path.open(encoding="utf-8")  # the new file, from its start
                inode = os.fstat(handle.fileno()).st_ino
            except FileNotFoundError:
                handle = None
            continue
        time.sleep(poll_s)


def _said(action: dict[str, Any]) -> str:
    kind = action.get("action", "?")
    text = " ".join(str(action.get("text") or "").split())
    if len(text) > 120:
        text = text[:119] + "…"
    reply = " (reply)" if action.get("reply_to") else ""
    vote = action.get("vote")
    voted = f" [vote {vote.get('choice')}]" if isinstance(vote, dict) else ""
    return f'{kind}{reply}{voted} "{text}"' if text else kind


def format_event(event: dict[str, Any]) -> str:
    """One readable line, e.g. `20:41:03 AB-1234 model claude_code/haiku 4.1s reply "…"`."""
    try:
        when = datetime.fromisoformat(str(event.get("ts"))).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        when = str(event.get("ts", "?"))
    kind = str(event.get("event", "?"))
    who = str(event.get("adapter", "?"))
    if event.get("model"):
        who += f"/{event['model']}"
    timing = event.get("duration_s")
    parts = [when, str(event.get("meeting_code") or "-"), f"{kind:<5}", who]
    if isinstance(timing, int | float):
        parts.append(f"{timing:.1f}s")
    actions = event.get("actions")
    if isinstance(actions, list) and actions:
        parts.append("; ".join(_said(a) for a in actions if isinstance(a, dict)))
    elif event.get("pass_reason"):
        parts.append(f"pass: {event['pass_reason']}")
    elif event.get("outcome"):
        parts.append(str(event["outcome"]))
    else:
        parts.append("pass")
    return printable(" ".join(parts))
