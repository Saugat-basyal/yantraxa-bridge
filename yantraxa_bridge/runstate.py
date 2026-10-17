"""`$YANTRAXA_HOME/run-state.json`: what the running bridge last did (0.2.1).

`run` writes it, in a terminal and as the background service: when it started, when it last
connected and as whom, and its last error with the time. `status` and `doctor` read it to
say whether anything is running and why it isn't connected, and `pair` reads it to wait for
the service's first connection. It holds no secret; it is 0600 like everything in the home.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .config import ensure_home, write_private, yantraxa_home

STATE_FILE = "run-state.json"  # not state.json: that is the caps store (caps.py)

log = logging.getLogger("yantraxa_bridge.runstate")


def state_path(home: Path | None = None) -> Path:
    return (home or yantraxa_home()) / STATE_FILE


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_state(home: Path | None = None) -> dict[str, Any]:
    """The last state written, or {} when there is none or it can't be read."""
    try:
        state = json.loads(state_path(home).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        when = datetime.fromisoformat(value)
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=UTC)


def local_time(value: Any) -> str:
    """`2026-10-17 12:41` in this machine's time zone, or the value as it is."""
    when = parse_time(value)
    return when.astimezone().strftime("%Y-%m-%d %H:%M") if when else str(value)


def pid_alive(pid: int) -> bool:
    from .service import _posix_pid_alive, windows_pid_alive

    if pid <= 0:
        return False
    return windows_pid_alive(pid) if os.name == "nt" else _posix_pid_alive(pid)


def is_running(state: dict[str, Any]) -> bool:
    """Whether the bridge that wrote `state` is still running: it hasn't recorded a stop,
    and its process is alive."""
    pid = state.get("pid")
    return isinstance(pid, int) and not state.get("stopped_at") and pid_alive(pid)


def connected_now(state: dict[str, Any]) -> bool:
    """Whether its newest event is a connection, not an error after one."""
    connected = parse_time(state.get("connected_at"))
    failed = parse_time(state.get("last_error_at"))
    return connected is not None and (failed is None or connected >= failed)


class StateWriter:
    """What one `run` records. Writing never fails the bridge: an unwritable home is logged."""

    def __init__(self, home: Path | None = None, *, mode: str) -> None:
        self.home = home or yantraxa_home()
        self.state: dict[str, Any] = {}
        self.mode = mode

    def _write(self) -> None:
        try:
            ensure_home(self.home)
            write_private(state_path(self.home), json.dumps(self.state, indent=2) + "\n")
        except OSError as error:
            log.debug("couldn't write %s: %s", state_path(self.home), error)

    def started(self) -> None:
        """A fresh record; the last error survives a restart, so a crash loop still shows it."""
        before = read_state(self.home)
        self.state = {
            "pid": os.getpid(),
            "mode": self.mode,
            "version": __version__,
            "started_at": now(),
        }
        for key in ("last_error", "last_error_at"):
            if key in before:
                self.state[key] = before[key]
        self._write()

    def connected(self, agent: dict[str, Any]) -> None:
        self.state.update(connected_at=now(), handle=agent.get("handle"))
        self._write()

    def error(self, reason: str) -> None:
        self.state.update(last_error=reason, last_error_at=now())
        self._write()

    def stopped(self, reason: str) -> None:
        self.state.update(stopped=reason, stopped_at=now())
        self._write()


def describe(state: dict[str, Any]) -> str | None:
    """`connected as @h since 2026-10-17 12:41` or `last error: <reason> at <time>`, from
    the newest of the two; None when neither is recorded."""
    if connected_now(state):
        return f"connected as @{state.get('handle')} since {local_time(state['connected_at'])}"
    if state.get("last_error"):
        return f"last error: {state['last_error']} at {local_time(state.get('last_error_at'))}"
    return None
