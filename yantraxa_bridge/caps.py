"""Owner-side caps on CLI turns (spec §4.6), counted in `$YANTRAXA_HOME/state.json`.

- At most `per_meeting` turns in one meeting and `per_day` turns in a local calendar day
  (D7: 40 and 200). Over a cap the turn passes without starting the CLI, and the bridge
  prints one line saying so (once per meeting, once per day). Project turns (0.2.0) count
  towards the day only.
- After two vendor rate-limit errors in a row the bridge stops calling the CLI for ten
  minutes, so a stuck meeting doesn't burn the owner's plan window.

The file is written 0600 and atomically. One bridge per home is assumed, as for the
credential.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ensure_home, write_private

log = logging.getLogger("yantraxa_bridge.caps")

STATE_FILE = "state.json"
VENDOR_ERRORS_BEFORE_BACKOFF = 2
VENDOR_BACKOFF_S = 600.0
# A meeting not heard from for this long is dropped from the counts.
MEETING_TTL_S = 7 * 24 * 3600


class Caps:
    def __init__(
        self,
        home: Path,
        *,
        per_meeting: int,
        per_day: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.home = home
        self.path = home / STATE_FILE
        self.per_meeting = per_meeting
        self.per_day = per_day
        self.clock = clock
        self._warned: set[str] = set()

    # -------------------------------------------------------------- state

    def _today(self) -> str:
        return datetime.fromtimestamp(self.clock()).strftime("%Y-%m-%d")

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        if data.get("day") != self._today():
            data["day"], data["day_count"] = self._today(), 0
        meetings = data.get("meetings")
        data["meetings"] = meetings if isinstance(meetings, dict) else {}
        return data

    def _save(self, data: dict[str, Any]) -> None:
        now = self.clock()
        data["meetings"] = {
            meeting: entry
            for meeting, entry in data["meetings"].items()
            if isinstance(entry, dict) and now - float(entry.get("last", 0)) < MEETING_TTL_S
        }
        ensure_home(self.home)
        write_private(self.path, json.dumps(data, indent=2, sort_keys=True) + "\n")

    # -------------------------------------------------------------- the checks

    def check(self, meeting_id: str | None) -> str | None:
        """Why this meeting's turn must pass without calling the CLI, or None. A project
        turn (`meeting_id` None) is held to the backoff and the day's cap only."""
        data = self._load()
        until = float(data.get("backoff_until") or 0)
        if until > self.clock():
            minutes = max(1, round((until - self.clock()) / 60))
            return self._once(
                f"backoff:{until}",
                "the vendor CLI reported its usage limit twice; passing for about "
                f"{minutes} more minute(s)",
            )
        count = int(data["meetings"].get(meeting_id, {}).get("count", 0)) if meeting_id else 0
        if meeting_id is not None and count >= self.per_meeting:
            return self._once(
                f"meeting:{meeting_id}",
                f"this meeting reached max_turns_per_meeting ({self.per_meeting}); passing",
            )
        if int(data.get("day_count", 0)) >= self.per_day:
            return self._once(
                f"day:{data['day']}",
                f"today's turns reached max_turns_per_day ({self.per_day}); passing until tomorrow",
            )
        return None

    def _once(self, key: str, reason: str) -> str:
        if key not in self._warned:
            self._warned.add(key)
            log.warning("%s", reason)
        return reason

    def record_turn(self, meeting_id: str | None) -> None:
        """Counts one CLI turn against this meeting (unless a project's) and today."""
        data = self._load()
        if meeting_id is not None:
            entry = data["meetings"].get(meeting_id)
            entry = entry if isinstance(entry, dict) else {}
            data["meetings"][meeting_id] = {
                "count": int(entry.get("count", 0)) + 1,
                "last": self.clock(),
            }
        data["day_count"] = int(data.get("day_count", 0)) + 1
        self._save(data)

    def vendor_limited(self) -> None:
        """The CLI reported a usage or rate limit; the second in a row starts the backoff."""
        data = self._load()
        errors = int(data.get("vendor_errors", 0)) + 1
        data["vendor_errors"] = errors
        if errors >= VENDOR_ERRORS_BEFORE_BACKOFF:
            data["backoff_until"] = self.clock() + VENDOR_BACKOFF_S
            data["vendor_errors"] = 0
            log.warning(
                "the vendor CLI reports its usage limit is reached; passing for %d minutes",
                VENDOR_BACKOFF_S // 60,
            )
        else:
            log.warning("the vendor CLI reports its usage limit is reached; passing")
        self._save(data)

    def vendor_ok(self) -> None:
        data = self._load()
        if data.get("vendor_errors"):
            data["vendor_errors"] = 0
            self._save(data)

    def usage(self, meeting_id: str | None = None) -> dict[str, int]:
        data = self._load()
        used = {"today": int(data.get("day_count", 0))}
        if meeting_id is not None:
            used["meeting"] = int(data["meetings"].get(meeting_id, {}).get("count", 0))
        return used
