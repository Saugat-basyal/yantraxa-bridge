import json
import logging
import stat
from datetime import datetime

from yantraxa_bridge.caps import VENDOR_BACKOFF_S, Caps

A = "3f2c6a5e-8a0e-4f7a-9d55-0d3c2a1b4c5d"
B = "4a3d7b6f-9b1f-4a8b-8e66-1e4d3b2c5d6e"


class Clock:
    def __init__(self, start: float) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def noon(day: int) -> float:
    return datetime(2026, 10, day, 12, 0).timestamp()


def caps(home, clock, per_meeting=3, per_day=5):
    return Caps(home, per_meeting=per_meeting, per_day=per_day, clock=clock)


def test_the_per_meeting_cap(home):
    c = caps(home, Clock(noon(14)))
    for _ in range(3):
        assert c.check(A) is None
        c.record_turn(A)
    assert "max_turns_per_meeting (3)" in c.check(A)
    assert c.check(B) is None  # another meeting has its own count


def test_the_daily_cap_resets_the_next_day(home):
    clock = Clock(noon(14))
    c = caps(home, clock, per_meeting=40)
    for meeting in (A, A, B, B, B):
        c.record_turn(meeting)
    assert "max_turns_per_day (5)" in c.check("5b4e8c70-0c2a-4b9c-9f77-2f5e4c3d6e7f")
    clock.now = noon(15)
    assert c.check(B) is None
    assert c.usage(A) == {"today": 0, "meeting": 2}  # meeting counts survive midnight


def test_counts_live_in_a_private_state_file(home):
    c = caps(home, Clock(noon(14)))
    c.record_turn(A)
    path = home / "state.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    data = json.loads(path.read_text())
    assert data["day"] == "2026-10-14" and data["day_count"] == 1
    assert data["meetings"][A]["count"] == 1
    # A new process (a bridge restart) sees the same counts.
    assert caps(home, Clock(noon(14))).usage(A) == {"today": 1, "meeting": 1}


def test_old_meetings_are_forgotten(home):
    clock = Clock(noon(1))
    c = caps(home, clock)
    c.record_turn(A)
    clock.now = noon(14)
    c.record_turn(B)
    assert A not in json.loads((home / "state.json").read_text())["meetings"]


def test_a_broken_state_file_starts_again(home):
    home.mkdir(parents=True)
    (home / "state.json").write_text("{not json")
    c = caps(home, Clock(noon(14)))
    assert c.check(A) is None
    c.record_turn(A)
    assert c.usage(A)["meeting"] == 1


def test_two_vendor_limits_in_a_row_back_off_for_ten_minutes(home):
    clock = Clock(noon(14))
    c = caps(home, clock)
    c.vendor_limited()
    assert c.check(A) is None  # one error: just that turn passes
    c.vendor_limited()
    assert "usage limit" in c.check(A)
    clock.now += VENDOR_BACKOFF_S - 1
    assert c.check(A) is not None
    clock.now += 2
    assert c.check(A) is None


def test_a_good_turn_resets_the_vendor_error_count(home):
    c = caps(home, Clock(noon(14)))
    c.vendor_limited()
    c.vendor_ok()
    c.vendor_limited()
    assert c.check(A) is None


def test_the_over_cap_line_prints_once(home, caplog):
    c = caps(home, Clock(noon(14)), per_meeting=1)
    c.record_turn(A)
    with caplog.at_level(logging.WARNING, logger="yantraxa_bridge.caps"):
        for _ in range(4):
            assert c.check(A) is not None
    assert len([r for r in caplog.records if "max_turns_per_meeting" in r.getMessage()]) == 1
