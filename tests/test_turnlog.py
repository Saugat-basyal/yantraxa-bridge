import asyncio
import json
import stat

import pytest
from typer.testing import CliRunner

from yantraxa_bridge.cli import app
from yantraxa_bridge.redaction import REDACTED, Redactor, secret_filter
from yantraxa_bridge.turnlog import (
    LoggedTurnHandler,
    TurnLog,
    format_event,
    log_path,
    printable,
    read_events,
)

from .test_turn_handler import KEY, FakeIO, Scripted, meeting_state, say

TURN = "0b6f3c1e-5a4d-4e2f-9c8b-7a6d5e4f3a2b"


def test_events_are_json_lines_in_a_private_file(home):
    log = TurnLog(home)
    log.record("model", meeting_code="AB-1234", turn_id=TURN, model=None, pass_reason="x")
    log.record("sent", meeting_code="AB-1234", actions=[{"action": "pass"}])
    path = log_path(home)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    first, second = (json.loads(line) for line in path.read_text().splitlines())
    assert first["event"] == "model" and first["turn_id"] == TURN and "model" not in first
    assert first["ts"].startswith("20")
    assert second["actions"] == [{"action": "pass"}]


def test_the_agent_secret_never_lands_in_the_log(home):
    secret_filter.add_secret("s3cret-agent-value-123")
    TurnLog(home).record(
        "model",
        raw_output='leaked s3cret-agent-value-123 and {"secret": "abc"} Bearer ag_abc234:xyz',
    )
    text = log_path(home).read_text()
    assert "s3cret-agent-value-123" not in text and '"abc"' not in text and "xyz" not in text
    assert REDACTED in text


def test_rotation_keeps_five_old_files(home):
    log = TurnLog(home, max_bytes=300, backups=5)
    for n in range(40):
        log.record("sent", n=n, padding="x" * 100)
    path = log_path(home)
    names = sorted(p.name for p in path.parent.iterdir())
    assert names == ["turns.jsonl", *(f"turns.jsonl.{n}" for n in range(1, 6))]
    assert all(p.stat().st_size <= 300 for p in path.parent.iterdir())
    assert read_events(home, 1)[0]["n"] == 39


def test_an_unwritable_log_never_fails_a_turn(home, tmp_path):
    blocker = tmp_path / "file-not-dir"
    blocker.write_text("")
    TurnLog(blocker).record("sent", n=1)  # the log dir can't be created: nothing raises


def test_read_events_takes_the_newest(home):
    log = TurnLog(home)
    for n in range(30):
        log.record("sent", n=n)
    assert [e["n"] for e in read_events(home, 3)] == [27, 28, 29]
    assert read_events(home / "elsewhere") == []


def test_format_event():
    line = format_event(
        {
            "ts": "2026-10-14T20:41:03.120+11:00",
            "event": "model",
            "meeting_code": "AB-1234",
            "adapter": "claude_code",
            "model": "haiku",
            "duration_s": 4.13,
            "actions": [
                {
                    "action": "reply",
                    "reply_to": "x",
                    "text": "Yes,\n the roof.",
                    "vote": {"choice": "support"},
                }
            ],
        }
    )
    assert line == (
        "2026-10-14 20:41:03 AB-1234 model claude_code/haiku 4.1s reply (reply) [vote support] "
        '"Yes, the roof."'
    )
    assert format_event({"ts": "x", "event": "model", "pass_reason": "refusal"}).endswith(
        "pass: refusal"
    )
    assert format_event({"event": "sent", "outcome": "ran out of time"}).endswith("ran out of time")


# ------------------------------------------------------------------ the `sent` line


def logged(adapter, home):
    return LoggedTurnHandler(
        adapter,
        Redactor(),
        timeout_s=40,
        chunk_interval_s=0,
        turnlog=TurnLog(home),
        adapter_name="claude_code",
    )


async def test_the_sent_line_holds_the_redacted_actions(home):
    handler = logged(Scripted(say(f"my key is {KEY}"), streams=False), home)
    io = FakeIO(meeting_state())
    actions = await handler.take_turn(io.request, io)
    [event] = read_events(home)
    assert event["event"] == "sent" and event["adapter"] == "claude_code"
    assert event["meeting_code"] == "AB-1234" and event["turn_id"] == io.request.turn_id
    assert event["actions"] == [a.model_dump(mode="json", exclude_none=True) for a in actions]
    assert KEY not in log_path(home).read_text()
    assert "duration_s" in event


class Slow:
    def describe(self):
        return Scripted().describe()

    async def take_turn(self, ctx):
        await asyncio.sleep(30)
        yield say("late")

    async def aclose(self):
        pass


async def test_a_turn_that_runs_out_of_time_is_logged(home):
    handler = logged(Slow(), home)
    io = FakeIO(meeting_state())
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(handler.take_turn(io.request, io), 0.2)
    [event] = read_events(home)
    assert event["outcome"] == "ran out of time; the bridge passed"


# ------------------------------------------------------------------ `yantraxa-bridge log`


def test_the_log_command(home):
    runner = CliRunner()
    assert "No turns logged yet" in runner.invoke(app, ["log"]).output
    TurnLog(home).record(
        "sent", meeting_code="AB-1234", adapter="codex_cli", actions=[{"action": "pass"}]
    )
    result = runner.invoke(app, ["log"])
    assert result.exit_code == 0 and "AB-1234 sent  codex_cli pass" in result.output
    raw = runner.invoke(app, ["log", "--json", "-n", "1"])
    assert json.loads(raw.output)["adapter"] == "codex_cli"


HOSTILE = "ok\x1b]0;owned\x07\x1b[2J\x9b31m done\r\x00"


def test_control_characters_never_reach_the_terminal(home):
    assert printable(HOSTILE) == "ok]0;owned[2J31m done"
    assert printable("two\nlines\tand a tab") == "two\nlines\tand a tab"
    line = format_event(
        {
            "event": "model",
            "adapter": "codex_cli",
            "actions": [{"action": "message", "text": HOSTILE}],
        }
    )
    assert not any(ch in line for ch in "\x1b\x07\x9b\r\x00")
    reason = format_event(
        {"event": "model", "pass_reason": f"Codex's answer isn't JSON: {HOSTILE}"}
    )
    assert "\x1b" not in reason and "\x07" not in reason
    TurnLog(home).record("model", adapter="codex_cli", meeting_code="AB-1234", pass_reason=HOSTILE)
    runner = CliRunner()
    for args in (["log"], ["log", "--json"]):
        output = runner.invoke(app, args).output
        assert "owned" in output
        assert not any(ch in output for ch in "\x1b\x07\x9b\r\x00"), args


async def test_the_stored_model_output_has_no_control_characters(home):
    from yantraxa_bridge.adapters.cli_runner import Attempt, CliAdapter

    from .contexts import make_context

    adapter = CliAdapter.__new__(CliAdapter)
    adapter.turnlog, adapter.adapter_name = TurnLog(home), "codex_cli"
    adapter._log(make_context(), model=None, attempt=Attempt(raw=HOSTILE, reason="x"), actions=None)
    [event] = read_events(home)
    assert event["raw_output"] == "ok]0;owned[2J31m done"


@pytest.mark.parametrize("separator", ["\n", "\t", "\r\n"])
def test_a_key_after_a_line_break_or_tab_in_a_value_is_redacted_in_the_log(home, separator):
    key = "sk-ant-api03-" + "Ab3_dE-fG" * 10 + "-AA"
    TurnLog(home).record(
        "model",
        raw_output=f"here it is:{separator}{key}",
        actions=[{"action": "message", "text": f"line one{separator}{key}"}],
    )
    text = log_path(home).read_text()
    assert "sk-ant" not in text and "Ab3_dE" not in text
    [event] = read_events(home)
    assert event["raw_output"] == f"here it is:{separator}{REDACTED}"
    assert event["actions"][0]["text"] == f"line one{separator}{REDACTED}"
