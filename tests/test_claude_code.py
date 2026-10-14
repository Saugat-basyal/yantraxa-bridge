import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner as TyperRunner

from yantraxa_bridge.adapters import make_turn_handler
from yantraxa_bridge.adapters.base import ContextMessage, Final
from yantraxa_bridge.adapters.claude_code import CLAUDE_ENV, ClaudeCodeAdapter, preflight
from yantraxa_bridge.adapters.cli_runner import CliRunner, PreflightError
from yantraxa_bridge.caps import Caps
from yantraxa_bridge.cli import app
from yantraxa_bridge.config import BridgeConfig
from yantraxa_bridge.prompt import output_schema
from yantraxa_bridge.turnlog import LoggedTurnHandler, TurnLog, read_events

from .contexts import M2, OTHER_SEAT, P1, make_context
from .fake_cli import claude_result, install

pytestmark = pytest.mark.skipif(os.name == "nt", reason="fake CLIs are POSIX scripts")

M3 = "66666666-6666-4666-8666-666666666666"
REPLY = {
    "action": "reply",
    "reply_to": "m2",
    "text": "Saugat can help fix the roof on Saturday.",
    "vote": {"motion_id": "p1", "choice": "support", "reason": "It leaks now."},
}


@pytest.fixture
def claude(monkeypatch, tmp_path):
    fake = install(monkeypatch, tmp_path, "claude")
    fake.answer(claude_result(REPLY))
    return fake


def adapter(**kw) -> ClaudeCodeAdapter:
    return ClaudeCodeAdapter(CliRunner(timeout_s=10), **kw)


async def take(adapter, ctx=None):
    events = [e async for e in adapter.take_turn(ctx or make_context())]
    assert len(events) == 1 and isinstance(events[0], Final)
    return events[0].actions


def addressed_context():
    ctx = make_context()
    ctx.recent_messages.append(
        ContextMessage(
            id=M3,
            seat_id=OTHER_SEAT,
            author="Priya",
            kind="question",
            body="@saugat.ai can you summarise the options?",
            created_at="2026-10-11T09:02:00.000Z",
        )
    )
    return ctx


# ------------------------------------------------------------------ the invocation


async def test_the_exact_confined_invocation(claude):
    await take(adapter())
    [call] = claude.calls()
    turn_dir = Path(call["cwd"])
    argv = call["argv"]
    system_file = argv[argv.index("--system-prompt-file") + 1]
    assert Path(system_file).resolve() == turn_dir / "system.txt"
    assert argv == [
        "-p",
        "--output-format", "json",
        "--tools", "",
        "--disallowedTools", "mcp__*",
        "--strict-mcp-config",
        "--setting-sources", "",
        "--disable-slash-commands",
        "--permission-mode", "dontAsk",
        "--permission-prompts", "none",
        "--no-session-persistence",
        "--max-turns", "1",
        "--model", "sonnet",
        "--effort", "low",
        "--system-prompt-file", system_file,
        "--json-schema", json.dumps(output_schema(), separators=(",", ":")),
    ]  # fmt: skip


async def test_meeting_text_and_approved_facts_never_reach_argv(claude):
    await take(adapter())
    [call] = claude.calls()
    argv = " ".join(call["argv"])
    for secretish in ("Saturday mornings", "shed roof leaks", "AB-1234", "Priya"):
        assert secretish not in argv
    system = call["files"]["system.txt"]
    assert system["mode"] == "0o600"
    assert "Saugat can volunteer on Saturday mornings." in system["text"]
    assert "no tools" in system["text"]
    assert "<<meeting_data>>\n" in call["stdin"] and "shed roof leaks" in call["stdin"]


async def test_the_environment_is_stripped_and_the_login_is_the_owners(claude, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-parent-key-should-not-pass")
    await take(adapter())
    env = claude.calls()[0]["env"]
    for key, value in CLAUDE_ENV.items():
        assert env[key] == value
    assert "ANTHROPIC_API_KEY" not in env


async def test_api_key_mode_passes_the_key(claude):
    await take(adapter(api_key="sk-ant-owner-chose-this"))
    assert claude.calls()[0]["env"]["ANTHROPIC_API_KEY"] == "sk-ant-owner-chose-this"


async def test_rotation_turns_use_the_full_model_and_addressed_turns_the_fast_one(claude):
    claude.answer(claude_result({**REPLY, "reply_to": "m3"}))
    a = adapter(model_fast="haiku", model_full="opus", effort="medium")
    await take(a)
    [action] = await take(a, addressed_context())
    first, second = claude.calls()
    assert first["argv"][first["argv"].index("--model") + 1] == "opus"
    assert second["argv"][second["argv"].index("--model") + 1] == "haiku"
    assert second["argv"][second["argv"].index("--effort") + 1] == "medium"
    assert 'reply_to "m3"' in second["files"]["system.txt"]["text"]
    assert action.action == "reply" and action.reply_to == M3


# ------------------------------------------------------------------ answers


async def test_a_structured_answer_becomes_actions_with_real_ids(claude):
    [action] = await take(adapter())
    assert action.action == "reply" and action.reply_to == M2
    assert action.vote is not None and action.vote.motion_id == P1
    assert action.text == "Saugat can help fix the roof on Saturday."


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "not json",
        "[]",
        claude_result(None),
        claude_result(REPLY, is_error=True, subtype="error_during_execution"),
        claude_result(REPLY, subtype="error_max_turns"),
        claude_result(REPLY, stop_reason="refusal"),
        claude_result({"action": "dance", "text": "no"}),
        claude_result({"action": "message", "text": "   "}),
        claude_result(REPLY, permission_denials=[{"tool_name": "Bash"}]),
        claude_result(REPLY, subagent_stats={"spawned": 1}),
        claude_result(REPLY, usage={"server_tool_use": {"web_fetch_requests": 1}}),
    ],
)
async def test_anything_but_a_clean_answer_passes(claude, stdout):
    claude.answer(stdout)
    assert [a.action for a in await take(adapter())] == ["pass"]


async def test_a_non_zero_exit_passes_even_with_an_answer(claude):
    claude.answer(claude_result(REPLY), exit=1, stderr="boom")
    assert [a.action for a in await take(adapter())] == ["pass"]


async def test_a_file_left_in_the_turn_dir_passes(claude):
    claude.answer(claude_result(REPLY), touch="notes.md")
    assert [a.action for a in await take(adapter())] == ["pass"]


async def test_a_left_behind_file_name_has_no_control_characters(claude, home, caplog):
    claude.answer(claude_result(REPLY), touch="\x1b[2Jnotes\x07.md")
    a = adapter(turnlog=TurnLog(home))
    assert [x.action for x in await take(a)] == ["pass"]
    [event] = read_events(home)
    assert event["pass_reason"] == "Claude Code left files behind: [2Jnotes.md"
    assert "\x1b" not in caplog.text and "\x07" not in caplog.text


async def test_a_timeout_passes(claude):
    claude.answer(claude_result(REPLY), sleep=10)
    a = ClaudeCodeAdapter(CliRunner(timeout_s=0.5, kill_grace_s=0.2))
    assert [x.action for x in await take(a)] == ["pass"]


async def test_a_missing_claude_passes(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert [a.action for a in await take(adapter())] == ["pass"]


MAX_TURNS = claude_result(
    None,
    is_error=True,
    subtype="error_max_turns",
    stop_reason="end_turn",
    result=None,
    errors=["Reached maximum number of turns (1)"],
)


async def test_error_max_turns_is_retried_once(claude, home):
    claude.behave(
        {"default": {"sequence": [{"stdout": MAX_TURNS}, {"stdout": claude_result(REPLY)}]}}
    )
    [action] = await take(adapter(turnlog=TurnLog(home)))
    assert action.action == "reply"
    assert len(claude.calls()) == 2
    [event] = read_events(home)
    assert event["retried"].startswith("Claude Code returned an error (error_max_turns)")


async def test_the_retry_of_an_addressed_turn_uses_the_full_model(claude):
    claude.behave(
        {"default": {"sequence": [{"stdout": MAX_TURNS}, {"stdout": claude_result(REPLY)}]}}
    )
    await take(adapter(model_fast="haiku", model_full="opus"), addressed_context())
    first, second = (call["argv"] for call in claude.calls())
    assert first[first.index("--model") + 1] == "haiku"
    assert second[second.index("--model") + 1] == "opus"


async def test_a_second_error_max_turns_passes(claude):
    claude.answer(MAX_TURNS)
    assert [a.action for a in await take(adapter())] == ["pass"]
    assert len(claude.calls()) == 2


async def test_other_errors_are_not_retried(claude):
    claude.answer(claude_result(REPLY, is_error=True, subtype="error_during_execution"))
    await take(adapter())
    assert len(claude.calls()) == 1


async def test_no_retry_without_time_for_it(claude):
    claude.behave(
        {
            "default": {
                "sequence": [{"stdout": MAX_TURNS, "sleep": 0.7}, {"stdout": claude_result(REPLY)}]
            }
        }
    )
    a = ClaudeCodeAdapter(CliRunner(timeout_s=1.2))
    assert [x.action for x in await take(a)] == ["pass"]
    assert len(claude.calls()) == 1


# ------------------------------------------------------------------ caps and the log


async def test_caps_pass_without_calling_claude(claude, home):
    caps = Caps(home, per_meeting=2, per_day=200)
    a = adapter(caps=caps)
    assert (await take(a))[0].action == "reply"
    assert (await take(a))[0].action == "reply"
    assert (await take(a))[0].action == "pass"
    assert len(claude.calls()) == 2


async def test_two_usage_limit_errors_back_off(claude, home):
    claude.answer(
        claude_result(None, is_error=True, subtype="error", result="Claude AI usage limit reached")
    )
    a = adapter(caps=Caps(home, per_meeting=40, per_day=200))
    for _ in range(3):
        assert (await take(a))[0].action == "pass"
    assert len(claude.calls()) == 2  # the third turn didn't start the CLI


async def test_the_model_line_in_the_transparency_log(claude, home):
    log = TurnLog(home)
    await take(adapter(turnlog=log))
    [event] = read_events(home)
    assert event["event"] == "model" and event["adapter"] == "claude_code"
    assert event["model"] == "sonnet" and event["meeting_code"] == "AB-1234"
    assert "<<meeting_data>>" in event["prompt"]
    assert "Saturday mornings" in event["system"]
    assert json.loads(event["raw_output"])["structured_output"] == REPLY
    assert event["actions"][0]["reply_to"] == M2
    assert event["api_ms"] == 3208 and "duration_s" in event
    assert "pass_reason" not in event


async def test_a_pass_reason_is_logged_not_posted(claude, home):
    claude.answer(claude_result(REPLY, stop_reason="refusal"))
    [action] = await take(adapter(turnlog=TurnLog(home)))
    assert action.text is None
    assert read_events(home)[0]["pass_reason"] == "Claude declined to answer (refusal)"


def test_make_turn_handler_builds_the_claude_adapter(home):
    handler = make_turn_handler(
        BridgeConfig(adapter="claude-code", claude_model="opus", claude_model_fast="haiku")
    )
    assert isinstance(handler, LoggedTurnHandler)
    assert isinstance(handler.adapter, ClaudeCodeAdapter)
    assert handler.adapter.model_full == "opus" and handler.adapter.caps is not None
    assert handler.adapter.runner.timeout_s == 35
    assert handler.adapter.describe().runtime_label == "Claude Code on owner's computer"


# ------------------------------------------------------------------ preflight


def preflight_behaviour(logged_in=True, probe=None):
    return {
        "--version": {"stdout": "2.1.299 (Claude Code)\n"},
        "auth status": {
            "stdout": json.dumps({"loggedIn": logged_in, "email": "owner@example.com"})
        },
        "default": {
            "stdout": probe or claude_result({"action": "reply", "reply_to": "m1", "text": "Yes."})
        },
    }


async def test_preflight_checks_version_sign_in_and_a_probe(claude):
    claude.behave(preflight_behaviour())
    summary = await preflight(adapter())
    assert summary.startswith("Claude Code 2.1.299, signed in as owner@example.com, probe OK (")
    version, status, turn = claude.calls()
    assert version["argv"] == ["--version"] and status["argv"] == ["auth", "status"]
    assert turn["argv"][turn["argv"].index("--model") + 1] == "sonnet"  # the default fast model


def test_both_claude_models_default_to_sonnet():
    config = BridgeConfig()
    assert config.claude_model == config.claude_model_fast == "sonnet"
    assert adapter().model_fast == adapter().model_full == "sonnet"


async def test_preflight_fails_when_signed_out(claude):
    claude.behave(preflight_behaviour(logged_in=False))
    with pytest.raises(PreflightError, match="not signed in"):
        await preflight(adapter())


async def test_preflight_fails_when_the_probe_fails(claude):
    claude.behave(preflight_behaviour(probe=claude_result(REPLY, is_error=True)))
    with pytest.raises(PreflightError, match="probe turn failed"):
        await preflight(adapter())


def test_pair_stops_before_the_server_when_preflight_fails(claude, home):
    claude.behave(preflight_behaviour(logged_in=False))
    result = TyperRunner().invoke(
        app, ["pair", "--server", "http://localhost:9", "--adapter", "claude-code"]
    )
    assert result.exit_code == 1
    assert "not signed in" in result.output
    assert not (home / "credentials").exists()


async def test_a_turn_without_proposals_never_offers_one(claude):
    ctx = make_context(allowed_actions=["message", "question", "reply", "pass"])
    await take(adapter(), ctx)
    [call] = claude.calls()
    argv = call["argv"]
    schema = json.loads(argv[argv.index("--json-schema") + 1])
    assert schema["properties"]["action"]["enum"] == ["message", "question", "reply", "pass"]
    assert '"proposal"' not in " ".join(argv)
    assert '"proposal"' not in call["stdin"]
    assert '"proposal"' not in call["files"]["system.txt"]["text"]
    # The meeting data still shows the open proposals, for votes.
    assert "Spend $400 on a new shed roof" in call["stdin"]
