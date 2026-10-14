import json
import os
from pathlib import Path

import pytest

from yantraxa_bridge.adapters import make_turn_handler
from yantraxa_bridge.adapters.base import Final
from yantraxa_bridge.adapters.cli_runner import CliRunner, PreflightError
from yantraxa_bridge.adapters.codex_cli import (
    DISABLED_FEATURES,
    CodexCliAdapter,
    parse_events,
    preflight,
    strict_schema,
)
from yantraxa_bridge.caps import Caps
from yantraxa_bridge.config import BridgeConfig
from yantraxa_bridge.prompt import output_schema
from yantraxa_bridge.turnlog import TurnLog, read_events

from .contexts import M2, P1, make_context
from .fake_cli import codex_events, install

pytestmark = pytest.mark.skipif(os.name == "nt", reason="fake CLIs are POSIX scripts")

REPLY = {
    "action": "reply",
    "reply_to": "m2",
    "text": "Saugat can help fix the roof on Saturday.",
    "vote": {"motion_id": "p1", "choice": "support", "reason": "It leaks now."},
}


@pytest.fixture
def codex(monkeypatch, tmp_path):
    fake = install(monkeypatch, tmp_path, "codex")
    fake.answer(codex_events(REPLY))
    return fake


@pytest.fixture
def codex_home(tmp_path):
    return tmp_path / "yantraxa-home" / "codex"


def adapter(codex_home, **kw) -> CodexCliAdapter:
    return CodexCliAdapter(CliRunner(timeout_s=10), codex_home=codex_home, **kw)


def turns(fake):
    """The `codex exec` runs, without the `--version` check before each."""
    return [call for call in fake.calls() if call["argv"][:1] == ["exec"]]


async def take(adapter, ctx=None):
    events = [e async for e in adapter.take_turn(ctx or make_context())]
    assert len(events) == 1 and isinstance(events[0], Final)
    return events[0].actions


# ------------------------------------------------------------------ the invocation


async def test_the_exact_confined_invocation(codex, codex_home):
    await take(adapter(codex_home))
    version, call = codex.calls()
    assert version["argv"] == ["--version"]
    turn_dir = Path(call["cwd"])
    argv = call["argv"]
    cd = argv[argv.index("-C") + 1]
    schema = argv[argv.index("--output-schema") + 1]
    assert Path(cd).resolve() == turn_dir and Path(schema).resolve() == turn_dir / "schema.json"
    disables = [arg for name in DISABLED_FEATURES for arg in ("--disable", name)]
    assert argv == [
        "exec",
        "--ephemeral", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check",
        "-s", "read-only",
        "-c", 'approval_policy="never"',
        "-c", 'web_search="disabled"',
        "-c", "mcp_servers={}",
        "-c", "skills.bundled.enabled=false",
        "-c", 'model_reasoning_effort="low"',
        *disables,
        "--json", "-C", cd, "--output-schema", schema, "-",
    ]  # fmt: skip
    assert "-m" not in argv  # the account's default model unless one is set


def test_every_feature_the_spec_disables_is_disabled():
    assert set(DISABLED_FEATURES) == {
        "shell_tool", "unified_exec", "code_mode_host", "multi_agent", "image_generation",
        "apps", "plugins", "hooks", "goals", "view_image", "sleep_tool", "computer_use",
        "browser_use", "browser_use_external", "browser_use_full_cdp_access", "in_app_browser",
        "in_app_local_automation", "skill_search", "tool_suggest", "realtime_conversation",
        "remote_plugin", "workspace_dependencies", "skill_mcp_dependency_install", "memories",
    }  # fmt: skip


async def test_a_model_and_effort_when_configured(codex, codex_home):
    await take(adapter(codex_home, model="gpt-5.6-terra", effort="medium"))
    argv = turns(codex)[0]["argv"]
    assert argv[argv.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="medium"' in argv


async def test_its_own_codex_home_and_a_stripped_environment(codex, codex_home, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-parent-key-should-not-pass")
    monkeypatch.setenv("CODEX_HOME", str(Path.home() / ".codex"))
    await take(adapter(codex_home))
    env = turns(codex)[0]["env"]
    assert env["CODEX_HOME"] == str(codex_home)
    assert codex_home.is_dir() and oct(codex_home.stat().st_mode & 0o777) == "0o700"
    assert oct(codex_home.parent.stat().st_mode & 0o777) == "0o700"  # the new home too
    assert "OPENAI_API_KEY" not in env and "CODEX_API_KEY" not in env


async def test_every_run_gets_an_empty_private_home_and_utc(codex, codex_home):
    await take(adapter(codex_home))
    homes = []
    for call in codex.calls():  # the version check and the turn
        env = call["env"]
        home = Path(env["HOME"])
        assert home != Path.home() and home.name.startswith("yantraxa-turn-home-")
        assert env["TZ"] == "UTC"
        homes.append(home)
    assert len(set(homes)) == 2  # a fresh one per run
    assert not any(home.exists() for home in homes)  # deleted afterwards


async def test_api_key_mode_passes_codex_api_key(codex, codex_home):
    await take(adapter(codex_home, api_key="sk-proj-owner-chose-this"))
    assert turns(codex)[0]["env"]["CODEX_API_KEY"] == "sk-proj-owner-chose-this"


async def test_rules_then_meeting_data_on_stdin_and_a_strict_schema_file(codex, codex_home):
    await take(adapter(codex_home))
    [call] = turns(codex)
    stdin = call["stdin"]
    assert stdin.startswith("You are Saugat's agent")
    assert stdin.index("no tools") < stdin.index("<<meeting_data>>\n")
    assert "Saturday mornings" not in " ".join(call["argv"])
    schema = call["files"]["schema.json"]
    assert schema["mode"] == "0o600"
    assert json.loads(schema["text"]) == strict_schema(output_schema())


def test_strict_schema_closes_both_objects():
    strict = strict_schema(output_schema())
    assert strict["additionalProperties"] is False
    assert strict["required"] == ["action", "reply_to", "text", "vote"]
    vote = next(s for s in strict["properties"]["vote"]["anyOf"] if s.get("type") == "object")
    assert vote["additionalProperties"] is False
    assert vote["required"] == ["motion_id", "choice", "reason"]
    assert "additionalProperties" not in output_schema()  # prompt.py's own is unchanged


# ------------------------------------------------------------------ answers


async def test_the_last_agent_message_becomes_actions_with_real_ids(codex, codex_home):
    [action] = await take(adapter(codex_home))
    assert action.action == "reply" and action.reply_to == M2
    assert action.vote is not None and action.vote.motion_id == P1


def test_transient_errors_and_notices_before_an_answer_are_fine():
    stream = codex_events(
        REPLY,
        {"type": "error", "message": "Reconnecting... 1/5 (stream disconnected)"},
        {"type": "item.completed", "item": {"id": "item_1", "type": "reasoning", "text": "…"}},
    )
    attempt = parse_events(stream)
    assert attempt.reason is None and attempt.answer == REPLY
    assert attempt.details["usage"]["output_tokens"] == 80


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "not json at all",
        codex_events(None),
        codex_events("not json"),
        codex_events({"action": "dance", "text": "x"}),
        codex_events(REPLY, {"type": "turn.failed", "error": {"message": "boom"}}),
        codex_events(
            REPLY,
            {
                "type": "item.started",
                "item": {"id": "i", "type": "command_execution", "command": "echo hi"},
            },
        ),
        codex_events(REPLY, {"type": "item.completed", "item": {"id": "i", "type": "file_change"}}),
        codex_events(
            REPLY, {"type": "item.completed", "item": {"id": "i", "type": "mcp_tool_call"}}
        ),
        codex_events(REPLY, {"type": "item.completed", "item": {"id": "i", "type": "web_search"}}),
    ],
)
async def test_anything_but_a_clean_answer_passes(codex, codex_home, stdout):
    codex.answer(stdout)
    assert [a.action for a in await take(adapter(codex_home))] == ["pass"]


async def test_a_non_zero_exit_passes(codex, codex_home):
    codex.answer(codex_events(REPLY), exit=1)
    assert [a.action for a in await take(adapter(codex_home))] == ["pass"]


async def test_a_file_left_behind_passes(codex, codex_home):
    codex.answer(codex_events(REPLY), touch="PWNED.txt")
    assert [a.action for a in await take(adapter(codex_home))] == ["pass"]


def test_a_usage_limit_is_recognised():
    attempt = parse_events(
        codex_events(
            None,
            {"type": "turn.failed", "error": {"message": "You've hit your usage limit."}},
        )
    )
    assert attempt.reason is not None and attempt.rate_limited


async def test_caps_and_the_log(codex, codex_home, home):
    a = adapter(codex_home, caps=Caps(home, per_meeting=1, per_day=200), turnlog=TurnLog(home))
    assert (await take(a))[0].action == "reply"
    assert (await take(a))[0].action == "pass"
    assert len(turns(codex)) == 1 and len(codex.calls()) == 2  # a capped turn runs nothing
    first, second = read_events(home)
    assert first["adapter"] == "codex_cli" and first["usage"]["input_tokens"] == 14000
    assert first["cli_version"] == "0.158.0"
    assert "<<meeting_data>>" in first["prompt"]
    assert "max_turns_per_meeting" in second["pass_reason"]


def test_make_turn_handler_builds_the_codex_adapter(home):
    handler = make_turn_handler(BridgeConfig(adapter="codex-cli"))
    assert isinstance(handler.adapter, CodexCliAdapter)
    assert handler.adapter.codex_home == home / "codex"
    assert handler.adapter.model is None
    custom = make_turn_handler(BridgeConfig(adapter="codex-cli", codex_home="/tmp/x-codex"))
    assert custom.adapter.codex_home == Path("/tmp/x-codex")


# ------------------------------------------------------------------ preflight


def preflight_behaviour():
    return {
        "--version": {"stdout": "codex-cli 0.158.0\n"},
        "login status": {
            "exit": 1,
            "stdout": "Not logged in\n",
            "marker": "auth.json",
            "then": {"exit": 0, "stdout": "Logged in using ChatGPT\n"},
        },
        "login": {"write_marker": "auth.json"},
        "default": {"stdout": codex_events({"action": "reply", "reply_to": "m1", "text": "Yes."})},
    }


async def test_preflight_without_a_sign_in_says_what_to_run(codex, codex_home):
    codex.behave(preflight_behaviour())
    with pytest.raises(PreflightError, match=r'CODEX_HOME=".*codex" codex login'):
        await preflight(adapter(codex_home), allow_login=False)
    assert ["login"] not in [call["argv"] for call in codex.calls()]


async def test_preflight_signs_in_once_in_its_own_home(codex, codex_home, capsys):
    codex.behave(preflight_behaviour())
    summary = await preflight(adapter(codex_home), allow_login=True)
    assert summary.startswith(f"Codex 0.158.0, signed in (own home {codex_home}), probe OK (")
    argvs = [call["argv"][:2] for call in codex.calls()]
    assert ["login"] in argvs
    login = next(call for call in codex.calls() if call["argv"] == ["login"])
    assert login["env"]["CODEX_HOME"] == str(codex_home)
    assert (codex_home / "auth.json").exists()
    assert "open your browser once" in capsys.readouterr().out
    await preflight(adapter(codex_home), allow_login=False)  # now signed in: no login


# ------------------------------------------------------------------ the version allow-list


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("codex-cli 0.158.0\n", "0.158.0"),
        ("codex-cli 0.158.12\n", "0.158.12"),
        ("codex-cli 0.159.0-alpha.1\n", "0.159.0-alpha.1"),
        ('{"type": "thread.started", "n": 80}}', None),
        ("", None),
    ],
)
def test_the_version_is_read_from_codex_version(stdout, expected):
    from yantraxa_bridge.adapters.cli_runner import CliResult
    from yantraxa_bridge.adapters.codex_cli import cli_version

    assert cli_version(CliResult(0, stdout, "", 0.1)) == expected
    assert cli_version(CliResult(1, stdout, "", 0.1)) is None


@pytest.mark.parametrize("version", ["0.158.0", "0.158.7"])
def test_only_0_158_is_tested(version):
    from yantraxa_bridge.adapters.codex_cli import tested

    assert tested(version)
    assert not tested("0.159.0") and not tested("0.157.9") and not tested("1.158.0")
    assert not tested(None)


@pytest.mark.parametrize("version_stdout", ["codex-cli 0.159.0\n", "garbled\n"])
async def test_an_untested_version_passes_every_turn_without_running_it(
    codex, codex_home, home, version_stdout, caplog
):
    codex.behave(
        {"--version": {"stdout": version_stdout}, "default": {"stdout": codex_events(REPLY)}}
    )
    a = adapter(codex_home, turnlog=TurnLog(home))
    assert (await take(a))[0].action == "pass"
    assert turns(codex) == []  # the turn itself never ran
    [event] = read_events(home)
    assert "hasn't been checked with this bridge (tested: 0.158.x)" in event["pass_reason"]
    assert "codex_allow_untested = true" in event["pass_reason"]
    assert "hasn't been checked" in caplog.text


async def test_the_version_is_checked_again_on_every_turn(codex, codex_home):
    a = adapter(codex_home)
    assert (await take(a))[0].action == "reply"
    codex.behave(
        {"--version": {"stdout": "codex-cli 0.160.0\n"}, "default": {"stdout": codex_events(REPLY)}}
    )
    assert (await take(a))[0].action == "pass"  # upgraded under the running bridge
    assert [c["argv"][:1] for c in codex.calls()] == [["--version"], ["exec"], ["--version"]]


async def test_codex_allow_untested_lets_an_untested_version_answer(codex, codex_home, home):
    codex.behave(
        {"--version": {"stdout": "codex-cli 0.159.0\n"}, "default": {"stdout": codex_events(REPLY)}}
    )
    a = adapter(codex_home, allow_untested=True, turnlog=TurnLog(home))
    assert (await take(a))[0].action == "reply"
    assert read_events(home)[0]["cli_version"] == "0.159.0"


async def test_doctor_says_how_to_proceed_on_an_untested_version(codex, codex_home):
    from yantraxa_bridge.adapters.codex_cli import facts

    behaviour = preflight_behaviour()
    behaviour["--version"] = {"stdout": "codex-cli 0.159.0\n"}
    codex.behave(behaviour)
    with pytest.raises(PreflightError) as error:
        await facts(adapter(codex_home))
    assert "Codex 0.159.0 hasn't been checked" in str(error.value)
    assert "npm install -g @openai/codex@0.158" in str(error.value)
    assert "codex_allow_untested = true" in str(error.value)
    summary = await facts(adapter(codex_home, allow_untested=True), allow_login=True)
    assert summary.startswith("Codex 0.159.0 (untested; codex_allow_untested is on), signed in")


def test_codex_allow_untested_is_off_by_default_and_reaches_the_adapter(home):
    assert BridgeConfig().codex_allow_untested is False
    assert make_turn_handler(BridgeConfig(adapter="codex-cli")).adapter.allow_untested is False
    handler = make_turn_handler(BridgeConfig(adapter="codex-cli", codex_allow_untested=True))
    assert handler.adapter.allow_untested is True


# ------------------------------------------------------------------ hostile tool calls (replay)

# What Codex 0.158.0 printed when a mock model called tools (security review, 2026-10-14):
# a rejected call leaves no JSONL item at all, only one of these stderr lines.
ROUTER_LINES = [
    "2026-10-14T10:40:24.295237Z ERROR codex_core::tools::router: error=unsupported call: "
    "request_user_input_async",
    "2026-10-14T10:40:33.500239Z ERROR codex_core::tools::router: error=code-mode host is disabled",
    "2026-10-14T10:40:33.508270Z ERROR codex_core::tools::router: error=collab spawn failed: "
    "Fatal error: failed to load model context for thread 01a1",
    "2026-10-14T10:40:35.064155Z ERROR codex_core::tools::router: error=patch rejected: writing "
    "is blocked by read-only sandbox; rejected by user approval settings",
]


@pytest.mark.parametrize("line", ROUTER_LINES)
async def test_a_tool_call_codex_rejected_itself_passes(codex, codex_home, home, line):
    # The model asked for a tool, Codex refused it, and the model then gave a clean answer.
    codex.answer(codex_events(REPLY), stderr=line + "\n")
    a = adapter(codex_home, turnlog=TurnLog(home))
    assert (await take(a))[0].action == "pass"
    [event] = read_events(home)
    assert event["pass_reason"].startswith("Codex tried to use a tool (stderr: ")
    assert "tools::router" in event["pass_reason"]


@pytest.mark.parametrize(
    "item",
    [
        {"type": "command_execution", "command": "id", "status": "in_progress"},
        {"type": "file_change", "changes": [{"path": "/tmp/x", "kind": "add"}]},
        {"type": "mcp_tool_call", "server": "x", "tool": "y"},
        {"type": "web_search", "query": "owner's address"},
        {"type": "collab_tool_call", "tool": "spawn_agent"},
    ],
)
async def test_a_replayed_tool_item_passes_even_with_an_answer_after_it(codex, codex_home, item):
    started = {"type": "item.started", "item": {"id": "item_1", **item}}
    codex.answer(codex_events(REPLY, started), stderr=ROUTER_LINES[0] + "\n")
    assert (await take(adapter(codex_home)))[0].action == "pass"
    codex.answer(codex_events(REPLY, started))  # the item alone is enough
    assert (await take(adapter(codex_home)))[0].action == "pass"


async def test_ordinary_stderr_noise_does_not_pass_a_turn(codex, codex_home):
    codex.answer(
        codex_events(REPLY),
        stderr="2026-10-14T10:40:24Z WARN codex_core::config: something unrelated\n",
    )
    assert (await take(adapter(codex_home)))[0].action == "reply"


async def test_a_hung_version_check_passes_quickly(codex, codex_home, monkeypatch):
    import time

    from yantraxa_bridge.adapters import codex_cli

    monkeypatch.setattr(codex_cli, "VERSION_TIMEOUT_S", 0.5)
    codex.behave({"--version": {"sleep": 30}, "default": {"stdout": codex_events(REPLY)}})
    started = time.monotonic()
    assert (await take(adapter(codex_home)))[0].action == "pass"
    assert time.monotonic() - started < 5 and turns(codex) == []


async def test_bundled_skills_are_off_so_their_root_path_never_reaches_the_model(codex, codex_home):
    # With them on, Codex 0.158 lists `CODEX_HOME/skills/.system` to the model, and
    # CODEX_HOME is under the owner's home, so the path names them (review of 1c).
    await take(adapter(codex_home))
    argv = turns(codex)[0]["argv"]
    assert argv[argv.index("skills.bundled.enabled=false") - 1] == "-c"


async def test_a_left_behind_file_name_is_shortened_and_has_no_control_characters(
    codex, codex_home, home, caplog
):
    codex.answer(codex_events(REPLY), touch="\x1b]0;owned\x07PWNED.txt")
    assert (await take(adapter(codex_home, turnlog=TurnLog(home))))[0].action == "pass"
    [event] = read_events(home)
    assert event["pass_reason"] == "Codex left files behind: ]0;ownedPWNED.txt"
    assert "\x1b" not in caplog.text and "\x07" not in caplog.text


async def test_a_turn_without_proposals_never_offers_one(codex, codex_home):
    ctx = make_context(allowed_actions=["message", "question", "reply", "pass"])
    await take(adapter(codex_home), ctx)
    [call] = turns(codex)
    schema = json.loads(call["files"]["schema.json"]["text"])
    assert schema["properties"]["action"]["enum"] == ["message", "question", "reply", "pass"]
    assert '"proposal"' not in " ".join(call["argv"])
    assert '"proposal"' not in call["stdin"]
    assert "Spend $400 on a new shed roof" in call["stdin"]
