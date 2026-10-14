"""The `claude_code` adapter: the owner's own signed-in Claude Code answers, with no tools.

Every turn is one `claude -p` run with the flags of spec §4.2, verified on Claude Code
2.1.299: no built-in tools, no MCP, no settings, hooks, skills, slash commands or session
file, one model turn, and a JSON-schema answer. The rules go in a 0600 system-prompt file,
the fenced meeting data on stdin. An addressed turn (an @mention or a reply to this seat)
uses the fast model, a rotation turn the full one (§4.5); both default to sonnet.

Anything other than a clean structured answer passes: a non-zero exit, a timeout, an
error result, a refusal, missing or unparsable output, a denied tool use, a sub-agent or
web request, or a file left in the working directory.

One retry: on 2026-10-14 a quarter to a third of haiku turns, and one sonnet turn in
forty, ended `error_max_turns`: the model wrote its answer as plain text instead of
through the structured-output tool, and that text isn't returned. Such a turn runs once
more, on the full model, when there is time left. `--max-turns 1` stays as verified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..caps import Caps
from ..prompt import output_schema
from ..turnlog import TurnLog
from .base import TurnContext
from .cli_runner import (
    RATE_LIMIT,
    Attempt,
    CliAdapter,
    CliRunner,
    PreflightError,
    base_env,
    probe,
    run_quick,
    short,
)

RUNTIME_LABEL = "Claude Code on owner's computer"
SYSTEM_FILE = "system.txt"
RETRY_SUBTYPE = "error_max_turns"

# Claude Code reads these itself; see spec §4.2 for why each is set.
CLAUDE_ENV = {
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
    "DISABLE_AUTOUPDATER": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}


def claude_args(turn_dir: Path, *, model: str, effort: str) -> list[str]:
    """The confinement flags of spec §4.2, in its order. Nothing from the meeting is here."""
    return [
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
        "--model", model,
        "--effort", effort,
        "--system-prompt-file", str(turn_dir / SYSTEM_FILE),
        "--json-schema", json.dumps(output_schema(), separators=(",", ":")),
    ]  # fmt: skip


def claude_env(api_key: str | None = None) -> dict[str, str]:
    """`ANTHROPIC_API_KEY` only in API-key mode (D4); otherwise the subscription login is used."""
    extra = dict(CLAUDE_ENV)
    if api_key:
        extra["ANTHROPIC_API_KEY"] = api_key
    return base_env(extra)


def _tool_use(result: dict[str, Any]) -> str | None:
    """Why the result shows the model reaching for anything beyond its answer, or None."""
    if result.get("permission_denials"):
        return "Claude Code reports a denied tool use"
    spawned = (result.get("subagent_stats") or {}).get("spawned")
    if isinstance(spawned, int) and spawned > 0:
        return "Claude Code reports a sub-agent"
    server = (result.get("usage") or {}).get("server_tool_use") or {}
    if any(isinstance(v, int) and v > 0 for v in server.values()):
        return "Claude Code reports a web search or fetch"
    return None


def parse_result(stdout: str) -> Attempt:
    """`--output-format json`'s one result object as an attempt (spec §4.2, parsing)."""
    attempt = Attempt(raw=stdout)
    try:
        result = json.loads(stdout)
    except ValueError:
        attempt.reason = f"Claude Code's output isn't JSON: {short(stdout, 160) or '(empty)'}"
        return attempt
    if not isinstance(result, dict):
        attempt.reason = "Claude Code's output isn't a result object"
        return attempt
    attempt.details.update(
        {
            "api_ms": result.get("duration_api_ms"),
            "cost_usd_estimate": result.get("total_cost_usd"),
            "model_usage": result.get("modelUsage"),
            "stop_reason": result.get("stop_reason"),
            "subtype": result.get("subtype"),
        }
    )
    message = str(result.get("result") or "")
    if result.get("is_error") or result.get("subtype") != "success":
        attempt.reason = (
            f"Claude Code returned an error ({result.get('subtype')}): {short(message, 200)}"
        )
        attempt.rate_limited = bool(
            RATE_LIMIT.search(message) or result.get("api_error_status") == 429
        )
        return attempt
    if result.get("stop_reason") == "refusal":
        attempt.reason = "Claude declined to answer (refusal)"
        return attempt
    if reason := _tool_use(result):
        attempt.reason = reason
        return attempt
    answer = result.get("structured_output")
    if not isinstance(answer, dict):
        attempt.reason = "Claude Code returned no structured answer"
        return attempt
    attempt.answer = answer
    return attempt


class ClaudeCodeAdapter(CliAdapter):
    adapter_name = "claude_code"
    runtime_label = RUNTIME_LABEL

    def __init__(
        self,
        runner: CliRunner,
        *,
        model_fast: str = "sonnet",
        model_full: str = "sonnet",
        effort: str = "low",
        api_key: str | None = None,
        caps: Caps | None = None,
        turnlog: TurnLog | None = None,
        command: str = "claude",
    ) -> None:
        super().__init__(runner, caps=caps, turnlog=turnlog, command=command)
        self.model_fast = model_fast
        self.model_full = model_full
        self.effort = effort
        self.api_key = api_key

    def _model(self, addressed: bool) -> str:
        return self.model_fast if addressed else self.model_full

    async def _attempt(
        self, ctx: TurnContext, system: str, user: str, model: str | None
    ) -> Attempt:
        budget = self.runner.timeout_s
        attempt = await self._run(system, user, model or self.model_full, budget)
        spent = attempt.details.get("duration_s", budget)
        if attempt.details.get("subtype") == RETRY_SUBTYPE and spent < budget / 2:
            retry = await self._run(system, user, self.model_full, budget - spent)
            retry.details["retried"] = attempt.reason
            retry.details["retry_model"] = self.model_full
            retry.details["duration_s"] = round(spent + retry.details.get("duration_s", 0), 2)
            return retry
        return attempt

    async def _run(self, system: str, user: str, model: str, timeout_s: float) -> Attempt:
        result = await self.runner.run(
            self.command,
            lambda turn_dir: claude_args(turn_dir, model=model, effort=self.effort),
            env=claude_env(self.api_key),
            stdin=user,
            files={SYSTEM_FILE: system},
            timeout_s=timeout_s,
        )
        timing = {"duration_s": round(result.duration_s, 2)}
        if result.timed_out:
            return Attempt(
                reason=f"Claude Code took longer than {timeout_s:.0f}s",
                details=timing,
            )
        if result.leftovers:
            return Attempt(
                reason=f"Claude Code left files behind: {short(', '.join(result.leftovers), 200)}",
                raw=result.stdout,
                details=timing,
            )
        attempt = parse_result(result.stdout)
        attempt.details.update(timing)
        if result.returncode != 0 and attempt.reason is None:
            attempt.reason = f"Claude Code exited {result.returncode}: {short(result.stderr, 200)}"
        if attempt.reason is not None and not attempt.rate_limited:
            attempt.rate_limited = bool(RATE_LIMIT.search(result.stderr))
        return attempt


async def preflight(adapter: ClaudeCodeAdapter) -> str:
    """`claude --version`, `claude auth status`, then one confined probe turn (spec §3.2).

    Returns e.g. `Claude Code 2.1.299, signed in as a@b.c, probe OK (4.1 s)`."""
    summary = await facts(adapter)
    action, seconds = await probe(adapter)
    return f"{summary}, probe OK ({seconds:.1f} s, {action.action})"


async def facts(adapter: ClaudeCodeAdapter) -> str:
    """The preflight without the probe turn, for `diagnose`: e.g.
    `Claude Code 2.1.299, signed in as a@b.c`. Raises PreflightError."""
    env = claude_env(adapter.api_key)
    runner = adapter.runner
    version = await run_quick(runner, adapter.command, ["--version"], env)
    if version.returncode != 0:
        raise PreflightError(f"`claude --version` failed: {short(version.stderr, 200)}")
    number = (version.stdout.split() or ["?"])[0]
    if adapter.api_key:
        who = "using an API key"
    else:
        status = await run_quick(runner, adapter.command, ["auth", "status"], env)
        try:
            auth = json.loads(status.stdout)
        except ValueError:
            auth = {}
        if not isinstance(auth, dict) or auth.get("loggedIn") is not True:
            raise PreflightError(
                "Claude Code is not signed in: run `claude`, sign in, then try again"
            )
        who = f"signed in as {auth.get('email') or auth.get('authMethod') or 'your account'}"
    return f"Claude Code {number}, {who}"
