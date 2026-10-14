"""The `codex_cli` adapter: the owner's own Codex answers, with no tools, from its own home.

Every turn is one `codex exec` run with the flags of spec §4.3, verified on Codex 0.158.0.
Codex's `read-only` sandbox alone still runs shell commands, so every tool-bearing feature is
disabled by name as well, web search and MCP servers are off, approvals are never granted,
and `CODEX_HOME` is the bridge's own (`~/.yantraxa/codex`, D5) so the owner's
`~/.codex` config, `AGENTS.md`, hooks, plugins, skills and sign-in are never read. `HOME`
is a fresh empty directory for every run, so Codex can't list the owner's
`~/.agents/skills` (names, descriptions, paths) to the model either, and
`skills.bundled.enabled=false` stops it listing its own bundled skills, whose root,
`CODEX_HOME/skills/.system`, is under the owner's home and so names them. `TZ=UTC` keeps
the owner's timezone out of the context it sends on macOS (on Linux Codex reads
`/etc/localtime` instead).

This confinement is a deny-list, checked on one Codex version: a new version may add a tool
under a new name. So every turn first runs `codex --version`, logs it, and passes unless it
is a tested version (`TESTED_CODEX`) or the owner set `codex_allow_untested = true`.

The rules go on stdin ahead of the fenced meeting data. Codex's `developer_instructions`
setting would put them in the developer role, but only through `-c` on the command line,
which would show the owner's approved facts in `ps`; so it isn't used (§4.1).

The answer is the last `agent_message` in the `--json` event stream. A failed turn, no
answer, or any item other than reasoning, notices and the answer (a command, a file change,
an MCP or web call, a sub-agent) passes. So does any tool call Codex rejected itself: that
leaves no item in the stream, only a `codex_core::tools::router` line on stderr.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from ..caps import Caps
from ..config import ensure_private_dir
from ..prompt import output_schema
from ..turnlog import TurnLog
from .base import TurnContext
from .cli_runner import (
    RATE_LIMIT,
    Attempt,
    CliAdapter,
    CliResult,
    CliRunner,
    PreflightError,
    base_env,
    probe,
    run_quick,
    short,
)

log = logging.getLogger("yantraxa_bridge.adapters.codex")

RUNTIME_LABEL = "Codex on owner's computer"
SCHEMA_FILE = "schema.json"
# The Codex versions this confinement was checked on, by replaying hostile tool calls.
TESTED_CODEX = re.compile(r"^0\.158\.\d+$")
TESTED_CODEX_LABEL = "0.158.x"
VERSION_TIMEOUT_S = 5.0  # the per-turn `codex --version`; a hung one mustn't eat the turn
VERSION = re.compile(r"(?<![\w.])(\d+\.\d+\.\d+(?:[-+][\w.]+)?)(?![\w.])")
# A tool call Codex refused by itself: no JSONL item, just a tracing line on stderr from
# a `tools` module, e.g. `ERROR codex_core::tools::router: error=unsupported call: …`.
TOOL_ATTEMPT = re.compile(r"\btools::\w+|\bunsupported call\b|\bpatch rejected\b")

# Every feature that gives the model a tool, a runtime or a way out (spec §4.3).
DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "code_mode_host",
    "multi_agent",
    "image_generation",
    "apps",
    "plugins",
    "hooks",
    "goals",
    "view_image",
    "sleep_tool",
    "computer_use",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "in_app_browser",
    "in_app_local_automation",
    "skill_search",
    "tool_suggest",
    "realtime_conversation",
    "remote_plugin",
    "workspace_dependencies",
    "skill_mcp_dependency_install",
    "memories",
)
# Items that are only the model talking: its reasoning, Codex's notices, the answer.
ANSWER_ITEMS = {"agent_message", "reasoning", "error"}


def codex_args(turn_dir: Path, *, effort: str, model: str | None = None) -> list[str]:
    """The confinement flags of spec §4.3, in its order. Nothing from the meeting is here."""
    args = [
        "exec",
        "--ephemeral", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check",
        "-s", "read-only",
        "-c", 'approval_policy="never"',
        "-c", 'web_search="disabled"',
        "-c", "mcp_servers={}",
        "-c", "skills.bundled.enabled=false",
        "-c", f"model_reasoning_effort={json.dumps(effort)}",
    ]  # fmt: skip
    if model:
        args += ["-m", model]
    for feature in DISABLED_FEATURES:
        args += ["--disable", feature]
    args += ["--json", "-C", str(turn_dir), "--output-schema", str(turn_dir / SCHEMA_FILE), "-"]
    return args


def codex_env(codex_home: Path, api_key: str | None = None) -> dict[str, str]:
    """`CODEX_API_KEY` only in API-key mode (D4). Every run replaces `HOME` with an empty
    directory (`CliRunner.run(private_home=True)`); the sign-in lives in `CODEX_HOME`."""
    extra = {"CODEX_HOME": str(codex_home)}
    if api_key:
        extra["CODEX_API_KEY"] = api_key
    return base_env(extra)


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """`prompt.output_schema()` as `--output-schema` needs it: every object closed
    (`additionalProperties: false`) with all its properties required."""
    strict = copy.deepcopy(schema)

    def close(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                close(value)
        elif isinstance(node, list):
            for value in node:
                close(value)

    close(strict)
    return strict


def cli_version(result: CliResult) -> str | None:
    """`0.158.0` from `codex --version`'s `codex-cli 0.158.0`, or None."""
    if result.returncode != 0:
        return None
    found = VERSION.search(result.stdout.strip().split("\n")[0])
    return found.group(1) if found else None


def tested(version: str | None) -> bool:
    return version is not None and TESTED_CODEX.match(version) is not None


def untested_message(version: str | None) -> str:
    """What `doctor` and a passing turn say about a Codex version this bridge wasn't
    checked on, and how to proceed."""
    which = f"Codex {version}" if version else "This Codex (its version couldn't be read)"
    return (
        f"{which} hasn't been checked with this bridge (tested: {TESTED_CODEX_LABEL}), so it "
        "passes every turn: a new version can add tools the bridge doesn't know to turn off. "
        f"Install Codex {TESTED_CODEX_LABEL} (`npm install -g @openai/codex@0.158`), or update "
        "yantraxa-bridge, or accept the risk with `codex_allow_untested = true` in config.toml"
    )


def tool_attempt(stderr: str) -> str | None:
    """The first stderr line showing a tool call Codex rejected itself, or None."""
    for line in stderr.splitlines():
        if TOOL_ATTEMPT.search(line):
            return line
    return None


def parse_events(stdout: str) -> Attempt:
    """The `--json` JSONL stream as an attempt (spec §4.3, parsing)."""
    attempt = Attempt(raw=stdout)
    answer: str | None = None
    failed: str | None = None
    errors: list[str] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind in ("item.started", "item.updated", "item.completed"):
            item = event.get("item") if isinstance(event.get("item"), dict) else {}
            item_type = item.get("type")
            if item_type not in ANSWER_ITEMS:
                attempt.reason = f"Codex tried to use a tool ({item_type})"
                return attempt
            if kind == "item.completed" and item_type == "agent_message":
                answer = item.get("text") if isinstance(item.get("text"), str) else None
        elif kind == "turn.failed":
            error = event.get("error")
            failed = str(error.get("message") if isinstance(error, dict) else error)
        elif kind == "error":
            errors.append(str(event.get("message", "")))
        elif kind == "turn.completed":
            attempt.details["usage"] = event.get("usage")
    if failed is not None:
        attempt.reason = f"Codex's turn failed: {short(failed, 200)}"
        attempt.rate_limited = bool(RATE_LIMIT.search(failed))
        return attempt
    if answer is None:
        last = errors[-1] if errors else ""
        attempt.reason = "Codex gave no answer" + (f": {short(last, 200)}" if last else "")
        attempt.rate_limited = bool(RATE_LIMIT.search(last))
        return attempt
    try:
        attempt.answer = json.loads(answer, strict=False)
    except ValueError:
        attempt.reason = f"Codex's answer isn't JSON: {short(answer, 160)}"
    return attempt


class CodexCliAdapter(CliAdapter):
    adapter_name = "codex_cli"
    runtime_label = RUNTIME_LABEL

    def __init__(
        self,
        runner: CliRunner,
        *,
        codex_home: Path,
        model: str | None = None,
        effort: str = "low",
        api_key: str | None = None,
        allow_untested: bool = False,
        caps: Caps | None = None,
        turnlog: TurnLog | None = None,
        command: str = "codex",
    ) -> None:
        super().__init__(runner, caps=caps, turnlog=turnlog, command=command)
        self.codex_home = codex_home
        self.model = model
        self.effort = effort
        self.api_key = api_key
        self.allow_untested = allow_untested

    def ensure_home(self) -> None:
        """Codex needs its home to exist already (§4.3); it holds a sign-in, so 0700."""
        ensure_private_dir(self.codex_home)
        os.chmod(self.codex_home, 0o700)

    def _model(self, addressed: bool) -> str | None:
        return self.model  # Codex has no faster tier; an addressed turn uses the same (§4.5)

    async def version(self) -> str | None:
        """`codex --version`, e.g. `0.158.0`; None when it can't be read. Raises CliNotFound."""
        result = await self.runner.run(
            self.command,
            lambda _dir: ["--version"],
            env=codex_env(self.codex_home, self.api_key),
            timeout_s=min(VERSION_TIMEOUT_S, self.runner.timeout_s),
            private_home=True,
        )
        return cli_version(result)

    async def _attempt(
        self, ctx: TurnContext, system: str, user: str, model: str | None
    ) -> Attempt:
        self.ensure_home()
        # Checked on every turn: Codex can be upgraded under a running bridge.
        version = await self.version()
        log.info("codex_cli turn %s: Codex %s", ctx.turn_id, version or "(version unreadable)")
        if not tested(version):
            if not self.allow_untested:
                return Attempt(reason=untested_message(version), details={"cli_version": version})
            log.warning("Codex %s is untested; codex_allow_untested is on", version)
        result = await self.runner.run(
            self.command,
            lambda turn_dir: codex_args(turn_dir, effort=self.effort, model=model),
            env=codex_env(self.codex_home, self.api_key),
            stdin=f"{system}\n\n{user}",
            files={SCHEMA_FILE: json.dumps(strict_schema(output_schema()))},
            private_home=True,
        )
        timing = {"duration_s": round(result.duration_s, 2), "cli_version": version}
        if result.timed_out:
            return Attempt(
                reason=f"Codex took longer than {self.runner.timeout_s:.0f}s", details=timing
            )
        if result.leftovers:
            return Attempt(
                reason=f"Codex left files behind: {short(', '.join(result.leftovers), 200)}",
                raw=result.stdout,
                details=timing,
            )
        if line := tool_attempt(result.stderr):
            return Attempt(
                reason=f"Codex tried to use a tool (stderr: {short(line, 200)})",
                raw=result.stdout,
                details=timing,
            )
        attempt = parse_events(result.stdout)
        attempt.details.update(timing)
        if result.returncode != 0 and attempt.reason is None:
            attempt.reason = f"Codex exited {result.returncode}: {short(result.stderr, 200)}"
        if attempt.reason is not None and not attempt.rate_limited:
            attempt.rate_limited = bool(RATE_LIMIT.search(result.stderr))
        return attempt


def login_command(codex_home: Path) -> str:
    """What the owner runs to give the bridge its own Codex sign-in."""
    return f'mkdir -p "{codex_home}" && CODEX_HOME="{codex_home}" codex login'


async def signed_in(adapter: CodexCliAdapter) -> bool:
    """`codex login status` as a turn sees it: the bridge's home, an empty `HOME`."""
    adapter.ensure_home()
    status = await run_quick(
        adapter.runner,
        adapter.command,
        ["login", "status"],
        codex_env(adapter.codex_home),
        private_home=True,
    )
    return status.returncode == 0


async def login(adapter: CodexCliAdapter) -> int:
    """`codex login` in the bridge's own home, on this terminal (the browser opens once).

    The owner's real `HOME` is kept here: no model runs, and the sign-in goes to
    `CODEX_HOME`, where `signed_in()` then checks it with an empty `HOME`."""
    env = codex_env(adapter.codex_home)
    env["TERM"] = os.environ.get("TERM", "xterm")
    executable = adapter.runner.resolve(adapter.command, env)
    process = await asyncio.create_subprocess_exec(executable, "login", env=env)
    return await process.wait()


async def preflight(adapter: CodexCliAdapter, *, allow_login: bool) -> str:
    """`codex --version`, the bridge's own sign-in (signing in once if `allow_login`), then
    one confined probe turn (spec §3.2)."""
    summary = await facts(adapter, allow_login=allow_login)
    action, seconds = await probe(adapter)
    return f"{summary}, probe OK ({seconds:.1f} s, {action.action})"


async def facts(adapter: CodexCliAdapter, *, allow_login: bool = False) -> str:
    """The preflight without the probe turn, for `diagnose`: e.g.
    `Codex 0.158.0, signed in (own home …)`. Raises PreflightError."""
    env = codex_env(adapter.codex_home, adapter.api_key)
    version = await run_quick(
        adapter.runner, adapter.command, ["--version"], env, private_home=True
    )
    if version.returncode != 0:
        raise PreflightError(f"`codex --version` failed: {short(version.stderr, 200)}")
    number = cli_version(version) or "?"
    if not tested(number):
        if not adapter.allow_untested:
            raise PreflightError(untested_message(number))
        number += " (untested; codex_allow_untested is on)"
    if adapter.api_key:
        who = "using an API key"
    else:
        if not await signed_in(adapter):
            if not allow_login:
                raise PreflightError(
                    "the bridge's Codex home has no sign-in yet; run: "
                    + login_command(adapter.codex_home)
                )
            print(
                "Codex will open your browser once so the bridge gets its own sign-in "
                f"(kept in {adapter.codex_home}, separate from ~/.codex).",
                flush=True,
            )
            if await login(adapter) != 0 or not await signed_in(adapter):
                raise PreflightError("the Codex sign-in didn't finish; run `pair` again")
        who = f"signed in (own home {adapter.codex_home})"
    return f"Codex {number}, {who}"
