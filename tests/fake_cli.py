"""Fake `claude` and `codex` executables: small Python scripts on PATH. No vendor is called.

The CLI runner strips the environment, so a fake can't be steered by environment variables.
It reads `behaviour.json` beside itself instead, and appends what it saw (argv, environment,
working directory and its contents, stdin) to `calls.jsonl` there.

A behaviour maps a key to a response. The key is the first two arguments (`"auth status"`),
then the first (`"--version"`), then `"default"`. A response may set `stdout`, `stderr`,
`exit`, `sleep` (seconds), `sequence` (one response per call, the last repeating),
`touch` (a file to create in the working directory), `spawn` (start a grandchild
`sleep` that holds the output pipes, for kill tests), `spawn_detached` (the same with its
stdio on /dev/null), `marker` (a file under `$CODEX_HOME` whose existence switches
to the `then` response) and `write_marker` (create that file).
"""

from __future__ import annotations

import json
import os
import shlex
import stat
import sys
from pathlib import Path
from typing import Any

SCRIPT = r"""import json, os, subprocess, sys, time
from pathlib import Path

here = Path(__file__).resolve().parent  # the .py beside the stub
args = sys.argv[1:]
stdin = sys.stdin.read() if ("-" in args or "-p" in args) else ""  # a turn
cwd = Path.cwd()
record = {{
    "argv": args,
    "env": dict(os.environ),
    "cwd": str(cwd),
    "cwd_mode": oct(cwd.stat().st_mode & 0o777),
    "files": {{p.name: {{"text": p.read_text(), "mode": oct(p.stat().st_mode & 0o777)}}
               for p in cwd.iterdir() if p.is_file()}},
    "stdin": stdin,
    "pid": os.getpid(),
}}
behaviour = json.loads((here / "behaviour.json").read_text())
response = None
for key in (" ".join(args[:2]), args[0] if args else "", "default"):
    if key in behaviour:
        response = behaviour[key]
        break
response = response or {{}}
if "sequence" in response:  # one response per call, the last repeating
    calls = here / "calls.jsonl"
    done = len(calls.read_text().splitlines()) if calls.exists() else 0
    response = response["sequence"][min(done, len(response["sequence"]) - 1)]
marker = response.get("marker")
if marker and (Path(os.environ.get("CODEX_HOME", "/nonexistent")) / marker).exists():
    response = response.get("then", response)
if response.get("write_marker"):
    home = Path(os.environ["CODEX_HOME"])
    (home / response["write_marker"]).write_text("signed in")
if response.get("spawn"):  # a helper that holds the output pipes
    child = subprocess.Popen(["sleep", "60"])
    record["child_pid"] = child.pid
if response.get("spawn_detached"):  # a helper that lets go of them, and lingers
    child = subprocess.Popen(
        ["sleep", "60"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    record["child_pid"] = child.pid
with open(here / "calls.jsonl", "a") as log:
    log.write(json.dumps(record) + "\n")
if response.get("touch"):
    (cwd / response["touch"]).write_text("x")
time.sleep(response.get("sleep", 0))
sys.stdout.write(response.get("stdout", ""))
sys.stderr.write(response.get("stderr", ""))
sys.exit(response.get("exit", 0))
"""


BASE_BEHAVIOUR: dict[str, dict[str, Any]] = {
    "codex": {"--version": {"stdout": "codex-cli 0.158.0\n"}},
}


class FakeCli:
    def __init__(self, directory: Path, name: str) -> None:
        self.directory = directory
        self.name = name
        directory.mkdir(parents=True, exist_ok=True)
        # A shell stub that execs this interpreter: a `#!` line can't hold a path with spaces.
        (directory / f"{name}.py").write_text(SCRIPT.format())
        path = directory / name
        path.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$0.py" "$@"\n')
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        # Responses every behaviour starts from (a test's own keys win): a fake `codex`
        # is a tested version, since the adapter checks it on every turn.
        self.base: dict[str, Any] = dict(BASE_BEHAVIOUR.get(name, {}))
        self.behave({})

    def behave(self, behaviour: dict[str, Any]) -> None:
        merged = {**self.base, **behaviour}
        (self.directory / "behaviour.json").write_text(json.dumps(merged))

    def answer(self, stdout: str = "", **response: Any) -> None:
        """One response for every call."""
        self.behave({"default": {"stdout": stdout, **response}})

    def calls(self) -> list[dict[str, Any]]:
        path = self.directory / "calls.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]


def install(monkeypatch, tmp_path: Path, name: str) -> FakeCli:
    """A fake `name` alone on PATH."""
    fake = FakeCli(tmp_path / "bin", name)
    monkeypatch.setenv("PATH", str(fake.directory) + os.pathsep + "/usr/bin" + os.pathsep + "/bin")
    return fake


def claude_result(answer: dict[str, Any] | None = None, **overrides: Any) -> str:
    """A `claude -p --output-format json` result, shaped like Claude Code 2.1.299's."""
    result: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "duration_ms": 3224,
        "duration_api_ms": 3208,
        "num_turns": 2,
        "stop_reason": "tool_use",
        "total_cost_usd": 0.00055,
        "usage": {"server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0}},
        "modelUsage": {"claude-haiku-5-5": {"inputTokens": 2, "outputTokens": 523}},
        "permission_denials": [],
        "subagent_stats": {"spawned": 0},
        "result": json.dumps(answer) if answer is not None else "",
    }
    if answer is not None:
        result["structured_output"] = answer
    result.update(overrides)
    return json.dumps(result)


def codex_events(answer: dict[str, Any] | str | None, *extra: dict[str, Any]) -> str:
    """A `codex exec --json` stream, shaped like Codex 0.158.0's, ending in `answer`."""
    events: list[dict[str, Any]] = [
        {"type": "thread.started", "thread_id": "01a139c9-b725-78f3-bc0f-38bd29fd93a7"},
        {
            "type": "item.completed",
            "item": {
                "id": "item_0",
                "type": "error",
                "message": "Code Mode is unavailable because code-mode host is disabled.",
            },
        },
        {"type": "turn.started"},
        *extra,
    ]
    if answer is not None:
        text = answer if isinstance(answer, str) else json.dumps(answer)
        events.append(
            {
                "type": "item.completed",
                "item": {"id": "item_2", "type": "agent_message", "text": text},
            }
        )
        events.append(
            {"type": "turn.completed", "usage": {"input_tokens": 14000, "output_tokens": 80}}
        )
    return "\n".join(json.dumps(e) for e in events) + "\n"
