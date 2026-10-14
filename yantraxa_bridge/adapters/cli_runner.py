"""One-shot runs of a vendor CLI (`claude`, `codex`) for a turn, and what the two adapters share.

Each run gets a fresh `yantraxa-turn-*` directory (0700) as its working directory, with the
files it needs written 0600, a stripped environment, the prompt on stdin and a wall-clock
limit: SIGTERM to the whole process group, then SIGKILL two seconds later. On Windows the
CLI runs in a Job Object, terminated as one. When the CLI exits by itself, whatever it left
running in its group is killed too. The directory is deleted afterwards. Prompts go through
stdin and files, never argv, so meeting text and approved facts never show in `ps`.
Anything the CLI leaves in the directory is reported, and the adapters treat it as a tool
call (spec §4.1, §4.6). A run may also get its own empty `HOME` (Codex does), so nothing
from the owner's real home reaches the model.

`kill_active()` stops every running CLI and removes its directories, for a signal handler;
`sweep_stale_turn_dirs()` removes the ones a killed bridge left behind.

`CliAdapter` is the turn loop both adapters share: caps, the addressed-turn model choice
(§4.5), the extra prompt lines, the transparency log and pass-on-failure.
"""

from __future__ import annotations

import asyncio
import contextlib
import getpass
import logging
import os
import re
import shutil
import signal
import stat
import tempfile
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..caps import Caps
from ..prompt import Refs, build_messages, parse_action
from ..protocol import TurnAction
from ..turnlog import TurnLog, printable
from .base import AdapterInfo, ContextMessage, Final, TurnContext, TurnEvent, pass_turn

log = logging.getLogger("yantraxa_bridge.adapters.cli")

# D10: the CLIs get at most this long, inside the bridge's own `timeout_s`.
CLI_WALL_CLOCK_S = 35.0
KILL_GRACE_S = 2.0
MAX_OUTPUT_BYTES = 4_000_000
TURN_DIR_PREFIX = "yantraxa-turn-"
HOME_DIR_PREFIX = TURN_DIR_PREFIX + "home-"
# A turn lasts at most CLI_WALL_CLOCK_S, so a turn directory this old was left by a bridge
# that was killed; another bridge's live turn is never this old.
STALE_TURN_DIR_S = 600.0
CAPABILITIES = ("message", "question", "reply", "proposal", "vote")

# §4.4: the one line both adapters add to the system prompt.
NO_TOOLS_RULE = (
    "6. You are answering from a terminal program that has no tools. You cannot look "
    "anything up, run anything, open files or browse. Never claim to have done so, and never "
    "claim to remember anything from outside this meeting."
)
# Words a vendor CLI uses when the owner's plan allowance or the API rate limit is used up.
RATE_LIMIT = re.compile(
    r"usage limit|rate[ _-]?limit|limit (?:is )?reached|too many requests|\b429\b|quota",
    re.IGNORECASE,
)


class CliNotFound(Exception):
    """The CLI isn't on the PATH."""


@dataclass(frozen=True)
class CliResult:
    returncode: int | None  # None when the run was killed
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    leftovers: tuple[str, ...] = ()  # what the CLI left in its working directory


def base_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The only environment a CLI turn gets: `HOME`, `PATH`, `USER`, `TERM=dumb`, `TZ=UTC`,
    plus `extra`.

    `USER` is required for Claude Code's keychain login on macOS (spec §4.2). `TZ=UTC`
    keeps the owner's timezone out of what Codex tells the model, verified on macOS only: on
    Linux Codex reads `/etc/localtime` instead, so the zone probably still shows there."""
    env = {
        "HOME": str(Path.home()),
        "PATH": os.environ.get("PATH", os.defpath),
        "USER": os.environ.get("USER") or os.environ.get("LOGNAME") or getpass.getuser(),
        "TERM": "dumb",
        "TZ": "UTC",
    }
    if os.name == "nt":  # untested: what a Windows process needs to start at all
        for key in ("SYSTEMROOT", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PATHEXT", "TEMP"):
            if key in os.environ:
                env[key] = os.environ[key]
    env.update(extra or {})
    return env


def _write_private(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def private_dir(prefix: str = TURN_DIR_PREFIX) -> Path:
    """A fresh 0700 directory in the system temp folder."""
    path = Path(tempfile.mkdtemp(prefix=prefix))
    os.chmod(path, 0o700)
    return path


def with_home(env: Mapping[str, str], home: Path) -> dict[str, str]:
    """`env` with `home` as the home directory (`USERPROFILE` too on Windows)."""
    env = {**env, "HOME": str(home)}
    if os.name == "nt":
        env["USERPROFILE"] = str(home)
    return env


# ------------------------------------------------------------------ Windows Job Objects


class WindowsJob:
    """Windows: a Job Object that kills every process in it when terminated or closed, the
    closest thing to a POSIX process group. `kernel32` is injectable for tests."""

    EXTENDED_LIMIT_INFORMATION = 9  # JobObjectExtendedLimitInformation
    KILL_ON_JOB_CLOSE = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    PROCESS_SET_QUOTA_AND_TERMINATE = 0x0100 | 0x0001

    def __init__(self, kernel32: Any = None) -> None:
        import ctypes

        if kernel32 is None:  # pragma: no cover - Windows only
            kernel32 = _kernel32()
        self._k = kernel32
        self._handle = self._k.CreateJobObjectW(None, None)
        if not self._handle:
            raise OSError("CreateJobObjectW failed")
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = self.KILL_ON_JOB_CLOSE
        if not self._k.SetInformationJobObject(
            self._handle,
            self.EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            self.close()
            raise OSError("SetInformationJobObject failed")

    def add(self, pid: int) -> None:
        """Puts the process in the job. A child it started before this escapes, which is
        why this runs straight after the spawn."""
        process = self._k.OpenProcess(self.PROCESS_SET_QUOTA_AND_TERMINATE, False, pid)
        if not process:
            raise OSError(f"OpenProcess({pid}) failed")
        try:
            if not self._k.AssignProcessToJobObject(self._handle, process):
                raise OSError("AssignProcessToJobObject failed")
        finally:
            self._k.CloseHandle(process)

    def kill(self) -> None:
        if self._handle:
            self._k.TerminateJobObject(self._handle, 1)

    def close(self) -> None:
        """Closing the last handle kills whatever is still in the job."""
        if self._handle:
            self._k.CloseHandle(self._handle)
            self._handle = None


def _limit_structures() -> type:
    import ctypes

    ulonglong, size_t, dword = ctypes.c_ulonglong, ctypes.c_size_t, ctypes.c_uint32

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]  # fmt: skip

    class BasicLimits(ctypes.Structure):  # JOBOBJECT_BASIC_LIMIT_INFORMATION
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", dword),
            ("MinimumWorkingSetSize", size_t),
            ("MaximumWorkingSetSize", size_t),
            ("ActiveProcessLimit", dword),
            ("Affinity", size_t),
            ("PriorityClass", dword),
            ("SchedulingClass", dword),
        ]

    class ExtendedLimits(ctypes.Structure):  # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        _fields_ = [
            ("BasicLimitInformation", BasicLimits),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", size_t),
            ("JobMemoryLimit", size_t),
            ("PeakProcessMemoryUsed", size_t),
            ("PeakJobMemoryUsed", size_t),
        ]

    return ExtendedLimits


_ExtendedLimits = _limit_structures()


def _kernel32() -> Any:  # pragma: no cover - Windows only
    """kernel32 with handle-sized return and argument types, so a 64-bit handle isn't cut
    to an int."""
    import ctypes
    from ctypes import wintypes

    k = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    handle = wintypes.HANDLE
    k.CreateJobObjectW.restype = handle
    k.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    k.SetInformationJobObject.restype = wintypes.BOOL
    k.SetInformationJobObject.argtypes = (handle, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    k.OpenProcess.restype = handle
    k.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k.AssignProcessToJobObject.restype = wintypes.BOOL
    k.AssignProcessToJobObject.argtypes = (handle, handle)
    k.TerminateJobObject.restype = wintypes.BOOL
    k.TerminateJobObject.argtypes = (handle, wintypes.UINT)
    k.CloseHandle.restype = wintypes.BOOL
    k.CloseHandle.argtypes = (handle,)
    return k


def _new_job() -> WindowsJob | None:
    """A job for one run on Windows; None elsewhere, or when Windows refuses one (the run
    then falls back to killing the CLI alone)."""
    if os.name != "nt":
        return None
    try:
        return WindowsJob()
    except OSError as error:  # pragma: no cover - Windows only
        log.warning("couldn't create a Job Object, so only the CLI itself can be killed: %s", error)
        return None


# ------------------------------------------------------------------ what is running now


@dataclass(eq=False)
class _Running:
    process: asyncio.subprocess.Process | None = None
    job: WindowsJob | None = None
    dirs: list[Path] = field(default_factory=list)


_ACTIVE: set[_Running] = set()


def _kill_group_now(process: asyncio.subprocess.Process, job: WindowsJob | None) -> None:
    """SIGKILL (or job termination) for the CLI and everything in its group, whether or
    not the CLI itself has exited."""
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        if job is not None:
            job.kill()
        elif os.name != "nt":
            os.killpg(process.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        elif process.returncode is None:
            process.kill()


def kill_active() -> int:
    """Kills every CLI this process is running, with its group, and deletes their turn
    directories. Synchronous, for a signal handler. Returns how many runs it stopped."""
    stopped = 0
    for running in list(_ACTIVE):
        if running.process is not None:
            _kill_group_now(running.process, running.job)
            stopped += 1
        for path in running.dirs:
            shutil.rmtree(path, ignore_errors=True)
    return stopped


def sweep_stale_turn_dirs(
    base: Path | None = None, *, older_than_s: float = STALE_TURN_DIR_S, now: float | None = None
) -> list[Path]:
    """Deletes `yantraxa-turn-*` directories in the temp folder that this user owns and
    that are older than `older_than_s`: what a bridge killed mid-turn left behind."""
    base = Path(base or tempfile.gettempdir())
    now = time.time() if now is None else now
    removed: list[Path] = []
    try:
        entries = list(base.iterdir())
    except OSError:
        return removed
    uid = os.getuid() if hasattr(os, "getuid") else None
    for path in entries:
        if not path.name.startswith(TURN_DIR_PREFIX):
            continue
        try:
            info = path.lstat()
        except OSError:
            continue
        if not stat.S_ISDIR(info.st_mode):  # a symlink or a file is never ours
            continue
        if uid is not None and info.st_uid != uid:
            continue
        if now - info.st_mtime < older_than_s:
            continue
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            removed.append(path)
    return removed


class CliRunner:
    """Runs one CLI process per call; see the module docstring."""

    def __init__(
        self,
        *,
        timeout_s: float = CLI_WALL_CLOCK_S,
        kill_grace_s: float = KILL_GRACE_S,
        max_output_bytes: int = MAX_OUTPUT_BYTES,
    ) -> None:
        self.timeout_s = timeout_s
        self.kill_grace_s = kill_grace_s
        self.max_output_bytes = max_output_bytes

    def resolve(self, command: str, env: Mapping[str, str]) -> str:
        found = shutil.which(command, path=env.get("PATH"))
        if found is None:
            raise CliNotFound(f"`{command}` is not on the PATH")
        return found

    async def run(
        self,
        command: str,
        args: Callable[[Path], list[str]],
        *,
        env: Mapping[str, str],
        stdin: str = "",
        files: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
        private_home: bool = False,
    ) -> CliResult:
        """Runs `command *args(turn_dir)` in a fresh turn directory holding `files`, for at
        most `timeout_s` (default: the runner's). With `private_home`, `HOME` is a fresh
        empty 0700 directory, deleted afterwards."""
        executable = self.resolve(command, env)
        running = _Running()
        _ACTIVE.add(running)
        turn_dir = private_dir()
        running.dirs.append(turn_dir)
        try:
            run_env = dict(env)
            if private_home:
                home = private_dir(HOME_DIR_PREFIX)
                running.dirs.append(home)
                run_env = with_home(run_env, home)
            for name, text in (files or {}).items():
                _write_private(turn_dir / name, text)
            started = time.monotonic()
            returncode, stdout, stderr, timed_out = await self._exec(
                [executable, *args(turn_dir)],
                turn_dir,
                run_env,
                stdin,
                self.timeout_s if timeout_s is None else timeout_s,
                running,
            )
            duration = time.monotonic() - started
            ours = set(files or {})
            try:
                leftovers = tuple(sorted(p.name for p in turn_dir.iterdir() if p.name not in ours))
            except FileNotFoundError:  # `kill_active()` already removed it
                leftovers = ()
            return CliResult(
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                duration_s=duration,
                timed_out=timed_out,
                leftovers=leftovers,
            )
        finally:
            _ACTIVE.discard(running)
            for path in running.dirs:
                shutil.rmtree(path, ignore_errors=True)

    async def _exec(
        self,
        argv: list[str],
        cwd: Path,
        env: dict[str, str],
        stdin: str,
        timeout_s: float,
        running: _Running,
    ) -> tuple[int | None, str, str, bool]:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name != "nt",  # its own process group, killed as one
        )
        running.process = process
        job: WindowsJob | None = None
        try:
            running.job = job = self._job_for(process)
            out, err = await asyncio.wait_for(process.communicate(stdin.encode("utf-8")), timeout_s)
        except TimeoutError:
            await self._kill(process, job)
            return None, "", "", True
        except BaseException:  # the turn was cancelled: never leave the CLI running
            await self._kill(process, job)
            raise
        finally:
            # Whatever the CLI left running in its group dies with the turn.
            _kill_group_now(process, job)
            if job is not None:
                job.close()
        return (
            process.returncode,
            out[: self.max_output_bytes].decode("utf-8", "replace"),
            err[: self.max_output_bytes].decode("utf-8", "replace"),
            False,
        )

    @staticmethod
    def _job_for(process: asyncio.subprocess.Process) -> WindowsJob | None:
        """The CLI's Job Object on Windows, or None. A job the CLI couldn't be put in is
        closed and dropped: an empty job would make every kill a no-op, so the kills fall
        back to the process instead."""
        job = _new_job()
        if job is None:
            return None
        try:
            job.add(process.pid)
        except BaseException as error:
            job.close()
            if not isinstance(error, OSError):
                raise
            log.warning("couldn't put the CLI in its Job Object, killing it alone: %s", error)
            return None
        return job

    async def _kill(
        self, process: asyncio.subprocess.Process, job: WindowsJob | None = None
    ) -> None:
        """SIGTERM to the whole group, then SIGKILL after the grace period, then reap.

        The group is signalled even when the CLI itself has already exited: a helper it
        started may still hold the output pipes, and would otherwise outlive the turn."""
        self._signal(process, signal.SIGTERM, job)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.shield(process.wait()), self.kill_grace_s)
        self._signal(process, getattr(signal, "SIGKILL", signal.SIGTERM), job)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.shield(process.wait()), self.kill_grace_s)

    @staticmethod
    def _signal(
        process: asyncio.subprocess.Process, sig: int, job: WindowsJob | None = None
    ) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            if job is not None:
                job.kill()
            elif os.name != "nt":
                os.killpg(process.pid, sig)
            elif process.returncode is None:
                process.terminate() if sig == signal.SIGTERM else process.kill()


# ------------------------------------------------------------------ addressed turns (§4.5)


def _mentions(body: str, handle: str) -> bool:
    pattern = rf"(?<![\w.@])@{re.escape(handle)}(?![\w-]|\.\w)"
    return re.search(pattern, body, re.IGNORECASE) is not None


def addressed_message(ctx: TurnContext) -> ContextMessage | None:
    """The newest message, when someone else wrote it to this seat: an @mention of its
    handle, or a reply to one of its own messages. None on a rotation turn."""
    spoken = [m for m in ctx.recent_messages if m.kind != "system"]
    if not spoken or spoken[-1].seat_id == ctx.me.seat_id:
        return None
    newest = spoken[-1]
    mine = {m.id for m in ctx.recent_messages if m.seat_id == ctx.me.seat_id}
    if newest.reply_to is not None and newest.reply_to in mine:
        return newest
    if ctx.me.handle and _mentions(newest.body, ctx.me.handle):
        return newest
    return None


def cli_prompt(ctx: TurnContext) -> tuple[str, str, Refs, ContextMessage | None]:
    """System and user text for a CLI turn: `prompt.build_messages` plus the no-tools rule
    and, on an addressed turn, a line saying which message to answer.

    That line names the message only by its ref: the author's name is participant-written,
    so it stays inside the fence."""
    messages, refs = build_messages(ctx)
    system, user = messages[0]["content"], messages[1]["content"]
    lines = [system, "", NO_TOOLS_RULE]
    addressed = addressed_message(ctx)
    if addressed is not None and (ref := refs.ref_for(addressed.id)):
        lines.append(
            f"7. The newest message, [{ref}], spoke to you directly. Answer it first, briefly, "
            f'as action "reply" with reply_to "{ref}".'
        )
    return "\n".join(lines), user, refs, addressed


# ------------------------------------------------------------------ the shared turn loop


@dataclass
class Attempt:
    """What one CLI run produced, for the turn loop and the transparency log."""

    answer: Any = None  # the answer object, when there is one
    reason: str | None = None  # why the turn passes, when it does
    rate_limited: bool = False
    raw: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class CliAdapter:
    """The turn loop of the `claude_code` and `codex_cli` adapters.

    Subclasses set `adapter_name` and `runtime_label` and implement `_model` and `_attempt`.
    """

    adapter_name = ""
    runtime_label = ""

    def __init__(
        self,
        runner: CliRunner,
        *,
        caps: Caps | None = None,
        turnlog: TurnLog | None = None,
        command: str,
    ) -> None:
        self.runner = runner
        self.caps = caps
        self.turnlog = turnlog
        self.command = command

    def describe(self) -> AdapterInfo:
        return AdapterInfo(self.runtime_label, CAPABILITIES, streams=False)

    async def aclose(self) -> None:
        return None

    def _model(self, addressed: bool) -> str | None:
        raise NotImplementedError

    async def _attempt(
        self, ctx: TurnContext, system: str, user: str, model: str | None
    ) -> Attempt:
        raise NotImplementedError

    async def take_turn(self, ctx: TurnContext) -> AsyncIterator[TurnEvent]:
        if self.caps is not None and (blocked := self.caps.check(ctx.meeting.id)):
            self._log(ctx, model=None, attempt=Attempt(reason=blocked), actions=None)
            yield pass_turn()
            return
        system, user, refs, addressed = cli_prompt(ctx)
        model = self._model(addressed is not None)
        if self.caps is not None:
            self.caps.record_turn(ctx.meeting.id)
        try:
            attempt = await self._attempt(ctx, system, user, model)
        except CliNotFound as error:
            attempt = Attempt(reason=str(error))
        except OSError as error:
            attempt = Attempt(reason=f"couldn't start `{self.command}`: {error}")
        actions = self._actions(attempt, ctx, refs)
        if self.caps is not None:
            if attempt.rate_limited:
                self.caps.vendor_limited()
            elif actions is not None:
                self.caps.vendor_ok()
        attempt.details["addressed"] = addressed.id if addressed is not None else None
        self._log(ctx, model=model, attempt=attempt, actions=actions, system=system, user=user)
        if actions is None:
            log.warning("%s turn %s passes: %s", self.adapter_name, ctx.turn_id, attempt.reason)
            yield pass_turn()
            return
        yield Final(actions)

    def _actions(self, attempt: Attempt, ctx: TurnContext, refs: Refs) -> list[TurnAction] | None:
        if attempt.reason is not None:
            return None
        action = parse_action(attempt.answer, ctx, refs)
        if action is None:
            attempt.reason = "the answer doesn't fit the output schema"
            return None
        return [action]

    def _log(
        self,
        ctx: TurnContext,
        *,
        model: str | None,
        attempt: Attempt,
        actions: list[TurnAction] | None,
        system: str | None = None,
        user: str | None = None,
    ) -> None:
        if self.turnlog is None:
            return
        self.turnlog.record(
            "model",
            meeting_id=ctx.meeting.id,
            meeting_code=ctx.meeting.code,
            turn_id=ctx.turn_id,
            adapter=self.adapter_name,
            model=model,
            system=system,
            prompt=user,
            raw_output=printable(attempt.raw) or None,
            actions=[a.model_dump(mode="json", exclude_none=True) for a in actions]
            if actions
            else None,
            pass_reason=attempt.reason,
            **attempt.details,
        )


def short(text: str, limit: int = 300) -> str:
    """One line of CLI output for a log reason, without control characters."""
    text = " ".join(printable(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ------------------------------------------------------------------ preflight (§3.2)


class PreflightError(Exception):
    """The CLI can't take turns yet; the message says what to do, in one line."""


PROBE_SEAT = "00000000-0000-4000-8000-000000000001"
PROBE_OTHER_SEAT = "00000000-0000-4000-8000-000000000002"
PROBE_MESSAGE = "00000000-0000-4000-8000-000000000003"


def probe_context() -> TurnContext:
    """A synthetic one-message meeting that @mentions the agent, for a probe turn."""
    from .base import Me, MeetingInfo

    return TurnContext(
        turn_id="00000000-0000-4000-8000-000000000004",
        me=Me(handle="bridge.probe", display_name="Bridge probe", seat_id=PROBE_SEAT),
        meeting=MeetingInfo(
            id="00000000-0000-4000-8000-000000000005",
            code="PROBE-1",
            title="Bridge check",
            purpose="Check that the agent can answer",
        ),
        recent_messages=[
            ContextMessage(
                id=PROBE_MESSAGE,
                seat_id=PROBE_OTHER_SEAT,
                author="Sam",
                kind="question",
                body="@bridge.probe can you hear us? One short sentence, please.",
                created_at="2026-10-14T09:00:00.000Z",
            )
        ],
        deadline_at="2099-01-01T00:00:00.000Z",
        max_output_chars=200,
    )


async def probe(adapter: CliAdapter) -> tuple[TurnAction, float]:
    """One confined turn against `probe_context()`; raises PreflightError unless it answers."""
    ctx = probe_context()
    system, user, refs, addressed = cli_prompt(ctx)
    started = time.monotonic()
    try:
        attempt = await adapter._attempt(ctx, system, user, adapter._model(addressed is not None))
    except CliNotFound as error:
        raise PreflightError(str(error)) from None
    actions = adapter._actions(attempt, ctx, refs)
    if actions is None:
        raise PreflightError(f"the probe turn failed: {attempt.reason}")
    return actions[0], time.monotonic() - started


async def run_quick(
    runner: CliRunner,
    command: str,
    args: list[str],
    env: Mapping[str, str],
    *,
    private_home: bool = False,
) -> CliResult:
    """A short non-turn command (`--version`, a sign-in check)."""
    try:
        return await runner.run(command, lambda _dir: args, env=env, private_home=private_home)
    except CliNotFound:
        raise PreflightError(
            f"`{command}` is not installed or not on the PATH; install it, sign in, and try again"
        ) from None
