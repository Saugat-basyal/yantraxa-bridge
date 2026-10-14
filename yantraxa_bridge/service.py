"""`yantraxa-bridge service install | uninstall | status`: the bridge in the background (§2.5).

One per user, never root, no admin:

- macOS: a launchd agent, `~/Library/LaunchAgents/com.yantraxa.bridge.plist`.
- Linux: a `systemd --user` unit, `~/.config/systemd/user/yantraxa-bridge.service`.
- Windows: a value under `HKCU\\...\\CurrentVersion\\Run` that starts a hidden supervisor
  (`pythonw`) at logon; the supervisor runs the bridge and restarts it if it dies.

Each runs `yantraxa-bridge run --service` with this `YANTRAXA_HOME`. `--service` sends the
log to `$YANTRAXA_HOME/log/bridge.log` and exits 0 when restarting can't help (disconnected,
revoked, replaced by another bridge, not paired, a config error), so only a crash is
restarted, after 10 s. The unit files are 0644 and hold no secret: the credential stays in
`$YANTRAXA_HOME/credentials` (0600), and an API key comes from the environment at run time.
"""

from __future__ import annotations

import getpass
import os
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

from .config import ensure_private_dir, yantraxa_home

LABEL = "com.yantraxa.bridge"
SYSTEMD_UNIT = "yantraxa-bridge.service"
WINDOWS_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
WINDOWS_RUN_VALUE = "YantraxaBridge"
LOG_DIR = "log"
BRIDGE_LOG = "bridge.log"  # the bridge's own log, rotated, written by `run --service`
SERVICE_OUT = "service.out"  # stdout and stderr: anything outside the log, e.g. a traceback
PID_FILE = "service.pid"  # Windows: the supervisor's pid
RESTART_DELAY_S = 10
MAX_SERVICE_OUT_BYTES = 5 * 1024 * 1024


class ServiceError(Exception):
    """The background service could not be installed, removed or inspected."""


@dataclass
class Result:
    returncode: int
    stdout: str = ""
    stderr: str = ""


Commands = Callable[[Sequence[str]], Result]


def run_command(args: Sequence[str]) -> Result:
    """Runs `launchctl`, `systemctl`, `loginctl` or `taskkill`; the one place that does."""
    try:
        done = subprocess.run(list(args), capture_output=True, text=True, timeout=30, check=False)
    except FileNotFoundError:
        return Result(127, "", f"{args[0]}: not found")
    except subprocess.TimeoutExpired:
        return Result(124, "", f"{args[0]}: timed out")
    return Result(done.returncode, done.stdout, done.stderr)


@dataclass
class Launch:
    """What the service runs, and with which environment."""

    program: list[str]  # argv of `yantraxa-bridge run --service`
    home: Path
    path_env: str
    log_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.log_dir = self.home / LOG_DIR

    @property
    def out_path(self) -> Path:
        return self.log_dir / SERVICE_OUT


@dataclass
class Status:
    installed: bool
    running: bool
    kind: str  # "launchd agent com.yantraxa.bridge", …
    where: str  # the file or registry value
    pid: int | None = None
    detail: str | None = None  # e.g. "last exit code 0"


def bridge_command(argv0: str | None = None) -> list[str]:
    """How to start this bridge again: the `yantraxa-bridge` script that is running now, as
    typed (a `uv tool` symlink keeps working across upgrades), else this Python with `-m`."""
    argv0 = argv0 if argv0 is not None else sys.argv[0]
    path = Path(argv0)
    if path.name.startswith("yantraxa-bridge") and path.is_file():
        return [os.path.abspath(path)]
    return [sys.executable, "-m", "yantraxa_bridge.cli"]


def service_path(env_path: str | None = None, extra: Sequence[str] = ()) -> str:
    """The PATH the service gets: this shell's (absolute entries only, so `claude` and `codex`
    are found as they are here), then `extra`, then the system defaults."""
    current = os.environ.get("PATH", "") if env_path is None else env_path
    defaults = (
        [] if os.name == "nt" else ["/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    )
    seen: list[str] = []
    for entry in [*current.split(os.pathsep), *extra, *defaults]:
        if entry and os.path.isabs(entry) and entry not in seen:
            seen.append(entry)
    return os.pathsep.join(seen)


def make_launch(home: Path | None = None, argv0: str | None = None) -> Launch:
    home = (home or yantraxa_home()).expanduser().absolute()
    program = [*bridge_command(argv0), "run", "--service"]
    extra = [str(Path(program[0]).parent), str(Path.home() / ".local" / "bin")]
    return Launch(program=program, home=home, path_env=service_path(extra=extra))


def prepare_logs(launch: Launch) -> None:
    """The 0700 log folder and a 0600 `service.out`, so launchd or systemd append to a
    private file instead of creating a world-readable one."""
    ensure_private_dir(launch.log_dir)
    fd = os.open(launch.out_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.close(fd)
    os.chmod(launch.out_path, 0o600)


def trim_service_out(home: Path) -> None:
    """Empties `service.out` once it passes 5 MB; called by `run --service` on start."""
    path = home / LOG_DIR / SERVICE_OUT
    try:
        if path.stat().st_size > MAX_SERVICE_OUT_BYTES:
            os.truncate(path, 0)
    except OSError:
        pass


def _write_public(path: Path, data: bytes) -> None:
    """A 0644 file, atomically: unit files hold no secret, and launchd wants them readable.
    Written through an unpredictably named new file (`mkstemp`), then `os.replace`."""
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class Backend(Protocol):
    kind: str

    def install(self, launch: Launch) -> list[str]: ...

    def uninstall(self) -> list[str]: ...

    def status(self) -> Status: ...


# ------------------------------------------------------------------ macOS: launchd


def render_plist(launch: Launch) -> bytes:
    return plistlib.dumps(
        {
            "Label": LABEL,
            "ProgramArguments": launch.program,
            "EnvironmentVariables": {
                "YANTRAXA_HOME": str(launch.home),
                "PATH": launch.path_env,
            },
            "RunAtLoad": True,
            # Restart after a crash (a non-zero exit), not after `run --service` exits 0.
            "KeepAlive": {"SuccessfulExit": False},
            "ThrottleInterval": RESTART_DELAY_S,
            "WorkingDirectory": str(launch.home),
            "StandardOutPath": str(launch.out_path),
            "StandardErrorPath": str(launch.out_path),
        }
    )


class Launchd:
    kind = f"launchd agent {LABEL}"

    def __init__(
        self,
        *,
        user_home: Path | None = None,
        uid: int | None = None,
        commands: Commands = run_command,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.plist = (user_home or Path.home()) / "Library" / "LaunchAgents" / f"{LABEL}.plist"
        self.domain = f"gui/{os.getuid() if uid is None else uid}"
        self.commands = commands
        self.sleep = sleep

    @property
    def target(self) -> str:
        return f"{self.domain}/{LABEL}"

    def _print(self) -> Result:
        return self.commands(["launchctl", "print", self.target])

    def install(self, launch: Launch) -> list[str]:
        prepare_logs(launch)
        if self._print().returncode == 0:
            self.commands(["launchctl", "bootout", self.target])
        self.plist.parent.mkdir(parents=True, exist_ok=True)
        _write_public(self.plist, render_plist(launch))
        self.commands(["launchctl", "enable", self.target])
        # Right after a bootout, bootstrap can fail with "5: Input/output error" for a moment.
        for attempt in range(5):
            done = self.commands(["launchctl", "bootstrap", self.domain, str(self.plist)])
            if done.returncode == 0:
                return [f"Installed {self.plist}"]
            self.sleep(0.5 * (attempt + 1))
        raise ServiceError(
            f"launchctl bootstrap failed: {(done.stderr or done.stdout).strip() or done.returncode}"
        )

    def uninstall(self) -> list[str]:
        lines = []
        if self._print().returncode == 0:
            self.commands(["launchctl", "bootout", self.target])
            lines.append(f"Stopped {LABEL}")
        if self.plist.exists():
            self.plist.unlink()
            lines.append(f"Removed {self.plist}")
        return lines

    def status(self) -> Status:
        status = Status(self.plist.exists(), False, self.kind, str(self.plist))
        shown = self._print()
        if shown.returncode != 0:
            if status.installed:
                status.detail = "not loaded (log out and in, or run `service install` again)"
            return status
        fields = dict(re.findall(r"^\t(state|pid|last exit code) = (.*)$", shown.stdout, re.M))
        status.running = fields.get("state") == "running"
        if fields.get("pid", "").isdigit():
            status.pid = int(fields["pid"])
        if "last exit code" in fields:
            status.detail = f"last exit code {fields['last exit code']}"
        return status


# ------------------------------------------------------------------ Linux: systemd --user


def _systemd_quote(value: str, *, exec_arg: bool = False) -> str:
    """One word for a unit file: `%` doubled, and `$` too in ExecStart; quoted if needed."""
    value = value.replace("%", "%%")
    if exec_arg:
        value = value.replace("$", "$$")
    if value and not re.search(r'[\s"\'\\;]', value):
        return value
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_unit(launch: Launch) -> str:
    exec_start = " ".join(_systemd_quote(arg, exec_arg=True) for arg in launch.program)
    out = str(launch.out_path).replace("%", "%%")
    return "\n".join(
        [
            "# Written by `yantraxa-bridge service install`; remove it with",
            "# `yantraxa-bridge service uninstall`. It holds no secret.",
            "[Unit]",
            "Description=Yantraxa bridge",
            "After=network-online.target",
            "",
            "[Service]",
            "Type=simple",
            f"ExecStart={exec_start}",
            "Environment=" + _systemd_quote(f"YANTRAXA_HOME={launch.home}"),
            "Environment=" + _systemd_quote(f"PATH={launch.path_env}"),
            # Restart after a crash, not after `run --service` exits 0.
            "Restart=on-failure",
            f"RestartSec={RESTART_DELAY_S}",
            f"StandardOutput=append:{out}",
            f"StandardError=append:{out}",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )


class Systemd:
    kind = f"systemd user unit {SYSTEMD_UNIT}"

    def __init__(
        self,
        *,
        config_home: Path | None = None,
        user: str | None = None,
        commands: Commands = run_command,
    ) -> None:
        base = (
            config_home
            or Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config").expanduser()
        )
        self.unit = base / "systemd" / "user" / SYSTEMD_UNIT
        self.dropins = self.unit.with_name(SYSTEMD_UNIT + ".d")
        self.user = user or getpass.getuser()
        self.commands = commands

    def _systemctl(self, *args: str, check: bool = True) -> Result:
        done = self.commands(["systemctl", "--user", *args])
        if check and done.returncode != 0:
            message = (done.stderr or done.stdout).strip() or f"exit {done.returncode}"
            raise ServiceError(
                f"`systemctl --user {' '.join(args)}` failed: {message}. Without a systemd "
                "user session, keep `yantraxa-bridge run` running another way (tmux, a "
                "terminal left open)."
            )
        return done

    def install(self, launch: Launch) -> list[str]:
        if self.commands(["systemctl", "--version"]).returncode != 0:
            raise ServiceError(
                "this system has no systemd; keep `yantraxa-bridge run` running another way "
                "(tmux, a terminal left open)"
            )
        prepare_logs(launch)
        self.unit.parent.mkdir(parents=True, exist_ok=True)
        _write_public(self.unit, render_unit(launch).encode())
        self._systemctl("daemon-reload")
        self._systemctl("enable", SYSTEMD_UNIT)
        self._systemctl("restart", SYSTEMD_UNIT)
        lines = [f"Installed {self.unit}"]
        linger = self.commands(["loginctl", "show-user", self.user, "--property=Linger"])
        if linger.returncode == 0 and linger.stdout.strip() != "Linger=yes":
            lines.append(
                "It runs while you're logged in. To keep it running after you log out, "
                f"run once: sudo loginctl enable-linger {self.user}"
            )
        return lines

    def uninstall(self) -> list[str]:
        lines = []
        if self.unit.exists():
            self._systemctl("disable", "--now", SYSTEMD_UNIT, check=False)
            self.unit.unlink()
            lines.append(f"Removed {self.unit}")
        if self.dropins.is_dir():
            shutil.rmtree(self.dropins)
            lines.append(f"Removed {self.dropins} (your overrides for this unit)")
        if lines:
            self._systemctl("daemon-reload", check=False)
            self._systemctl("reset-failed", SYSTEMD_UNIT, check=False)
        return lines

    def status(self) -> Status:
        status = Status(self.unit.exists(), False, self.kind, str(self.unit))
        shown = self._systemctl(
            "show",
            SYSTEMD_UNIT,
            "--property=ActiveState,SubState,MainPID,ExecMainStatus",
            check=False,
        )
        if shown.returncode != 0:
            return status
        fields = dict(line.split("=", 1) for line in shown.stdout.splitlines() if "=" in line)
        status.running = fields.get("ActiveState") == "active"
        if fields.get("MainPID", "0").isdigit() and int(fields.get("MainPID", "0")):
            status.pid = int(fields["MainPID"])
        if status.installed and fields.get("ActiveState"):
            status.detail = f"{fields['ActiveState']} ({fields.get('SubState', '?')})"
            if not status.running and fields.get("ExecMainStatus"):
                status.detail += f", last exit code {fields['ExecMainStatus']}"
        return status


# ------------------------------------------------------------------ Windows: Run key


class RunKey(Protocol):
    """The `HKCU\\…\\Run` value that starts the supervisor at logon."""

    def get(self) -> str | None: ...

    def set(self, command: str) -> None: ...

    def delete(self) -> bool: ...


class WinRegRunKey:  # pragma: no cover - Windows only
    def get(self) -> str | None:
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, WINDOWS_RUN_KEY) as key:
                value, _kind = winreg.QueryValueEx(key, WINDOWS_RUN_VALUE)
                return str(value)
        except FileNotFoundError:
            return None

    def set(self, command: str) -> None:
        import winreg

        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER, WINDOWS_RUN_KEY, 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.SetValueEx(key, WINDOWS_RUN_VALUE, 0, winreg.REG_SZ, command)

    def delete(self) -> bool:
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, WINDOWS_RUN_KEY, 0, winreg.KEY_SET_VALUE
            ) as key:
                winreg.DeleteValue(key, WINDOWS_RUN_VALUE)
                return True
        except FileNotFoundError:
            return False


def windows_pid_alive(pid: int) -> bool:  # pragma: no cover - Windows only
    """Whether `pid` is running. Never `os.kill(pid, 0)`: on Windows that kills it."""
    import ctypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel32.CloseHandle(handle)


def _windows_python(console: bool) -> str:
    """`python.exe` or `pythonw.exe` (no console window) of this environment."""
    here = Path(sys.executable)
    want = "python.exe" if console else "pythonw.exe"
    candidate = here.with_name(want)
    return str(candidate) if candidate.exists() else str(here)


def _windows_quote(args: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(args))


def _start_detached(args: Sequence[str]) -> None:  # pragma: no cover - Windows only
    flags = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED | NEW_PROCESS_GROUP | NO_WINDOW
    subprocess.Popen(
        list(args),
        creationflags=flags,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )


class WindowsRun:
    kind = f"Windows sign-in entry {WINDOWS_RUN_VALUE}"

    def __init__(
        self,
        *,
        run_key: RunKey | None = None,
        commands: Commands = run_command,
        start: Callable[[Sequence[str]], None] = _start_detached,
        pid_alive: Callable[[int], bool] = windows_pid_alive,
        home: Path | None = None,
        pythonw: str | None = None,
    ) -> None:
        self.run_key = run_key or WinRegRunKey()
        self.commands = commands
        self.start = start
        self.pid_alive = pid_alive
        self.home = home
        self.pythonw = pythonw

    def _home(self) -> Path:
        """The home in the installed entry (`--home "…"` comes last), else this one."""
        command = self.run_key.get() or ""
        found = re.search(r'--home (?:"([^"]+)"|(\S+))\s*$', command)
        if found:
            return Path(found.group(1) or found.group(2))
        return self.home or yantraxa_home()

    def supervisor_args(self, home: Path) -> list[str]:
        pythonw = self.pythonw or _windows_python(console=False)
        return [pythonw, "-m", "yantraxa_bridge.cli", "service", "supervise", "--home", str(home)]

    def _running_pid(self, home: Path) -> int | None:
        try:
            pid = int((home / PID_FILE).read_text().strip())
        except (OSError, ValueError):
            return None
        return pid if self.pid_alive(pid) else None

    def _stop(self, home: Path) -> bool:
        pid = self._running_pid(home)
        if pid is not None:
            self.commands(["taskkill", "/PID", str(pid), "/T", "/F"])
        (home / PID_FILE).unlink(missing_ok=True)
        return pid is not None

    def install(self, launch: Launch) -> list[str]:
        prepare_logs(launch)
        args = self.supervisor_args(launch.home)
        self.run_key.set(_windows_quote(args))
        self._stop(launch.home)
        self.start(args)
        return [f"Added the sign-in entry HKCU\\{WINDOWS_RUN_KEY}\\{WINDOWS_RUN_VALUE}"]

    def uninstall(self) -> list[str]:
        lines = []
        if self._stop(self._home()):
            lines.append("Stopped the background bridge")
        if self.run_key.delete():
            lines.append(f"Removed the sign-in entry {WINDOWS_RUN_VALUE}")
        return lines

    def status(self) -> Status:
        home = self._home()
        command = self.run_key.get()
        pid = self._running_pid(home)
        return Status(
            installed=command is not None,
            running=pid is not None,
            kind=self.kind,
            where=f"HKCU\\{WINDOWS_RUN_KEY}\\{WINDOWS_RUN_VALUE}",
            pid=pid,
        )


def supervise(
    home: Path,
    *,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    sleep: Callable[[float], None] = time.sleep,
    pid_alive: Callable[[int], bool] | None = None,
) -> int:
    """Windows: run `yantraxa-bridge run --service` until it exits 0, restarting it 10 s
    after a crash. One supervisor per home (`service.pid`). Runs under `pythonw`, so it
    never prints."""
    alive = pid_alive or (windows_pid_alive if os.name == "nt" else _posix_pid_alive)
    pid_file = home / PID_FILE
    try:
        other = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        other = None
    if other is not None and other != os.getpid() and alive(other):
        return 0
    ensure_private_dir(home / LOG_DIR)
    pid_file.write_text(str(os.getpid()))
    args = [_windows_python(console=True), "-m", "yantraxa_bridge.cli", "run", "--service"]
    env = {**os.environ, "YANTRAXA_HOME": str(home)}
    flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
    try:
        while True:
            with open(home / LOG_DIR / SERVICE_OUT, "ab") as out:
                child = popen(
                    args,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    env=env,
                    creationflags=flags,
                )
                code = child.wait()
            if code == 0:
                return 0
            sleep(RESTART_DELAY_S)
    finally:
        try:
            if pid_file.read_text().strip() == str(os.getpid()):
                pid_file.unlink()
        except OSError:
            pass


def _posix_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ------------------------------------------------------------------ choosing one


def current_backend() -> Backend:
    """The backend for this OS; tests replace it (see `tests/conftest.py`)."""
    if sys.platform == "darwin":
        return Launchd()
    if os.name == "nt":
        return WindowsRun()
    if sys.platform.startswith("linux"):
        return Systemd()
    raise ServiceError(f"no background service for {sys.platform}; use `yantraxa-bridge run`")


def last_turn(home: Path) -> str | None:
    """When the transparency log last recorded a turn, e.g. `2026-10-14 20:41 (3 min ago)`."""
    from .turnlog import read_events

    events = read_events(home, 1)
    if not events or not isinstance(events[-1].get("ts"), str):
        return None
    try:
        when = datetime.fromisoformat(events[-1]["ts"])
    except ValueError:
        return events[-1]["ts"]
    minutes = int((datetime.now(when.tzinfo) - when).total_seconds() // 60)
    if minutes < 1:
        ago = "just now"
    elif minutes < 120:
        ago = f"{minutes} min ago"
    elif minutes < 48 * 60:
        ago = f"{minutes // 60} h ago"
    else:
        ago = f"{minutes // (24 * 60)} days ago"
    return f"{when.strftime('%Y-%m-%d %H:%M')} ({ago})"


def last_log_line(home: Path) -> str | None:
    """The last line of `bridge.log`: why a stopped service stopped."""
    path = home / LOG_DIR / BRIDGE_LOG
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 4096))
            lines = handle.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return None
    lines = [line for line in lines if line.strip()]
    return lines[-1] if lines else None
