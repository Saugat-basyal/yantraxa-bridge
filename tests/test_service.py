"""`yantraxa-bridge service`, easy `pair` and `run --service`. The OS layer (launchctl,
systemctl, loginctl, taskkill, the Windows registry) is faked: nothing is installed."""

import logging
import os
import plistlib
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import httpx
import pytest
import respx
from typer.testing import CliRunner

from yantraxa_bridge import cli, runstate
from yantraxa_bridge import service as svc
from yantraxa_bridge.cli import app
from yantraxa_bridge.config import DEFAULT_SERVER, BridgeConfig, load_config, save_config
from yantraxa_bridge.credentials import Credentials, save_credentials
from yantraxa_bridge.turnlog import TurnLog

from .fake_cli import FakeCli, claude_result, install

runner = CliRunner()
SECRET = "s3cret-value-never-in-a-unit"
POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="fake CLIs are POSIX scripts")


def mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def agent_me():
    """`status` asks the server for its view; answer it here instead of localhost:4000."""
    with respx.mock(assert_all_called=False) as mock:
        mock.get("http://localhost:4000/v1/agent/me").mock(
            return_value=httpx.Response(
                200, json={"agent": {"handle": "saugat.ai", "state": "active"}, "permissions": {}}
            )
        )
        yield mock


def later(**delta) -> str:
    """An ISO time relative to now, the way runstate writes them."""
    from datetime import UTC, datetime, timedelta

    return (datetime.now(UTC) + timedelta(**delta)).isoformat(timespec="seconds")


def paired(adapter="claude-code", **config):
    save_config(BridgeConfig(server_url="http://localhost:4000", adapter=adapter, **config))
    save_credentials(Credentials(agent_id="ag_abc234", secret=SECRET))


class Commands:
    """A fake `launchctl` / `systemctl` / `loginctl` / `taskkill`: records argv, answers from
    a table keyed by the first words."""

    def __init__(self, answers=None):
        self.calls: list[list[str]] = []
        self.answers = answers or {}

    def __call__(self, args):
        args = list(args)
        self.calls.append(args)
        for n in range(len(args), 0, -1):
            answer = self.answers.get(" ".join(args[:n]))
            if answer is not None:
                if isinstance(answer, list):  # one per call, the last repeating
                    return answer.pop(0) if len(answer) > 1 else answer[0]
                return answer
        return svc.Result(0)

    def ran(self, *prefix):
        return [c for c in self.calls if c[: len(prefix)] == list(prefix)]


@pytest.fixture
def launch(home):
    home.mkdir(parents=True, exist_ok=True)
    return svc.Launch(
        program=["/Users/o b/.local/bin/yantraxa-bridge", "run", "--service"],
        home=home,
        path_env="/Users/o b/.local/bin:/opt/homebrew/bin:/usr/bin:/bin",
    )


@pytest.fixture
def restore_logging():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    for handler in root.handlers:
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)


# ------------------------------------------------------------------ what the service runs


def test_bridge_command_reuses_the_running_script(tmp_path):
    script = tmp_path / "bin" / "yantraxa-bridge"
    script.parent.mkdir()
    script.write_text("#!/bin/sh\n")
    assert svc.bridge_command(str(script)) == [str(script)]
    assert svc.bridge_command("/usr/bin/pytest") == [sys.executable, "-m", "yantraxa_bridge.cli"]


def test_service_path_keeps_this_shells_absolute_entries():
    path = svc.service_path("/opt/homebrew/bin::relative/bin:/usr/bin:/opt/homebrew/bin", ["/x"])
    entries = path.split(os.pathsep)
    assert entries[:3] == ["/opt/homebrew/bin", "/usr/bin", "/x"]
    assert "relative/bin" not in entries and "" not in entries
    assert len(entries) == len(set(entries))
    if os.name != "nt":
        assert "/bin" in entries


def test_make_launch_runs_run_service_with_this_home(home):
    launch = svc.make_launch()
    assert launch.program[-2:] == ["run", "--service"]
    assert launch.home == home.absolute()
    assert launch.log_dir == home.absolute() / "log"


# ------------------------------------------------------------------ macOS


def test_plist_has_the_home_path_and_restart_policy_and_no_secret(home, launch):
    paired()
    data = svc.render_plist(launch)
    plist = plistlib.loads(data)
    assert plist["Label"] == "com.yantraxa.bridge"
    assert plist["ProgramArguments"] == launch.program
    assert plist["EnvironmentVariables"] == {
        "YANTRAXA_HOME": str(home),
        "PATH": launch.path_env,
    }
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] == {"SuccessfulExit": False}
    assert plist["ThrottleInterval"] == 10
    assert plist["StandardOutPath"] == plist["StandardErrorPath"] == str(home / "log/service.out")
    assert SECRET.encode() not in data and b"ag_abc234" not in data


@pytest.mark.skipif(shutil.which("plutil") is None, reason="macOS plutil")
def test_plist_passes_plutil_lint(tmp_path, launch):
    copy = tmp_path / "com.yantraxa.bridge.plist"
    copy.write_bytes(svc.render_plist(launch))
    lint = subprocess.run(["plutil", "-lint", str(copy)], capture_output=True, text=True)
    assert lint.returncode == 0, lint.stdout + lint.stderr


NOT_LOADED = svc.Result(113, "", 'Could not find service "com.yantraxa.bridge"')
LOADED = svc.Result(
    0,
    "gui/501/com.yantraxa.bridge = {\n"
    "\tactive count = 1\n"
    "\tpath = /Users/o/Library/LaunchAgents/com.yantraxa.bridge.plist\n"
    "\tstate = running\n"
    "\tprogram = /Users/o/.local/bin/yantraxa-bridge\n"
    "\tpid = 5150\n"
    "\tlast exit code = (never exited)\n"
    "\tevent triggers = {\n"
    "\t\tstate = 0\n"
    "\t}\n"
    "}\n",
)


def test_launchd_install_writes_the_plist_and_bootstraps(tmp_path, launch):
    commands = Commands({"launchctl print": NOT_LOADED})
    backend = svc.Launchd(user_home=tmp_path / "user", uid=501, commands=commands)
    lines = backend.install(launch)
    plist = tmp_path / "user/Library/LaunchAgents/com.yantraxa.bridge.plist"
    assert lines == [f"Installed {plist}"]
    assert mode(plist) == 0o644
    assert plistlib.loads(plist.read_bytes())["ProgramArguments"] == launch.program
    assert mode(launch.log_dir) == 0o700 and mode(launch.out_path) == 0o600
    assert commands.ran("launchctl", "bootout") == []
    assert commands.ran("launchctl", "enable") == [
        ["launchctl", "enable", "gui/501/com.yantraxa.bridge"]
    ]
    assert commands.ran("launchctl", "bootstrap") == [
        ["launchctl", "bootstrap", "gui/501", str(plist)]
    ]


def test_launchd_reinstall_boots_out_first_and_retries_bootstrap(tmp_path, launch):
    commands = Commands(
        {
            "launchctl print": LOADED,
            "launchctl bootstrap": [
                svc.Result(5, "", "Bootstrap failed: 5: Input/output error"),
                svc.Result(0),
            ],
        }
    )
    sleeps = []
    backend = svc.Launchd(user_home=tmp_path, uid=501, commands=commands, sleep=sleeps.append)
    backend.install(launch)
    names = [c[1] for c in commands.calls]
    assert names.index("bootout") < names.index("bootstrap")
    assert len(commands.ran("launchctl", "bootstrap")) == 2 and sleeps == [0.5]


def test_launchd_bootstrap_that_keeps_failing_says_why(tmp_path, launch):
    commands = Commands(
        {"launchctl print": NOT_LOADED, "launchctl bootstrap": svc.Result(5, "", "nope")}
    )
    backend = svc.Launchd(user_home=tmp_path, uid=501, commands=commands, sleep=lambda _: None)
    with pytest.raises(svc.ServiceError, match="launchctl bootstrap failed: nope"):
        backend.install(launch)


def test_launchd_status_and_uninstall(tmp_path, launch):
    commands = Commands({"launchctl print": LOADED})
    backend = svc.Launchd(user_home=tmp_path, uid=501, commands=commands)
    backend.install(launch)
    status = backend.status()
    assert status.installed and status.running and status.pid == 5150
    assert status.detail == "last exit code (never exited)"
    assert status.home == launch.home and status.last_exit == "(never exited)"

    lines = backend.uninstall()
    assert commands.ran("launchctl", "bootout")[-1] == [
        "launchctl",
        "bootout",
        "gui/501/com.yantraxa.bridge",
    ]
    assert not backend.plist.exists() and len(lines) == 2

    commands.answers["launchctl print"] = NOT_LOADED
    assert backend.uninstall() == []
    assert backend.status() == svc.Status(False, False, backend.kind, str(backend.plist))


def test_launchd_label_can_be_replaced_for_tests(tmp_path, launch, monkeypatch):
    """A real-service test on a Mac runs under its own label, never the person's own
    com.yantraxa.bridge."""
    commands = Commands({"launchctl print": NOT_LOADED})
    backend = svc.Launchd(user_home=tmp_path, uid=501, commands=commands, label="com.test.b")
    backend.install(launch)
    assert backend.plist.name == "com.test.b.plist"
    assert plistlib.loads(backend.plist.read_bytes())["Label"] == "com.test.b"
    assert commands.ran("launchctl", "bootstrap")[-1][-1] == str(backend.plist)
    assert backend.target == "gui/501/com.test.b"


@pytest.mark.skipif(sys.platform != "darwin", reason="launchd only")
def test_current_backend_reads_the_test_label(monkeypatch):
    monkeypatch.undo()  # the real current_backend: it only builds a Launchd, runs nothing
    monkeypatch.setenv(svc.LABEL_ENV, "com.yantraxa.bridge.test-1")
    assert svc.current_backend().label == "com.yantraxa.bridge.test-1"
    monkeypatch.delenv(svc.LABEL_ENV)
    assert svc.current_backend().label == "com.yantraxa.bridge"


# ------------------------------------------------------------------ Linux


def test_unit_quotes_paths_and_escapes_specifiers(launch):
    launch.program = ["/home/o b/bin/yantraxa-bridge", "run", "--service"]
    launch.path_env = "/home/o b/bin:/usr/bin:/odd%dir"
    unit = svc.render_unit(launch)
    assert 'ExecStart="/home/o b/bin/yantraxa-bridge" run --service' in unit
    assert 'Environment="PATH=/home/o b/bin:/usr/bin:/odd%%dir"' in unit
    assert f"Environment=YANTRAXA_HOME={launch.home}" in unit
    assert "Restart=on-failure" in unit and "RestartSec=10" in unit
    assert f"StandardOutput=append:{launch.out_path}" in unit
    assert "WantedBy=default.target" in unit


def test_systemd_quote_escapes_dollars_in_exec_and_quotes():
    assert svc._systemd_quote("a$b", exec_arg=True) == "a$$b"
    assert svc._systemd_quote('say "hi"') == '"say \\"hi\\""'
    assert svc._systemd_quote("") == '""'


def test_systemd_install_enables_restarts_and_mentions_lingering(tmp_path, launch, home):
    paired()
    commands = Commands({"loginctl show-user": svc.Result(0, "Linger=no\n")})
    backend = svc.Systemd(config_home=tmp_path / "cfg", user="saugat", commands=commands)
    lines = backend.install(launch)
    unit = tmp_path / "cfg/systemd/user/yantraxa-bridge.service"
    assert mode(unit) == 0o644 and SECRET not in unit.read_text()
    assert [c[2:] for c in commands.ran("systemctl", "--user")] == [
        ["daemon-reload"],
        ["enable", "yantraxa-bridge.service"],
        ["restart", "yantraxa-bridge.service"],
    ]
    assert "sudo loginctl enable-linger saugat" in lines[-1]
    assert not any("sudo" in " ".join(c) for c in commands.calls)  # never run, only shown


def test_systemd_install_without_lingering_hint_when_lingering(tmp_path, launch):
    commands = Commands({"loginctl show-user": svc.Result(0, "Linger=yes\n")})
    backend = svc.Systemd(config_home=tmp_path, user="saugat", commands=commands)
    assert len(backend.install(launch)) == 1


def test_systemd_missing_or_without_a_user_session(tmp_path, launch):
    no_systemd = Commands({"systemctl --version": svc.Result(127, "", "not found")})
    with pytest.raises(svc.ServiceError, match="no systemd"):
        svc.Systemd(config_home=tmp_path, commands=no_systemd).install(launch)
    no_bus = Commands(
        {"systemctl --user daemon-reload": svc.Result(1, "", "Failed to connect to bus")}
    )
    with pytest.raises(svc.ServiceError, match="Failed to connect to bus"):
        svc.Systemd(config_home=tmp_path, commands=no_bus).install(launch)


def test_systemd_status_and_uninstall_removes_overrides(tmp_path, launch):
    commands = Commands(
        {
            "systemctl --user show": svc.Result(
                0, "ActiveState=active\nSubState=running\nMainPID=777\nExecMainStatus=0\n"
            ),
        }
    )
    backend = svc.Systemd(config_home=tmp_path, user="saugat", commands=commands)
    backend.install(launch)
    status = backend.status()
    assert status.installed and status.running and status.pid == 777
    assert status.detail == "active (running)"
    assert status.home == launch.home

    backend.dropins.mkdir()
    (backend.dropins / "override.conf").write_text("[Service]\nEnvironment=K=v\n")
    lines = backend.uninstall()
    assert ["systemctl", "--user", "disable", "--now", "yantraxa-bridge.service"] in commands.calls
    assert not backend.unit.exists() and not backend.dropins.exists()
    assert len(lines) == 2
    assert backend.uninstall() == []


def test_systemd_status_of_a_stopped_unit(tmp_path, launch):
    commands = Commands(
        {
            "systemctl --user show": svc.Result(
                0, "ActiveState=inactive\nSubState=dead\nMainPID=0\nExecMainStatus=0\n"
            )
        }
    )
    backend = svc.Systemd(config_home=tmp_path, commands=commands)
    backend.install(launch)
    status = backend.status()
    assert not status.running and status.pid is None
    assert status.detail == "inactive (dead), last exit code 0"


# ------------------------------------------------------------------ Windows


class FakeRunKey:
    def __init__(self):
        self.value = None

    def get(self):
        return self.value

    def set(self, command):
        self.value = command

    def delete(self):
        had, self.value = self.value is not None, None
        return had


def test_windows_install_adds_a_sign_in_entry_and_starts_the_supervisor(tmp_path, launch):
    key, started, alive = FakeRunKey(), [], {111}
    commands = Commands()
    backend = svc.WindowsRun(
        run_key=key,
        commands=commands,
        start=started.append,
        pid_alive=lambda pid: pid in alive,
        pythonw=r"C:\Users\o b\uv\tools\yantraxa-bridge\Scripts\pythonw.exe",
    )
    (launch.home / "service.pid").write_text("111")  # an older supervisor
    backend.install(launch)
    assert key.value.startswith(r'"C:\Users\o b\uv\tools\yantraxa-bridge\Scripts\pythonw.exe"')
    assert key.value.endswith(f" -m yantraxa_bridge.cli service supervise --home {launch.home}")
    assert commands.calls == [["taskkill", "/PID", "111", "/T", "/F"]]
    assert started == [backend.supervisor_args(launch.home)]
    assert mode(launch.out_path) == 0o600


def test_windows_status_and_uninstall_use_the_installed_home(tmp_path, launch):
    key, alive = FakeRunKey(), {222}
    commands = Commands()
    backend = svc.WindowsRun(
        run_key=key,
        commands=commands,
        start=lambda _args: None,
        pid_alive=lambda pid: pid in alive,
        home=tmp_path / "elsewhere",
        pythonw="pythonw.exe",
    )
    assert backend.status().installed is False
    backend.install(launch)
    (launch.home / "service.pid").write_text("222")
    status = backend.status()
    assert status.installed and status.running and status.pid == 222

    lines = backend.uninstall()
    assert commands.calls[-1] == ["taskkill", "/PID", "222", "/T", "/F"]
    assert key.value is None and not (launch.home / "service.pid").exists()
    assert len(lines) == 2


class FakeChild:
    def __init__(self, code):
        self.code = code

    def wait(self):
        return self.code


def test_supervise_restarts_after_a_crash_and_stops_after_a_clean_exit(home):
    home.mkdir()
    codes, launched, sleeps = [1, 0], [], []

    def popen(args, **kwargs):
        launched.append((args, kwargs["env"]["YANTRAXA_HOME"]))
        assert (home / "service.pid").read_text() == str(os.getpid())
        return FakeChild(codes.pop(0))

    assert svc.supervise(home, popen=popen, sleep=sleeps.append) == 0
    assert len(launched) == 2 and sleeps == [10]
    assert launched[0][0][-3:] == ["yantraxa_bridge.cli", "run", "--service"]
    assert launched[0][1] == str(home)
    assert not (home / "service.pid").exists()


def test_supervise_leaves_a_running_supervisor_alone(home):
    home.mkdir()
    (home / "service.pid").write_text("999999")
    launched = []
    assert svc.supervise(home, popen=launched.append, pid_alive=lambda pid: True) == 0
    assert launched == [] and (home / "service.pid").read_text() == "999999"


# ------------------------------------------------------------------ service commands


def test_service_install_needs_pairing(home, service_backend):
    result = runner.invoke(app, ["service", "install"])
    assert result.exit_code == 1 and "not paired yet" in result.output
    assert service_backend.installed == []


def test_service_install_status_uninstall(home, service_backend, monkeypatch):
    paired(adapter="ollama", model="llama3.1:8b")
    result = runner.invoke(app, ["service", "install"])
    assert result.exit_code == 0, result.output
    [launch] = service_backend.installed
    assert launch.program[-2:] == ["run", "--service"] and launch.home == home.absolute()
    assert "runs in the background (fake service)" in result.output
    assert str(home.absolute() / "log" / "bridge.log") in result.output
    assert SECRET not in result.output

    TurnLog(home).record("sent", meeting="AB-1234")
    status = runner.invoke(app, ["service", "status"])
    assert status.exit_code == 0
    assert "Installed:   yes, fake" in status.output
    assert "Running:     yes (pid 4242)" in status.output
    assert "(just now)" in status.output

    assert "Service:     installed, running" in runner.invoke(app, ["status"]).output

    removed = runner.invoke(app, ["service", "uninstall"])
    assert removed.exit_code == 0 and "Removed the background service." in removed.output
    assert "No background service installed." in runner.invoke(app, ["service", "uninstall"]).output
    status = runner.invoke(app, ["service", "status"])
    assert "Installed:   no" in status.output and "Last turn:" in status.output


def test_service_status_of_a_stopped_service_shows_the_last_log_line(home, service_backend):
    service_backend.is_installed = True
    (home / "log").mkdir(parents=True)
    (home / "log" / "bridge.log").write_text("… ERROR stopped: not paired yet\n\n")
    result = runner.invoke(app, ["service", "status"])
    assert "Running:     no" in result.output
    assert "Last turn:   none logged yet" in result.output
    assert "Last log:    … ERROR stopped: not paired yet" in result.output


def test_service_install_warns_when_the_cli_is_not_on_the_path(home, service_backend, monkeypatch):
    paired(adapter="codex-cli")
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setattr(svc, "service_path", lambda env_path=None, extra=(): "/nonexistent")
    result = runner.invoke(app, ["service", "install"])
    assert result.exit_code == 0 and "`codex` isn't on this PATH" in result.output


def test_service_install_explains_api_key_mode(home, service_backend):
    paired(api_key_env="ANTHROPIC_API_KEY")
    result = runner.invoke(app, ["service", "install"])
    assert "API-key mode: the service needs ANTHROPIC_API_KEY" in result.output


def test_service_errors_exit_1(home, monkeypatch):
    def broken():
        raise svc.ServiceError("no background service for plan9")

    monkeypatch.setattr(svc, "current_backend", broken)
    paired()
    for command in ("install", "uninstall", "status"):
        result = runner.invoke(app, ["service", command])
        assert result.exit_code == 1 and "plan9" in result.output
    assert "not available on this system" in runner.invoke(app, ["status"]).output


# ------------------------------------------------------------------ run --service


def test_run_service_exits_1_when_not_paired_and_says_what_to_run(home, restore_logging):
    """Nothing to retry: a non-zero exit, so the service manager starts it again 10 s later
    and a pairing made meanwhile is picked up."""
    result = runner.invoke(app, ["run", "--service"])
    assert result.exit_code == 1
    log = home / "log" / "bridge.log"
    assert "stopped: not paired yet: run `" in log.read_text()
    assert " pair`" in log.read_text()
    assert mode(log) == 0o600 and mode(home / "log") == 0o700
    state = runstate.read_state(home)
    assert state["mode"] == "service" and state["last_error"].startswith("not paired yet")
    assert mode(home / "run-state.json") == 0o600


def test_run_service_exits_1_on_a_config_error(home, restore_logging):
    paired(adapter="ollama")  # no model set
    result = runner.invoke(app, ["run", "--service"])
    assert result.exit_code == 1
    assert "set `model`" in (home / "log" / "bridge.log").read_text()


@pytest.mark.parametrize(
    ("reason", "service", "code"),
    [
        ("superseded", True, 0),
        ("disconnected", True, 0),
        ("superseded", False, 1),
        ("disconnected", False, 0),
        ("unauthorized", False, 1),
    ],
)
def test_run_exit_codes_for_final_stops(home, restore_logging, monkeypatch, reason, service, code):
    from yantraxa_bridge.client import StopReason

    paired(adapter="interactive-mcp")

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self):
            return StopReason(reason)

        async def stop(self):
            pass

    monkeypatch.setattr(cli, "BridgeClient", Client)
    monkeypatch.setattr(cli, "run_diagnose", _no_checks)
    args = ["run", "--service"] if service else ["run"]
    result = runner.invoke(app, args)
    assert result.exit_code == code, result.output
    if service:
        assert "stopped" in (home / "log" / "bridge.log").read_text()
    assert runstate.read_state(home)["stopped_at"]


async def _no_checks():
    return []


class ScriptedClient:
    """A BridgeClient stand-in that plays one script entry per `run()`: a StopReason, or
    "connect" (reports a connection, then waits to be stopped)."""

    script: ClassVar[list] = []
    made: ClassVar[list] = []

    def __init__(self, config, credentials, *, turn_handler, on_connected, on_error):
        self.credentials = credentials
        self.on_connected, self.on_error = on_connected, on_error
        self.stopped = None
        ScriptedClient.made.append(self)

    async def run(self):
        import asyncio

        from yantraxa_bridge.client import StopReason

        step = ScriptedClient.script.pop(0)
        if step == "connect":
            self.on_connected({"handle": "saugat.ai"})
            self.stopped = asyncio.Event()
            ScriptedClient.on_connect(self)
            await self.stopped.wait()
            return StopReason.STOPPED
        if step is StopReason.UNAUTHORIZED:
            self.on_error("server closed the connection (4001): unauthorized")
        return step

    async def stop(self):
        if self.stopped is not None:
            self.stopped.set()


def test_run_service_stays_up_through_a_refused_credential_and_backs_off(
    home, restore_logging, monkeypatch, agent_me
):
    """The service never stops for good on a refusal: it backs off (10 s doubling to 5 min,
    shortened here), records the last error for `status`, reads the credential again each
    time, and connects once the server accepts it."""
    import os
    import signal

    from yantraxa_bridge.client import StopReason

    paired(adapter="interactive-mcp")
    delays = []
    real_backoff = cli.Backoff

    class RecordingBackoff(real_backoff):
        def next(self):
            delay = super().next()
            delays.append(delay)
            return delay

    monkeypatch.setattr(cli, "Backoff", RecordingBackoff)
    monkeypatch.setattr(cli, "SERVICE_RETRY_INITIAL_S", 0.01)
    monkeypatch.setattr(cli, "SERVICE_RETRY_MAX_S", 0.04)
    ScriptedClient.script = [StopReason.UNAUTHORIZED] * 4 + ["connect"]
    ScriptedClient.made = []
    statuses = []

    def on_connect(client):
        # What `status` says while it runs: connected, after the errors.
        statuses.append(runner.invoke(app, ["status"]).output)
        statuses.append(runstate.read_state(home))
        os.kill(os.getpid(), signal.SIGTERM)  # launchd stopping it

    ScriptedClient.on_connect = on_connect
    monkeypatch.setattr(cli, "BridgeClient", ScriptedClient)
    result = runner.invoke(app, ["run", "--service"])

    assert result.exit_code == 0, result.output
    assert len(ScriptedClient.made) == 5  # four refusals, then a connection
    assert len(delays) == 4
    assert delays[0] == pytest.approx(0.01, rel=0.25)
    assert delays[-1] == pytest.approx(0.04, rel=0.25)  # capped
    log = (home / "log" / "bridge.log").read_text()
    assert log.count("the server refused this bridge's credential; retrying in") == 4
    assert "stopped:" not in log
    state = statuses[1]
    assert state["handle"] == "saugat.ai" and state["mode"] == "service"
    assert state["last_error"] == "the server refused this bridge's credential"
    assert runstate.connected_now(state)
    final = runstate.read_state(home)
    assert final["stopped"] == "stopped by a signal"


def test_status_shows_a_running_service_s_last_error(home, service_backend, monkeypatch, agent_me):
    import os

    paired(adapter="interactive-mcp")
    service_backend.is_installed = service_backend.running = True
    service_backend.home = home
    writer = runstate.StateWriter(home, mode="service")
    writer.started()
    writer.connected({"handle": "saugat.ai"})
    failed_at = later(seconds=1)  # after the connection, whatever the date the test runs
    monkeypatch.setattr(runstate, "now", lambda: failed_at)
    writer.error("can't connect to wss://api.yantraxa.com/ws/agent: timed out")
    assert writer.state["pid"] == os.getpid()
    result = runner.invoke(app, ["status"])
    when = runstate.local_time(failed_at)
    assert (
        "Service:     installed, running, last error: can't connect to "
        f"wss://api.yantraxa.com/ws/agent: timed out at {when}" in result.output
    )
    service_status = runner.invoke(app, ["service", "status"]).output
    assert "Connection:  last error: can't connect" in service_status


def test_status_of_a_connected_service(home, service_backend, agent_me):
    paired(adapter="interactive-mcp")
    service_backend.is_installed = service_backend.running = True
    writer = runstate.StateWriter(home, mode="service")
    writer.started()
    writer.error("connection closed (1006); reconnecting")
    writer.state["last_error_at"] = later(hours=-1)
    writer.connected({"handle": "saugat.ai"})
    output = runner.invoke(app, ["status"]).output
    assert "Service:     installed, running, connected as @saugat.ai since " in output


def test_a_state_file_left_by_a_crash_survives_the_restart(home):
    writer = runstate.StateWriter(home, mode="service")
    writer.started()
    writer.error("boom")
    again = runstate.StateWriter(home, mode="service")
    again.started()
    assert again.state["last_error"] == "boom" and "connected_at" not in again.state


def test_run_service_trims_a_huge_service_out(home, restore_logging):
    (home / "log").mkdir(parents=True)
    out = home / "log" / "service.out"
    with out.open("wb") as handle:
        handle.truncate(svc.MAX_SERVICE_OUT_BYTES + 1)
    runner.invoke(app, ["run", "--service"])
    assert out.stat().st_size == 0


# ------------------------------------------------------------------ easy pair


@pytest.fixture
def fake_pair(monkeypatch):
    """`pair` without a server or a CLI: the preflight passes and approval is instant."""
    seen = {}

    async def preflight(config, *, allow_login):
        seen["preflight"] = config.adapter
        return "Preflight OK"

    async def run_pair(server, adapter, host_label):
        seen.update(server=server, adapter=adapter)
        save_credentials(Credentials(agent_id="ag_abc234", secret=SECRET))
        save_config(load_config().model_copy(update={"server_url": server, "adapter": adapter}))

    monkeypatch.setattr(cli, "_preflight", preflight)
    monkeypatch.setattr(cli, "run_pair", run_pair)
    return seen


@pytest.fixture
def both_clis(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, "claude")
    FakeCli(tmp_path / "bin", "codex")


def interactive(monkeypatch, value=True):
    monkeypatch.setattr(cli, "_interactive", lambda: value)


@POSIX_ONLY
def test_pair_with_no_flags_uses_the_only_cli_and_the_default_server(
    home, fake_pair, monkeypatch, tmp_path, service_backend
):
    install(monkeypatch, tmp_path, "codex")
    result = runner.invoke(app, ["pair"])
    assert result.exit_code == 0, result.output
    assert "Using Codex (`codex`)." in result.output
    assert fake_pair == {"preflight": "codex-cli", "server": DEFAULT_SERVER, "adapter": "codex-cli"}
    assert DEFAULT_SERVER == "https://api.yantraxa.com"
    # Not a terminal (`curl … | sh`): nothing asked, and the service is set up all the same.
    assert len(service_backend.installed) == 1
    assert "?" not in result.output.replace("Start automatically", "")


@POSIX_ONLY
def test_pair_asks_which_cli_when_both_are_installed(home, fake_pair, both_clis, monkeypatch):
    interactive(monkeypatch)
    result = runner.invoke(app, ["pair", "--no-service"], input="3\n2\n")
    assert result.exit_code == 0, result.output
    assert "Which should answer for this agent?" in result.output
    assert fake_pair["adapter"] == "codex-cli"


@POSIX_ONLY
def test_pair_without_a_terminal_and_both_clis_needs_adapter(home, fake_pair, both_clis):
    result = runner.invoke(app, ["pair", "--yes"])
    assert result.exit_code == 1
    assert "pass --adapter claude-code or --adapter codex-cli" in result.output
    assert "server" not in fake_pair


def test_pair_without_any_cli_says_what_to_do(home, fake_pair, monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    result = runner.invoke(app, ["pair"])
    assert result.exit_code == 1
    assert "neither `claude` nor `codex` is on the PATH" in result.output


def connects_on_install(service_backend, handle="saugat.ai"):
    """The fake service's `run --service` connects as soon as it is installed."""

    def on_install(launch):
        writer = runstate.StateWriter(launch.home, mode="service")
        writer.started()
        writer.connected({"handle": handle})

    service_backend.on_install = on_install


def test_pair_yes_installs_the_service(home, fake_pair, service_backend):
    connects_on_install(service_backend)
    result = runner.invoke(app, ["pair", "--adapter", "claude-code", "--yes"])
    assert result.exit_code == 0, result.output
    assert "Preflight OK" in result.output
    assert len(service_backend.installed) == 1
    assert "runs in the background" in result.output
    assert result.output.rstrip().endswith(
        "Connected as @saugat.ai. It starts automatically when you log in."
    )


@pytest.mark.parametrize("terminal", [True, False])
def test_pair_starts_the_service_without_asking_and_waits_for_it(
    home, fake_pair, service_backend, monkeypatch, terminal
):
    """0.2.1: a person watching the browser never sees a terminal question, so pairing ends
    with the service started, on a terminal or not, and says whether it connected."""
    interactive(monkeypatch, terminal)
    connects_on_install(service_backend)
    result = runner.invoke(app, ["pair", "--adapter", "claude-code"], input="n\n")
    assert result.exit_code == 0, result.output
    assert "[Y/n]" not in result.output and "?" not in result.output
    assert len(service_backend.installed) == 1
    assert service_backend.installed[0].home == home.absolute()
    assert "Connected as @saugat.ai. It starts automatically when you log in." in result.output


def test_pair_says_why_the_service_has_not_connected(home, fake_pair, service_backend):
    def on_install(launch):
        writer = runstate.StateWriter(launch.home, mode="service")
        writer.started()
        writer.error("can't connect to wss://api.yantraxa.com/ws/agent: timed out")

    service_backend.on_install = on_install
    result = runner.invoke(app, ["pair", "--adapter", "claude-code"])
    assert result.exit_code == 0, result.output
    assert (
        "Paired, but not connected yet: can't connect to wss://api.yantraxa.com/ws/agent: "
        "timed out." in result.output
    )
    # The fake service runs: `run` would start a second bridge and replace it (4009).
    assert "The background bridge is running and keeps trying." in result.output
    lines = result.output.splitlines()
    assert "If it is still not connected in a minute, run: " + cli._command("doctor") in lines
    # The install's `Runs: … run --service` line contains the `run` command, so this checks
    # the hint lines, not a bare substring (CI's unquoted path matched it; 0.2.1 release).
    assert not [line for line in lines if line.endswith(": " + cli._command("run"))]
    assert "  Runs: " + cli._shell_line([*svc.bridge_command(), "run", "--service"]) in lines
    assert f"Log: {home / 'log' / 'bridge.log'}" in result.output


def test_pair_says_when_the_service_is_not_running(home, fake_pair, service_backend):
    def on_install(launch):
        service_backend.running = False
        service_backend.last_exit = "1"

    service_backend.on_install = on_install
    result = runner.invoke(app, ["pair", "--adapter", "claude-code"])
    assert (
        "Paired, but not connected yet: the background bridge isn't running (last exit code 1)."
        in (result.output)
    )
    assert "If it is still not connected in a minute, run: " + cli._command("run") in (
        result.output
    )


def test_pair_ignores_an_older_connection_in_the_state_file(home, fake_pair, service_backend):
    writer = runstate.StateWriter(home, mode="service")
    writer.started()
    writer.state["connected_at"] = "2026-10-15T06:00:00+00:00"
    writer.state["handle"] = "old"
    writer._write()
    result = runner.invoke(app, ["pair", "--adapter", "claude-code"])
    assert "Connected as" not in result.output
    assert "Paired, but not connected yet: no connection within 0 s." in result.output


def test_pair_again_ignores_the_stop_of_the_service_it_replaces(home, fake_pair, service_backend):
    """Reinstalling stops the running service, which records its stop; that is not the new
    service stopping."""
    old = runstate.StateWriter(home, mode="service")
    old.started()
    old.state["started_at"] = "2026-10-15T06:00:00+00:00"
    old.connected({"handle": "saugat.ai"})

    def on_install(launch):
        old.stopped("stopped by a signal")  # launchd bootout, during the install
        new = runstate.StateWriter(launch.home, mode="service")
        new.started()
        new.connected({"handle": "saugat.ai"})

    service_backend.on_install = on_install
    result = runner.invoke(app, ["pair", "--adapter", "claude-code"])
    assert "Connected as @saugat.ai." in result.output, result.output


def test_command_prints_the_full_path_of_the_installed_bridge(monkeypatch, tmp_path):
    script = tmp_path / "o b" / ".local" / "bin" / "yantraxa-bridge"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\n")
    monkeypatch.setattr(sys, "argv", [str(script)])
    assert cli._command("run") == f"'{script}' run"
    windows = r"C:\Users\O B\.local\bin\yantraxa-bridge.exe"
    monkeypatch.setattr(svc, "bridge_command", lambda: [windows])
    monkeypatch.setattr(cli.os, "name", "nt")
    assert cli._command("service", "install") == f'& "{windows}" service install'
    plain = r"C:\Users\ob\.local\bin\yantraxa-bridge.exe"
    monkeypatch.setattr(svc, "bridge_command", lambda: [plain])
    assert cli._command("run") == f"{plain} run"


def test_pair_yes_no_service_asks_nothing_and_installs_nothing(
    home, fake_pair, service_backend, monkeypatch
):
    interactive(monkeypatch)  # even on a terminal
    result = runner.invoke(
        app,
        ["pair", "--server", DEFAULT_SERVER, "--adapter", "claude-code", "--yes", "--no-service"],
    )
    assert result.exit_code == 0, result.output
    assert "?" not in result.output
    assert service_backend.installed == []
    assert "Start the bridge with: " + cli._command("run") in result.output


def test_pair_no_service_points_at_an_installed_service(home, fake_pair, service_backend):
    service_backend.is_installed = True
    service_backend.home = home
    result = runner.invoke(app, ["pair", "--adapter", "claude-code", "--no-service"])
    assert "restart it with the new pairing: " + cli._command("service", "install") in result.output


def test_pair_no_service_ignores_a_service_for_another_home(
    home, fake_pair, service_backend, tmp_path
):
    """One service per user: a service running ~/.yantraxa is not this home's."""
    service_backend.is_installed = True
    service_backend.home = tmp_path / "someone-elses-home"
    result = runner.invoke(app, ["pair", "--adapter", "claude-code", "--no-service"])
    assert "restart it" not in result.output
    assert "Start the bridge with: " + cli._command("run") in result.output


def test_pair_succeeds_even_when_the_service_cannot_install(home, fake_pair, monkeypatch):
    def broken():
        raise svc.ServiceError("no background service for plan9")

    monkeypatch.setattr(svc, "current_backend", broken)
    result = runner.invoke(app, ["pair", "--adapter", "claude-code", "--yes"])
    assert result.exit_code == 0
    assert "Couldn't set up the background service: no background service for plan9" in (
        result.output
    )
    assert "Start the bridge with: " + cli._command("run") in result.output


def test_pair_keeps_the_installers_flags(home, fake_pair):
    """phase 4a's installer runs exactly this."""
    result = runner.invoke(
        app, ["pair", "--server", "https://api.yantraxa.com", "--adapter", "ollama"]
    )
    assert result.exit_code == 0, result.output
    assert fake_pair == {"server": "https://api.yantraxa.com", "adapter": "ollama"}


# ------------------------------------------------------------------ diagnose and doctor


@POSIX_ONLY
def test_diagnose_reports_the_claude_facts_without_a_turn(home, monkeypatch, tmp_path):
    import json

    claude = install(monkeypatch, tmp_path, "claude")
    claude.behave(
        {
            "--version": {"stdout": "2.1.299 (Claude Code)\n"},
            "auth status": {"stdout": json.dumps({"loggedIn": True, "email": "o@example.com"})},
            "default": {"stdout": claude_result({"action": "pass"})},
        }
    )
    paired()
    result = runner.invoke(app, ["diagnose"])
    assert "adapter      Claude Code 2.1.299, signed in as o@example.com" in result.output
    assert [c["argv"] for c in claude.calls()] == [["--version"], ["auth", "status"]]

    result = runner.invoke(app, ["doctor"])
    assert "probe        probe OK (" in result.output
    assert len(claude.calls()) == 5  # diagnose's two, then the probe turn


@POSIX_ONLY
def test_diagnose_says_how_to_sign_codex_in_and_doctor_skips_the_probe(home, monkeypatch, tmp_path):
    codex = install(monkeypatch, tmp_path, "codex")
    codex.behave({"--version": {"stdout": "codex-cli 0.158.0\n"}, "login status": {"exit": 1}})
    paired(adapter="codex-cli")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "[FAIL] adapter" in result.output and "codex login" in result.output
    assert "probe        skipped: fix the adapter first" in result.output
    assert not any(c["argv"][:1] == ["exec"] for c in codex.calls())


# ------------------------------------------------------------------ doctor: is it running?


@pytest.fixture
def no_server(monkeypatch):
    """`doctor` without the network: diagnose's checks are skipped."""

    async def checks():
        from yantraxa_bridge.diagnose import Check

        return [Check("config", True, "fake")]

    monkeypatch.setattr(cli, "run_diagnose", checks)


def running_line(output):
    return next(line for line in output.splitlines() if " running " in line[:20])


def test_doctor_fails_when_paired_and_nothing_runs(home, service_backend, no_server):
    paired(adapter="interactive-mcp")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    line = running_line(result.output)
    assert line.startswith("[FAIL] running")
    assert "paired, but nothing runs the bridge" in line
    assert cli._command("service", "install") in line


def test_doctor_fails_when_the_service_is_installed_but_stopped(home, service_backend, no_server):
    """The owner's Mac on 2026-10-15: installed, not running, last exit code 0."""
    paired(adapter="interactive-mcp")
    service_backend.is_installed, service_backend.home = True, home
    service_backend.last_exit = "0"
    (home / "log").mkdir(parents=True, exist_ok=True)
    (home / "log" / "bridge.log").write_text(
        "2026-10-15 06:44:55 ERROR yantraxa_bridge.cli: stopped: the server refused this "
        "bridge's credential\n"
    )
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    line = running_line(result.output)
    assert "installed but not running (last exit code 0)" in line
    assert "last log: 2026-10-15 06:44:55 ERROR" in line
    assert "Fix: run `" + cli._command("service", "install") in line


def test_doctor_fails_when_the_service_runs_another_home(
    home, service_backend, no_server, tmp_path
):
    paired(adapter="interactive-mcp")
    service_backend.is_installed = service_backend.running = True
    service_backend.home = tmp_path / "other"
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "runs another home" in running_line(result.output)


def test_doctor_ok_when_the_service_is_connected(home, service_backend, no_server):
    paired(adapter="interactive-mcp")
    service_backend.is_installed = service_backend.running = True
    service_backend.home = home
    writer = runstate.StateWriter(home, mode="service")
    writer.started()
    writer.connected({"handle": "saugat.ai"})
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert running_line(result.output).startswith("[  ok] running")
    assert "running in the background service" in result.output
    assert "connected as @saugat.ai" in result.output


def test_doctor_fails_when_it_runs_but_keeps_failing(home, service_backend, no_server):
    paired(adapter="interactive-mcp")
    writer = runstate.StateWriter(home, mode="terminal")
    writer.started()
    writer.error("the server refused this bridge's credential")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    line = running_line(result.output)
    assert "running in a terminal" in line and "not connected: last error: the server" in line


def test_doctor_ignores_a_dead_bridge_s_state(home, service_backend, no_server, monkeypatch):
    paired(adapter="interactive-mcp")
    writer = runstate.StateWriter(home, mode="terminal")
    writer.started()
    writer.connected({"handle": "saugat.ai"})
    monkeypatch.setattr(runstate, "pid_alive", lambda pid: False)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "nothing runs the bridge" in running_line(result.output)


def test_doctor_says_nothing_about_running_before_pairing(home, service_backend, no_server):
    result = runner.invoke(app, ["doctor"])
    assert " running " not in result.output


# ------------------------------------------------------------------ review fixes (0.2.1)


def test_service_backoff_is_10_s_doubling_to_5_min():
    assert cli.SERVICE_RETRY_INITIAL_S == 10
    assert cli.SERVICE_RETRY_MAX_S == 300
    backoff = cli.Backoff(cli.SERVICE_RETRY_INITIAL_S, cli.SERVICE_RETRY_MAX_S, rng=lambda: 0.5)
    assert [backoff.next() for _ in range(7)] == [10, 20, 40, 80, 160, 300, 300]


def test_run_state_is_not_the_caps_store(home):
    """`state.json` holds the turn caps (caps.py); the run state lives beside it."""
    from yantraxa_bridge import caps

    assert runstate.STATE_FILE != caps.STATE_FILE
    assert runstate.state_path(home).name == "run-state.json"


MEETING = "3f2c6a5e-8a0e-4f7a-9d55-0d3c2a1b4c5d"


def used_caps(home):
    """Seven turns today, two in one meeting, and the vendor backoff on: what a reconnect
    must not wipe."""
    from yantraxa_bridge.caps import Caps

    home.mkdir(parents=True, exist_ok=True)
    caps = Caps(home, per_meeting=5, per_day=50)
    for meeting in [MEETING, MEETING, None, None, None, None, None]:
        caps.record_turn(meeting)
    caps.vendor_limited()
    caps.vendor_limited()
    return caps


def assert_caps_kept(caps, before: bytes):
    assert caps.path.read_bytes() == before
    data = caps._load()
    assert data["day_count"] == 7
    assert data["meetings"][MEETING]["count"] == 2
    assert float(data["backoff_until"]) > 0


def test_a_service_run_leaves_the_caps_counts_alone(home, restore_logging, monkeypatch):
    """A service run that is refused, retries, connects and reconnects writes its run state
    many times; the caps' counts and vendor backoff in state.json stay as they were."""
    import os
    import signal

    from yantraxa_bridge.client import StopReason

    paired(adapter="interactive-mcp")
    caps = used_caps(home)
    before = caps.path.read_bytes()
    monkeypatch.setattr(cli, "SERVICE_RETRY_INITIAL_S", 0.01)
    ScriptedClient.script = [StopReason.UNAUTHORIZED, "connect"]
    ScriptedClient.made = []

    def on_connect(client):
        client.on_error("connection closed (1006); reconnecting")
        client.on_connected({"handle": "saugat.ai"})
        os.kill(os.getpid(), signal.SIGTERM)

    ScriptedClient.on_connect = on_connect
    monkeypatch.setattr(cli, "BridgeClient", ScriptedClient)
    result = runner.invoke(app, ["run", "--service"])
    assert result.exit_code == 0, result.output
    assert_caps_kept(caps, before)
    assert runstate.read_state(home)["handle"] == "saugat.ai"


async def test_caps_and_run_state_coexist_across_a_real_reconnect(home):
    """The reviewer's probe (usage 7 → 0 on reconnect while both used state.json), against
    the fake gateway: a connection, a drop, a reconnect."""
    import asyncio

    from yantraxa_bridge.client import BridgeClient

    from .fake_server import FakeServer
    from .test_client import FAST

    caps = used_caps(home)
    before = caps.path.read_bytes()
    writer = runstate.StateWriter(home, mode="terminal")
    writer.started()
    async with FakeServer() as server:
        client = BridgeClient(
            BridgeConfig(server_url=server.url, adapter="ollama"),
            Credentials(agent_id="ag_abc234", secret=SECRET),
            home=home,
            timing=FAST,
            on_connected=writer.connected,
            on_error=writer.error,
        )
        task = asyncio.create_task(client.run())
        await server.authed.wait()
        server.authed.clear()
        await server.ws.close(1011, "crash")
        await server.authed.wait()
        assert server.connections == 2
        await client.stop()
        await asyncio.wait_for(task, 2)
    assert_caps_kept(caps, before)
    state = runstate.read_state(home)
    assert state["handle"] == "saugat.ai"
    assert state["last_error"] == "connection closed (1011); reconnecting"


def test_run_service_records_a_bad_server_url(home, restore_logging, monkeypatch):
    """A plain-http remote server (no YANTRAXA_ALLOW_INSECURE_HTTP) is 'nothing to run':
    logged, recorded, exit 1; not a traceback the supervisor restarts silently."""
    monkeypatch.delenv("YANTRAXA_ALLOW_INSECURE_HTTP", raising=False)
    save_config(BridgeConfig(server_url="http://198.51.100.114:4000", adapter="interactive-mcp"))
    save_credentials(Credentials(agent_id="ag_abc234", secret=SECRET))
    result = runner.invoke(app, ["run", "--service"])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit), result.exception
    log = (home / "log" / "bridge.log").read_text()
    assert "stopped: server URL must use https:// unless it points to loopback" in log
    state = runstate.read_state(home)
    assert state["last_error"].startswith("server URL must use https://")
    assert state["stopped_at"]


def test_run_in_a_terminal_refuses_beside_the_running_service(home, service_backend):
    """A second bridge would replace the service (4009), which then stops for good."""
    paired(adapter="interactive-mcp")
    service_backend.is_installed = service_backend.running = True
    service_backend.home = home
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 1
    assert "The background service already runs this bridge (pid 4242)" in result.output
    assert cli._command("service", "uninstall") in result.output


def test_run_in_a_terminal_ignores_a_service_for_another_home(
    home, service_backend, monkeypatch, tmp_path
):
    from yantraxa_bridge.client import StopReason

    paired(adapter="interactive-mcp")
    service_backend.is_installed = service_backend.running = True
    service_backend.home = tmp_path / "other"

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self):
            return StopReason.STOPPED

        async def stop(self):
            pass

    monkeypatch.setattr(cli, "BridgeClient", Client)
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 0, result.output


def test_doctor_warns_not_to_start_a_second_bridge(home, service_backend, monkeypatch):
    async def checks():
        return []

    monkeypatch.setattr(cli, "run_diagnose", checks)
    paired(adapter="interactive-mcp")
    writer = runstate.StateWriter(home, mode="service")
    writer.started()
    writer.error("can't connect to wss://api.yantraxa.com/ws/agent: timed out")
    output = runner.invoke(app, ["doctor"]).output
    assert "Don't also start `run`: a second bridge replaces this one" in output
