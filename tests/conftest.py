import os

import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway YANTRAXA_HOME, so no test touches the real ~/.yantraxa."""
    path = tmp_path / "yantraxa-home"
    monkeypatch.setenv("YANTRAXA_HOME", str(path))
    monkeypatch.delenv("CONNECTOR_HOME", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    return path


class FakeBackend:
    """Stands in for launchd, systemd and the Windows Run key in every test."""

    kind = "fake service"

    def __init__(self) -> None:
        self.installed: list = []  # the Launch of each install
        self.uninstalls = 0
        self.is_installed = False
        self.running = False
        self.home = None  # the YANTRAXA_HOME of the installed service
        self.last_exit = None
        # Called with the Launch on install: stands in for the service's own `run --service`,
        # e.g. writing run-state.json as a connection would.
        self.on_install = None

    def install(self, launch):
        self.installed.append(launch)
        self.is_installed = self.running = True
        self.home = launch.home
        if self.on_install is not None:
            self.on_install(launch)
        return ["Installed the fake service"]

    def uninstall(self):
        self.uninstalls += 1
        was = self.is_installed
        self.is_installed = self.running = False
        return ["Removed the fake service"] if was else []

    def status(self):
        from yantraxa_bridge.service import Status

        return Status(
            self.is_installed,
            self.running,
            self.kind,
            "fake",
            pid=4242 if self.running else None,
            home=self.home,
            last_exit=self.last_exit,
        )


@pytest.fixture(autouse=True)
def turn_dir_sweeps(monkeypatch):
    """`run` never sweeps the real temp folder in a test; the sweeps it asked for are here."""
    calls: list = []
    monkeypatch.setattr(
        "yantraxa_bridge.cli.sweep_stale_turn_dirs", lambda *a, **k: calls.append((a, k)) or []
    )
    return calls


@pytest.fixture(autouse=True)
def service_backend(monkeypatch):
    """No test ever installs a real service: `current_backend()` is always this fake."""
    backend = FakeBackend()
    monkeypatch.setattr("yantraxa_bridge.service.current_backend", lambda: backend)
    return backend


@pytest.fixture(autouse=True)
def bridge_python(monkeypatch):
    """`BRIDGE_TEST_PYTHON=/home/runner/.venv/bin/python pytest` runs every test as the release
    runner does: the bridge as `<that python> -m yantraxa_bridge.cli`, from a path with no space
    to quote (this checkout's path has one, so printed commands are quoted here and not there)."""
    python = os.environ.get("BRIDGE_TEST_PYTHON")
    if not python:
        return
    from yantraxa_bridge import service as svc

    real = svc.bridge_command

    def bridge_command(argv0=None):
        command = real(argv0)
        if argv0 is None and command[1:] == ["-m", "yantraxa_bridge.cli"]:
            return [python, *command[1:]]
        return command

    monkeypatch.setattr(svc, "bridge_command", bridge_command)


@pytest.fixture(autouse=True)
def short_connect_wait(monkeypatch):
    """`pair` and `service install` wait for the service's first connection; not 20 s here."""
    monkeypatch.setattr("yantraxa_bridge.cli.CONNECT_WAIT_S", 0.3)
    monkeypatch.setattr("yantraxa_bridge.cli.CONNECT_POLL_S", 0.02)
