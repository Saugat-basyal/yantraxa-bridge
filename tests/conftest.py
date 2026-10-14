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

    def install(self, launch):
        self.installed.append(launch)
        self.is_installed = self.running = True
        return ["Installed the fake service"]

    def uninstall(self):
        self.uninstalls += 1
        was = self.is_installed
        self.is_installed = self.running = False
        return ["Removed the fake service"] if was else []

    def status(self):
        from yantraxa_bridge.service import Status

        return Status(
            self.is_installed, self.running, self.kind, "fake", pid=4242 if self.running else None
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
