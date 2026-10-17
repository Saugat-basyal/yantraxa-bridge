import httpx
import respx
from typer.testing import CliRunner

from yantraxa_bridge.cli import app
from yantraxa_bridge.config import BridgeConfig, save_config
from yantraxa_bridge.credentials import Credentials, load_credentials, save_credentials

runner = CliRunner()


def test_help_lists_exactly_the_five_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("pair", "run", "status", "diagnose", "revoke-local"):
        assert command in result.output


def test_pair_adapter_choices():
    result = runner.invoke(app, ["pair", "--server", "http://x:4000", "--adapter", "demo"])
    assert result.exit_code != 0


def test_pair_rejects_a_bad_server_url(home):
    result = runner.invoke(app, ["pair", "--server", "localhost:4000", "--adapter", "ollama"])
    assert result.exit_code == 1


def test_run_needs_pairing(home):
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 1
    assert "not paired" in result.output


def test_status_unpaired(home):
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "not paired" in result.output
    assert str(home) in result.output


@respx.mock
def test_status_paired_never_prints_the_secret(home):
    save_config(BridgeConfig(server_url="http://localhost:4000", adapter="ollama"))
    save_credentials(Credentials(agent_id="ag_abc234", secret="s3cret-value"))
    respx.get("http://localhost:4000/v1/agent/me").mock(
        return_value=httpx.Response(
            200, json={"agent": {"handle": "saugat.ai", "state": "active"}, "permissions": {}}
        )
    )
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "ag_abc234" in result.output and "@saugat.ai, active" in result.output
    assert "s3cret-value" not in result.output


def test_revoke_local(home):
    save_credentials(Credentials(agent_id="ag_abc234", secret="s3cret"))
    assert runner.invoke(app, ["revoke-local"], input="n\n").exit_code == 1
    assert load_credentials() is not None
    result = runner.invoke(app, ["revoke-local", "--yes"])
    assert result.exit_code == 0
    assert load_credentials() is None
    assert "No stored credential" in runner.invoke(app, ["revoke-local", "-y"]).output


def test_diagnose_exit_code(home):
    result = runner.invoke(app, ["diagnose"])
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_run_needs_a_model_for_ollama(home):
    save_config(BridgeConfig(server_url="http://localhost:4000", adapter="ollama"))
    save_credentials(Credentials(agent_id="ag_abc234", secret="s3cret-value"))
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 1
    assert "model" in result.output


def test_run_hands_the_adapter_to_the_client(home, monkeypatch):
    from yantraxa_bridge import cli
    from yantraxa_bridge.adapters.handler import AdapterTurnHandler
    from yantraxa_bridge.adapters.ollama import OllamaAdapter
    from yantraxa_bridge.client import StopReason

    seen = {}

    class FakeClient:
        def __init__(self, config, credentials, *, turn_handler, **hooks):
            seen["handler"] = turn_handler

        async def run(self):
            return StopReason.STOPPED

        async def stop(self):
            pass

    monkeypatch.setattr(cli, "BridgeClient", FakeClient)
    save_config(
        BridgeConfig(server_url="http://localhost:4000", adapter="ollama", model="llama3.1:8b")
    )
    save_credentials(Credentials(agent_id="ag_abc234", secret="s3cret-value"))
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 0, result.output
    assert isinstance(seen["handler"], AdapterTurnHandler)
    assert isinstance(seen["handler"].adapter, OllamaAdapter)
    assert seen["handler"].adapter.model == "llama3.1:8b"


def test_run_sweeps_old_turn_folders_and_stops_cleanly_on_sighup(
    home, monkeypatch, turn_dir_sweeps
):
    import asyncio
    import os
    import signal

    from yantraxa_bridge import cli
    from yantraxa_bridge.client import StopReason

    events = []

    class FakeClient:
        def __init__(self, config, credentials, *, turn_handler, **hooks):
            self.stopped = asyncio.Event()

        async def run(self):
            registered = set(asyncio.get_running_loop()._signal_handlers)
            events.append(registered)
            assert signal.SIGHUP in registered  # else the SIGHUP below would end pytest
            os.kill(os.getpid(), signal.SIGHUP)  # the terminal closed
            await asyncio.wait_for(self.stopped.wait(), 5)
            return StopReason.STOPPED

        async def stop(self):
            events.append("client.stop")
            self.stopped.set()

    monkeypatch.setattr(cli, "BridgeClient", FakeClient)
    monkeypatch.setattr(cli, "kill_active", lambda: events.append("kill_active") or 1)
    save_config(
        BridgeConfig(server_url="http://localhost:4000", adapter="ollama", model="llama3.1:8b")
    )
    save_credentials(Credentials(agent_id="ag_abc234", secret="s3cret-value"))
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 0, result.output
    assert {signal.SIGINT, signal.SIGTERM, signal.SIGHUP} <= events[0]
    assert events[1:] == ["kill_active", "client.stop"]  # the CLI dies first, then the client
    assert len(turn_dir_sweeps) == 1


def test_run_keeps_an_ignored_sighup_ignored_for_nohup(home, monkeypatch):
    import asyncio
    import signal

    from yantraxa_bridge import cli
    from yantraxa_bridge.client import StopReason

    seen = {}

    class FakeClient:
        def __init__(self, config, credentials, *, turn_handler, **hooks):
            pass

        async def run(self):
            seen["handled"] = set(asyncio.get_running_loop()._signal_handlers)
            seen["hup"] = signal.getsignal(signal.SIGHUP)
            return StopReason.STOPPED

        async def stop(self):
            pass

    monkeypatch.setattr(cli, "BridgeClient", FakeClient)
    save_config(
        BridgeConfig(server_url="http://localhost:4000", adapter="ollama", model="llama3.1:8b")
    )
    save_credentials(Credentials(agent_id="ag_abc234", secret="s3cret-value"))
    previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)  # what `nohup` does
    try:
        result = runner.invoke(app, ["run"])
        assert result.exit_code == 0, result.output
        assert signal.SIGHUP not in seen["handled"]
        assert seen["hup"] == signal.SIG_IGN
        assert {signal.SIGINT, signal.SIGTERM} <= seen["handled"]
    finally:
        signal.signal(signal.SIGHUP, previous)
    assert cli._stop_signals() == (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def test_version_prints_the_version_and_exits_zero():
    from yantraxa_bridge import __version__

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output == f"yantraxa-bridge {__version__}\n"
