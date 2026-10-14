import logging
import os
import stat
from pathlib import Path

import pytest

from yantraxa_bridge import config
from yantraxa_bridge.config import (
    BridgeConfig,
    ConfigError,
    load_config,
    normalise_adapter_url,
    save_config,
    ws_url,
    yantraxa_home,
)
from yantraxa_bridge.credentials import (
    Credentials,
    delete_credentials,
    load_credentials,
    save_credentials,
)


def test_home_comes_from_yantraxa_home(home):
    assert yantraxa_home() == home


@pytest.fixture
def no_home_env(monkeypatch, tmp_path):
    """No home variables, `$HOME` in a temp dir, and the once-per-process warning re-armed."""
    monkeypatch.delenv("YANTRAXA_HOME", raising=False)
    monkeypatch.delenv("CONNECTOR_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(config, "_warned_legacy_home", False)
    return tmp_path


def test_home_defaults_to_dot_yantraxa(no_home_env, caplog):
    with caplog.at_level(logging.WARNING, logger="yantraxa_bridge"):
        assert yantraxa_home() == no_home_env / ".yantraxa"
    assert caplog.records == []


def test_home_falls_back_to_an_existing_dot_connector(no_home_env, caplog):
    (no_home_env / ".connector").mkdir()
    with caplog.at_level(logging.WARNING, logger="yantraxa_bridge"):
        assert yantraxa_home() == no_home_env / ".connector"
        assert yantraxa_home() == no_home_env / ".connector"
    assert len(caplog.records) == 1
    assert "mv ~/.connector ~/.yantraxa" in caplog.records[0].getMessage()


def test_dot_yantraxa_wins_over_dot_connector(no_home_env, caplog):
    (no_home_env / ".connector").mkdir()
    (no_home_env / ".yantraxa").mkdir()
    with caplog.at_level(logging.WARNING, logger="yantraxa_bridge"):
        assert yantraxa_home() == no_home_env / ".yantraxa"
    assert caplog.records == []


def test_legacy_connector_home_variable_is_honoured(no_home_env, monkeypatch, caplog):
    legacy = no_home_env / "old-home"
    monkeypatch.setenv("CONNECTOR_HOME", str(legacy))
    with caplog.at_level(logging.WARNING, logger="yantraxa_bridge"):
        assert yantraxa_home() == legacy
        assert yantraxa_home() == legacy
    assert len(caplog.records) == 1
    assert "YANTRAXA_HOME" in caplog.records[0].getMessage()


def test_yantraxa_home_variable_wins_over_the_legacy_one(no_home_env, monkeypatch, caplog):
    monkeypatch.setenv("CONNECTOR_HOME", str(no_home_env / "old-home"))
    monkeypatch.setenv("YANTRAXA_HOME", str(no_home_env / "new-home"))
    with caplog.at_level(logging.WARNING, logger="yantraxa_bridge"):
        assert yantraxa_home() == no_home_env / "new-home"
    assert caplog.records == []


@pytest.mark.parametrize(
    ("server", "expected"),
    [
        ("http://localhost:4000", "ws://localhost:4000/ws/agent"),
        ("http://127.0.0.1:4000/", "ws://127.0.0.1:4000/ws/agent"),
        ("http://[::1]:4000/", "ws://[::1]:4000/ws/agent"),
        ("https://yantraxa.example.com", "wss://yantraxa.example.com/ws/agent"),
        ("https://example.com/api/", "wss://example.com/api/ws/agent"),
    ],
)
def test_ws_url_follows_the_server_scheme(server, expected):
    assert ws_url(server) == expected


@pytest.mark.parametrize(
    "server",
    [
        "ftp://x",
        "localhost:4000",
        "",
        "ws://x",
        "http://198.51.100.114:4000",
        "http://api.example.com",
        "https://user:password@api.example.com",
    ],
)
def test_ws_url_rejects_other_schemes(server, monkeypatch):
    monkeypatch.delenv("YANTRAXA_ALLOW_INSECURE_HTTP", raising=False)
    with pytest.raises(ConfigError):
        ws_url(server)


def test_a_lan_development_server_needs_an_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("YANTRAXA_ALLOW_INSECURE_HTTP", raising=False)
    with pytest.raises(ConfigError, match="YANTRAXA_ALLOW_INSECURE_HTTP=1"):
        ws_url("http://198.51.100.114:4000")
    monkeypatch.setenv("YANTRAXA_ALLOW_INSECURE_HTTP", "1")
    assert ws_url("http://198.51.100.114:4000") == "ws://198.51.100.114:4000/ws/agent"
    # The opt-in relaxes only the scheme rule.
    with pytest.raises(ConfigError):
        ws_url("http://user:password@198.51.100.114:4000")


def test_adapter_urls_require_https_off_loopback_unless_explicitly_allowed():
    assert normalise_adapter_url("http://127.0.0.1:11434/") == "http://127.0.0.1:11434"
    assert normalise_adapter_url("http://0.0.0.0:11434") == "http://0.0.0.0:11434"
    assert normalise_adapter_url("https://agent.example.com/") == "https://agent.example.com"
    with pytest.raises(ConfigError):
        normalise_adapter_url("http://198.51.100.109:11434")
    assert (
        normalise_adapter_url("http://198.51.100.109:11434", allow_insecure_http=True)
        == "http://198.51.100.109:11434"
    )


@pytest.mark.parametrize("server", ["http://:4000", "https://example.com:not-a-port"])
def test_urls_reject_missing_hosts_and_invalid_ports(server):
    with pytest.raises(ConfigError):
        ws_url(server)


def test_defaults_when_no_file(home):
    config = load_config()
    assert config.server_url is None
    assert config.ollama_host == "http://127.0.0.1:11434"
    assert config.timeout_s == 40
    assert config.max_tokens == 400
    assert config.auto_accept_pending is False


def test_ollama_host_env_is_the_default(home, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://198.51.100.109:11434")
    assert load_config().ollama_host == "http://198.51.100.109:11434"


def test_round_trip(home):
    config = BridgeConfig(
        server_url="http://localhost:4000",
        adapter="personal-agent",
        host_label='lab "box"',
        model="llama3.1:8b",
        redaction_patterns=[r"proj-\d+"],
        auto_accept_pending=True,
    )
    save_config(config)
    assert load_config() == config
    assert stat.S_IMODE(os.stat(home / "config.toml").st_mode) == 0o600


def test_bad_config_file_is_a_config_error(home):
    home.mkdir(parents=True)
    (home / "config.toml").write_text("adapter = 'telepathy'\n")
    with pytest.raises(ConfigError):
        load_config()
    (home / "config.toml").write_text("this is = = not toml")
    with pytest.raises(ConfigError):
        load_config()


def test_credentials_round_trip_with_0600(home):
    assert load_credentials() is None
    save_credentials(Credentials(agent_id="ag_abc234", secret="s3cret"))
    path = home / "credentials"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(home).st_mode) == 0o700
    loaded = load_credentials()
    assert loaded is not None
    assert loaded.agent_id == "ag_abc234"
    assert loaded.secret_value() == "s3cret"


def test_loose_credentials_are_tightened(home):
    save_credentials(Credentials(agent_id="ag_abc234", secret="s3cret"))
    os.chmod(home / "credentials", 0o644)
    assert load_credentials() is not None
    assert stat.S_IMODE(os.stat(home / "credentials").st_mode) == 0o600


def test_delete_credentials(home):
    save_credentials(Credentials(agent_id="ag_abc234", secret="s3cret"))
    assert delete_credentials() is True
    assert load_credentials() is None
    assert delete_credentials() is False


def test_write_private_uses_a_fresh_unpredictable_temp_file(home, monkeypatch):
    import tempfile

    from yantraxa_bridge.config import write_private

    home.mkdir(parents=True)
    target = home / "caps.json"
    # A planted `.tmp` (the old fixed name) is neither followed nor truncated.
    planted = home / "caps.json.tmp"
    victim = home / "victim.txt"
    victim.write_text("keep me")
    planted.symlink_to(victim)
    made = []
    real_mkstemp = tempfile.mkstemp

    def spy(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        made.append(name)
        return fd, name

    monkeypatch.setattr(tempfile, "mkstemp", spy)
    write_private(target, "one")
    write_private(target, "two")
    assert target.read_text() == "two" and oct(target.stat().st_mode & 0o777) == "0o600"
    assert victim.read_text() == "keep me"
    assert len(made) == 2 and made[0] != made[1]
    assert all(Path(name).parent == home and not Path(name).exists() for name in made)
    assert sorted(p.name for p in home.iterdir()) == ["caps.json", "caps.json.tmp", "victim.txt"]


def test_write_private_leaves_no_temp_file_when_it_fails(home, monkeypatch):
    import os as os_module

    from yantraxa_bridge.config import write_private

    def broken_replace(src, dst):
        raise OSError("disk full")

    home.mkdir(parents=True)
    monkeypatch.setattr(os_module, "replace", broken_replace)
    with pytest.raises(OSError):
        write_private(home / "config.toml", "x")
    assert list(home.iterdir()) == []
