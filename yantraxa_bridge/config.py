"""`$YANTRAXA_HOME/config.toml` (default `~/.yantraxa`) as a `BridgeConfig`."""

from __future__ import annotations

import json
import logging
import os
import socket
import tempfile
import tomllib
from ipaddress import ip_address
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .redaction import secret_filter

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
Adapter = Literal["ollama", "personal-agent", "interactive-mcp", "claude-code", "codex-cli"]
ADAPTERS: tuple[str, ...] = (
    "ollama",
    "personal-agent",
    "interactive-mcp",
    "claude-code",
    "codex-cli",
)
CLI_ADAPTERS: tuple[str, ...] = ("claude-code", "codex-cli")
# The command each CLI adapter runs; `pair` looks for them on the PATH.
CLI_COMMANDS: dict[str, str] = {"claude-code": "claude", "codex-cli": "codex"}
DEFAULT_SERVER = "https://api.yantraxa.com"
CODEX_HOME_DIR = "codex"


class ConfigError(Exception):
    """A missing, unreadable or invalid bridge configuration."""


log = logging.getLogger("yantraxa_bridge.config")

# Before the Yantraxa rename the bridge lived in `~/.connector`, set by `CONNECTOR_HOME`.
# Both are still honoured, with one warning per process, until the owner moves them.
LEGACY_HOME_ENV = "CONNECTOR_HOME"
LEGACY_HOME_DIR = ".connector"
_warned_legacy_home = False


def _warn_legacy_home(message: str) -> None:
    global _warned_legacy_home
    if not _warned_legacy_home:
        _warned_legacy_home = True
        log.warning(message)


def yantraxa_home() -> Path:
    """`YANTRAXA_HOME`, or `~/.yantraxa`. Tests and throwaway runs set the variable.

    Falls back to the pre-rename `CONNECTOR_HOME`, then to an existing `~/.connector`
    when `~/.yantraxa` does not exist yet."""
    override = os.environ.get("YANTRAXA_HOME")
    if override:
        return Path(override).expanduser()
    legacy_override = os.environ.get(LEGACY_HOME_ENV)
    if legacy_override:
        _warn_legacy_home(
            f"{LEGACY_HOME_ENV} is deprecated: set YANTRAXA_HOME={legacy_override} instead"
        )
        return Path(legacy_override).expanduser()
    home = Path.home() / ".yantraxa"
    legacy = Path.home() / LEGACY_HOME_DIR
    if not home.exists() and legacy.exists():
        _warn_legacy_home(
            f"using the old home {legacy}: move it with `mv ~/{LEGACY_HOME_DIR} ~/.yantraxa`"
        )
        return legacy
    return home


def ensure_home(home: Path | None = None) -> Path:
    home = home or yantraxa_home()
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(home, 0o700)
    return home


def ensure_private_dir(path: Path) -> Path:
    """`path`, created if missing with every new folder on the way 0700 (`mkdir(parents=True)`
    would give the parents the default mode)."""
    missing = []
    probe = path
    while not probe.exists():
        missing.append(probe)
        probe = probe.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700, exist_ok=True)
        os.chmod(directory, 0o700)
    return path


def _default_ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST


class BridgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_url: str | None = None
    adapter: Adapter = "ollama"
    ollama_host: str = Field(default_factory=_default_ollama_host)
    model: str | None = None
    personal_agent_url: str | None = None
    host_label: str = Field(default_factory=lambda: socket.gethostname()[:80], max_length=80)
    timeout_s: float = Field(default=40, gt=0)
    max_tokens: int = Field(default=400, gt=0)
    # Added to the defaults in redaction.py.
    redaction_patterns: list[str] = Field(default_factory=list)
    email_allowlist: list[str] = Field(default_factory=list)
    # Accept invitations that arrive `auto_accepted: false` over REST instead of
    # leaving them for the owner to handle in Yantraxa.
    auto_accept_pending: bool = False
    # Explicit escape hatch for a trusted private network. Without it, adapter prompts
    # may use cleartext HTTP only when the endpoint is on this machine.
    allow_insecure_adapter_http: bool = False
    # `claude-code` and `codex-cli` (spec §4, D6, D7). Claude uses the full model on
    # rotation turns and the fast one when someone @mentions or replies to the agent. Both
    # are sonnet: measured on 2026-10-14, haiku missed the structured answer on 25-35% of
    # addressed turns and was slower (median 3.9 s with the retry, sonnet 3.0 s).
    claude_model: str = Field(default="sonnet", min_length=1)
    claude_model_fast: str = Field(default="sonnet", min_length=1)
    # Unset: the Codex account's default model (a ChatGPT login refuses some models).
    codex_model: str | None = None
    # Codex's confinement was checked on 0.158.x only; any other version passes every turn
    # unless this is true. Set it only after checking that the new version adds no tools.
    codex_allow_untested: bool = False
    cli_effort: Literal["low", "medium", "high"] = "low"
    max_turns_per_meeting: int = Field(default=40, gt=0)
    max_turns_per_day: int = Field(default=200, gt=0)
    # The bridge's own Codex home, with its own sign-in; default `$YANTRAXA_HOME/codex`.
    codex_home: str | None = None
    # API-key mode (D4): the name of an environment variable holding the key, passed to
    # the CLI as ANTHROPIC_API_KEY or CODEX_API_KEY. Unset: the CLI's own sign-in.
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")


def codex_home_path(config: BridgeConfig, home: Path | None = None) -> Path:
    if config.codex_home:
        return Path(config.codex_home).expanduser()
    return (home or yantraxa_home()) / CODEX_HOME_DIR


def api_key(config: BridgeConfig) -> str | None:
    """The API key for API-key mode, or None for the CLI's own sign-in. The key is added to
    the log's secret filter, so no log line can show it."""
    if not config.api_key_env:
        return None
    value = os.environ.get(config.api_key_env)
    if not value:
        raise ConfigError(f"api_key_env is {config.api_key_env}, but that variable is not set")
    secret_filter.add_secret(value)
    return value


def config_path(home: Path | None = None) -> Path:
    return (home or yantraxa_home()) / "config.toml"


def load_config(home: Path | None = None) -> BridgeConfig:
    path = config_path(home)
    if not path.exists():
        return BridgeConfig()
    try:
        data = tomllib.loads(path.read_text())
        return BridgeConfig.model_validate(data)
    except (tomllib.TOMLDecodeError, OSError) as error:
        raise ConfigError(f"{path}: {error}") from error
    except ValidationError as error:
        fields = ", ".join(".".join(map(str, e["loc"])) for e in error.errors())
        raise ConfigError(f"{path}: invalid {fields}") from error


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # a JSON string is a valid TOML basic string
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"can't write {type(value).__name__} to TOML")


def save_config(config: BridgeConfig, home: Path | None = None) -> Path:
    home = ensure_home(home)
    path = config_path(home)
    lines = [
        f"{key} = {_toml_value(value)}"
        for key, value in config.model_dump().items()
        if value is not None
    ]
    write_private(path, "# Yantraxa Bridge configuration\n" + "\n".join(lines) + "\n")
    return path


def write_private(path: Path, text: str) -> None:
    """Write `text` to `path` with mode 0600, atomically: through a new, unpredictably named
    file in the same directory (`mkstemp`: O_EXCL, 0600), then `os.replace`."""
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _normalise_http_url(
    value: str, *, label: str, allow_insecure_http: bool = False, insecure_hint: str = ""
) -> str:
    try:
        parts = urlsplit(value.strip())
        hostname = parts.hostname
        _ = parts.port  # Validate a supplied port now, not on the first request.
    except ValueError as error:
        raise ConfigError(f"invalid {label} URL: {error}") from error
    if parts.scheme not in ("http", "https") or not parts.netloc or not hostname:
        raise ConfigError(f"{label} URL must start with http:// or https://, got {value!r}")
    if parts.username is not None or parts.password is not None:
        raise ConfigError(f"{label} URL must not contain credentials")
    if parts.scheme == "http" and not allow_insecure_http:
        is_loopback = hostname == "localhost"
        if not is_loopback and "%" not in hostname:
            try:
                # 0.0.0.0 (Ollama's "listen everywhere" value) connects to this machine too.
                address = ip_address(hostname)
                is_loopback = address.is_loopback or address.is_unspecified
            except ValueError:
                is_loopback = False
        if not is_loopback:
            raise ConfigError(
                f"{label} URL must use https:// unless it points to loopback{insecure_hint}"
            )
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


# A development server on a trusted LAN (the Mac running `npm run dev`) is plain HTTP.
# Reaching it from another machine needs this set to 1; the credential then crosses the
# LAN in cleartext, so never set it for a server reached over the internet.
INSECURE_SERVER_ENV = "YANTRAXA_ALLOW_INSECURE_HTTP"


def normalise_server_url(server_url: str) -> str:
    """A credential-safe server URL without a trailing slash."""
    return _normalise_http_url(
        server_url,
        label="server",
        allow_insecure_http=os.environ.get(INSECURE_SERVER_ENV) == "1",
        insecure_hint=f" (set {INSECURE_SERVER_ENV}=1 for a development server on a trusted LAN)",
    )


def normalise_adapter_url(adapter_url: str, *, allow_insecure_http: bool = False) -> str:
    """A prompt endpoint URL; remote cleartext HTTP requires an explicit opt-in."""
    return _normalise_http_url(
        adapter_url,
        label="adapter",
        allow_insecure_http=allow_insecure_http,
        insecure_hint=" (or set allow_insecure_adapter_http = true for a trusted LAN)",
    )


def ws_url(server_url: str) -> str:
    """The `/ws/agent` URL for a server: `http` becomes `ws`, `https` becomes `wss`."""
    parts = urlsplit(normalise_server_url(server_url))
    scheme = "wss" if parts.scheme == "https" else "ws"
    return urlunsplit((scheme, parts.netloc, parts.path + "/ws/agent", "", ""))
