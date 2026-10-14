"""The `yantraxa-bridge` command: pair, run, service, status, diagnose, doctor, log,
revoke-local."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import signal
import socket
import sys
from enum import StrEnum
from logging.handlers import RotatingFileHandler
from pathlib import Path

import httpx
import typer
from pydantic import ValidationError

from . import __version__
from . import service as svc
from .adapters import cli_probe, make_cli_adapter, make_turn_handler
from .adapters.cli_runner import PreflightError, kill_active, sweep_stale_turn_dirs
from .client import BridgeClient, StopReason
from .config import (
    CLI_ADAPTERS,
    CLI_COMMANDS,
    DEFAULT_SERVER,
    BridgeConfig,
    ConfigError,
    ensure_private_dir,
    load_config,
    normalise_server_url,
    save_config,
    yantraxa_home,
)
from .credentials import credentials_path, delete_credentials, load_credentials
from .diagnose import Check
from .diagnose import diagnose as run_diagnose
from .pairing import PairingError
from .pairing import pair as run_pair
from .redaction import secret_filter
from .turnlog import follow, format_event, printable, read_events

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Connects a locally hosted AI to Yantraxa meetings. "
    "Files live in $YANTRAXA_HOME (default ~/.yantraxa).",
)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"yantraxa-bridge {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_print_version,
        is_eager=True,
        help="Print the version and exit.",
    ),
) -> None:
    """Connects a locally hosted AI to Yantraxa meetings."""


class PairAdapter(StrEnum):
    ollama = "ollama"
    personal_agent = "personal-agent"
    interactive_mcp = "interactive-mcp"
    claude_code = "claude-code"
    codex_cli = "codex-cli"


ADAPTER_NAMES = {PairAdapter.claude_code: "Claude Code", PairAdapter.codex_cli: "Codex"}
SERVICE_LOG_BYTES = 5 * 1024 * 1024
SERVICE_LOG_BACKUPS = 3


class _PrivateRotatingFileHandler(RotatingFileHandler):
    """`bridge.log` and its rotations, 0600."""

    def _open(self):
        stream = super()._open()
        os.chmod(self.baseFilename, 0o600)
        return stream


def _setup_logging(level: str, log_file: Path | None = None) -> None:
    handler: logging.Handler
    if log_file is None:
        handler = logging.StreamHandler()
    else:
        handler = _PrivateRotatingFileHandler(
            log_file, maxBytes=SERVICE_LOG_BYTES, backupCount=SERVICE_LOG_BACKUPS
        )
    handler.addFilter(secret_filter)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.WARNING)
    logging.getLogger("yantraxa_bridge").setLevel(level.upper())


def _fail(message: str) -> typer.Exit:
    typer.secho(message, fg=typer.colors.RED, err=True)
    return typer.Exit(1)


def _interactive() -> bool:
    """Whether `pair` may ask: stdin is a terminal (not `curl … | sh`)."""
    return sys.stdin.isatty()


def _choose_adapter(*, ask: bool) -> PairAdapter:
    """With no `--adapter`: whichever of `claude` and `codex` is on the PATH, asking when
    both are."""
    found = [a for a in ADAPTER_NAMES if shutil.which(CLI_COMMANDS[a.value])]
    if not found:
        raise ConfigError(
            "neither `claude` nor `codex` is on the PATH: install and sign in to one, or "
            "pass --adapter ollama, personal-agent or interactive-mcp"
        )
    if len(found) == 1:
        typer.echo(f"Using {ADAPTER_NAMES[found[0]]} (`{CLI_COMMANDS[found[0].value]}`).")
        return found[0]
    if not ask:
        raise ConfigError(
            "both `claude` and `codex` are installed: pass --adapter claude-code or "
            "--adapter codex-cli"
        )
    typer.echo("Both Claude Code and Codex are installed. Which should answer for this agent?")
    typer.echo("  1) Claude Code\n  2) Codex")
    while (choice := typer.prompt("Choose 1 or 2", default="1").strip()) not in ("1", "2"):
        pass
    return found[int(choice) - 1]


@app.command()
def pair(
    server: str = typer.Option(DEFAULT_SERVER, "--server", help="Yantraxa server URL."),
    adapter: PairAdapter | None = typer.Option(
        None,
        "--adapter",
        help="How this bridge reaches the AI. Default: Claude Code or Codex, whichever is "
        "installed (asks if both are).",
    ),
    host_label: str = typer.Option(
        socket.gethostname()[:80], "--host-label", help="Shown to the owner when approving."
    ),
    api_key_env: str | None = typer.Option(
        None,
        "--api-key-env",
        help="claude-code/codex-cli only: use the API key in this environment variable "
        "instead of the CLI's own sign-in.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Ask nothing: start the bridge automatically when you log in (unless --no-service).",
    ),
    no_service: bool = typer.Option(
        False, "--no-service", help="Don't set up the background service."
    ),
) -> None:
    """Pair this machine with an agent: prints a code to enter in Yantraxa."""
    ask = not yes and _interactive()
    try:
        normalise_server_url(server)
        if adapter is None:
            adapter = _choose_adapter(ask=ask)
        if adapter.value in CLI_ADAPTERS:
            config = load_config().model_copy(
                update={"adapter": adapter.value, "api_key_env": api_key_env}
            )
            BridgeConfig.model_validate(config.model_dump())
            typer.echo(asyncio.run(_preflight(config, allow_login=True)))
        elif api_key_env:
            raise ConfigError("--api-key-env is only for claude-code and codex-cli")
        asyncio.run(run_pair(server, adapter.value, host_label[:80]))
        if adapter.value in CLI_ADAPTERS:
            save_config(load_config().model_copy(update={"api_key_env": api_key_env}))
    except (PairingError, ConfigError, PreflightError) as error:
        raise _fail(str(error)) from None
    except ValidationError as error:
        raise _fail(f"invalid option: {error.errors()[0]['msg']}") from None
    except (KeyboardInterrupt, typer.Abort):
        raise _fail("pairing cancelled") from None
    _after_pair(yes=yes, no_service=no_service, ask=ask)


def _after_pair(*, yes: bool, no_service: bool, ask: bool) -> None:
    """Offer the background service, or say how to start the bridge."""
    if no_service:
        installed = _installed_service()
        if installed:
            typer.echo(
                "A background bridge is installed; restart it with the new pairing: "
                "yantraxa-bridge service install"
            )
        else:
            typer.echo("Start the bridge with: yantraxa-bridge run")
        return
    if not yes:
        if not ask:
            typer.echo(
                "Start the bridge with: yantraxa-bridge run\n"
                "Or keep it running in the background: yantraxa-bridge service install"
            )
            return
        try:
            wanted = typer.confirm("Start automatically when you log in?", default=True)
        except typer.Abort:
            wanted = False
        if not wanted:
            typer.echo("Start the bridge with: yantraxa-bridge run")
            return
    try:
        _install_service()
    except (svc.ServiceError, ConfigError, OSError) as error:
        typer.secho(f"Couldn't set up the background service: {error}", fg="red", err=True)
        typer.echo("Start the bridge with: yantraxa-bridge run")


async def _preflight(config: BridgeConfig, *, allow_login: bool) -> str:
    """The CLI adapter's checks and one confined probe turn; raises PreflightError."""
    from .adapters.claude_code import ClaudeCodeAdapter
    from .adapters.claude_code import preflight as claude_preflight
    from .adapters.codex_cli import preflight as codex_preflight

    adapter = make_cli_adapter(config, caps=False)
    if isinstance(adapter, ClaudeCodeAdapter):
        return await claude_preflight(adapter)
    return await codex_preflight(adapter, allow_login=allow_login)


# SIGHUP is what a closed terminal sends; without a handler it ends the bridge at once and
# strands a running CLI and its turn folder.
_STOP_SIGNALS = tuple(
    sig for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None)) if sig
)
_STOPPING: set[asyncio.Future] = set()


def _stop_signals() -> tuple[signal.Signals, ...]:
    """The signals `run` handles. SIGHUP is left alone when it is ignored already, as under
    `nohup yantraxa-bridge run &`: handling it would end the bridge when the terminal closes."""
    hup = getattr(signal, "SIGHUP", None)
    return tuple(
        sig for sig in _STOP_SIGNALS if not (sig == hup and signal.getsignal(sig) == signal.SIG_IGN)
    )


def _on_stop_signal(sig: signal.Signals, client: BridgeClient, log: logging.Logger) -> None:
    """Kills any running CLI turn with its process group and deletes its folders straight
    away, then stops the client."""
    if stopped := kill_active():
        log.info("%s: stopped %d running CLI turn(s)", signal.Signals(sig).name, stopped)
    task = asyncio.ensure_future(client.stop())
    _STOPPING.add(task)
    task.add_done_callback(_STOPPING.discard)


@app.command()
def run(
    log_level: str = typer.Option(
        os.environ.get("YANTRAXA_LOG_LEVEL", "info"), "--log-level", help="debug, info, warning"
    ),
    service: bool = typer.Option(
        False,
        "--service",
        hidden=True,
        help="Run as the background service: log to $YANTRAXA_HOME/log/bridge.log, and "
        "exit 0 when restarting can't help, so only a crash is restarted.",
    ),
) -> None:
    """Connect to Yantraxa and stay connected."""
    home = yantraxa_home()
    if service:
        svc.trim_service_out(home)
        log_dir = ensure_private_dir(home / svc.LOG_DIR)
        _setup_logging(log_level, log_dir / svc.BRIDGE_LOG)
    else:
        _setup_logging(log_level)
    log = logging.getLogger("yantraxa_bridge.cli")

    def stop(message: str) -> typer.Exit:
        """A stop that restarting can't fix: exit 1 in a terminal, 0 under the service."""
        if service:
            log.error("stopped: %s", message)
            typer.echo(message, err=True)
            return typer.Exit(0)
        return _fail(message)

    try:
        config = load_config()
        credentials = load_credentials()
    except ConfigError as error:
        raise stop(str(error)) from None
    if credentials is None or not config.server_url:
        raise stop("not paired yet: run `yantraxa-bridge pair`")
    secret_filter.add_secret(credentials.secret_value())
    try:
        turns = make_turn_handler(config)
    except ConfigError as error:
        raise stop(str(error)) from None
    if service:
        log.info("starting as the background service (yantraxa-bridge %s)", __version__)
    if swept := sweep_stale_turn_dirs():
        log.info("removed %d turn folder(s) left by an earlier bridge", len(swept))

    async def main() -> StopReason:
        client = BridgeClient(config, credentials, turn_handler=turns)
        loop = asyncio.get_running_loop()
        for sig in _stop_signals():
            with contextlib.suppress(NotImplementedError):  # Windows
                loop.add_signal_handler(sig, _on_stop_signal, sig, client, log)
        try:
            return await client.run()
        finally:
            aclose = getattr(turns, "aclose", None)
            if aclose is not None:
                await aclose()

    reason = asyncio.run(main())
    if reason is StopReason.DISCONNECTED:
        if service:
            log.info("stopped: the owner disconnected this agent in Yantraxa")
        typer.echo("The owner disconnected this agent in Yantraxa. Stopped.")
    elif reason is StopReason.SUPERSEDED:
        raise stop("Another bridge connected as this agent, so this one stopped.")
    elif reason is StopReason.UNAUTHORIZED and service:
        raise stop("the server refused this bridge's credential; run `yantraxa-bridge diagnose`")
    elif reason is StopReason.UNAUTHORIZED:
        typer.secho(
            "The server refused this bridge's credential. Diagnosing:",
            fg=typer.colors.RED,
            err=True,
        )
        _print_checks(asyncio.run(run_diagnose()))
        raise typer.Exit(1)


@app.command()
def status() -> None:
    """Show what this bridge is paired with."""
    try:
        config = load_config()
        credentials = load_credentials()
    except ConfigError as error:
        raise _fail(str(error)) from None
    typer.echo(f"Bridge:      yantraxa-bridge {__version__}")
    typer.echo(f"Home:        {yantraxa_home()}")
    typer.echo(f"Server:      {config.server_url or 'not set'}")
    typer.echo(f"Adapter:     {config.adapter}")
    typer.echo(f"Service:     {_service_summary()}")
    if credentials is None:
        typer.echo("Agent:       not paired")
        return
    typer.echo(f"Agent:       {credentials.agent_id}")
    if not config.server_url:
        return
    try:
        response = httpx.get(
            f"{normalise_server_url(config.server_url)}/v1/agent/me",
            headers={"Authorization": credentials.bearer()},
            timeout=10,
        )
    except httpx.HTTPError as error:
        typer.echo(f"Server view: unreachable ({error})")
        return
    if response.status_code == 200:
        agent = response.json().get("agent", {})
        typer.echo(f"Server view: @{agent.get('handle')}, {agent.get('state')}")
    else:
        typer.echo(f"Server view: HTTP {response.status_code} (run `yantraxa-bridge diagnose`)")


def _print_checks(checks) -> bool:
    marks = {
        True: ("ok", typer.colors.GREEN),
        False: ("FAIL", typer.colors.RED),
        None: ("warn", typer.colors.YELLOW),
    }
    for check in checks:
        label, colour = marks[check.ok]
        typer.secho(f"[{label:>4}] {check.name:<12} {check.detail}", fg=colour)
    return all(check.ok is not False for check in checks)


@app.command()
def diagnose() -> None:
    """Check config, credentials, server reachability, clock skew and the adapter."""
    if not _print_checks(asyncio.run(run_diagnose())):
        raise typer.Exit(1)


@app.command()
def doctor() -> None:
    """`diagnose`, plus a confined probe turn through Claude Code or Codex."""
    checks = asyncio.run(run_diagnose())
    try:
        config = load_config()
    except ConfigError:
        config = None
    if config is not None and config.adapter in CLI_ADAPTERS:
        adapter_ok = next((c.ok for c in checks if c.name == "adapter"), None)
        if adapter_ok is False:
            checks.append(Check("probe", None, "skipped: fix the adapter first"))
        else:
            try:
                checks.append(Check("probe", True, asyncio.run(cli_probe(config))))
            except (PreflightError, ConfigError) as error:
                checks.append(Check("probe", False, str(error)))
    if not _print_checks(checks):
        raise typer.Exit(1)


service_app = typer.Typer(
    no_args_is_help=True,
    help="Keep the bridge running in the background, started when you log in.",
)
app.add_typer(service_app, name="service")


def _installed_service() -> bool:
    try:
        return svc.current_backend().status().installed
    except svc.ServiceError:
        return False


def _service_summary() -> str:
    try:
        status = svc.current_backend().status()
    except svc.ServiceError:
        return "not available on this system"
    if not status.installed:
        return "not installed (`yantraxa-bridge service install`)"
    return "installed, " + ("running" if status.running else "not running")


def _install_service() -> None:
    """Install (or reinstall, which restarts) the background service; raises ServiceError."""
    config = load_config()
    if load_credentials() is None or not config.server_url:
        raise svc.ServiceError("not paired yet: run `yantraxa-bridge pair` first")
    backend = svc.current_backend()
    launch = svc.make_launch()
    if config.adapter in CLI_ADAPTERS:
        command = CLI_COMMANDS[config.adapter]
        if shutil.which(command, path=launch.path_env) is None:
            typer.secho(
                f"Warning: `{command}` isn't on this PATH, so the service won't find it either.",
                fg="yellow",
                err=True,
            )
    for line in backend.install(launch):
        typer.echo(line)
    typer.echo(f"The bridge now runs in the background ({backend.kind}).")
    typer.echo("It starts when you log in and restarts if it crashes.")
    typer.echo(f"  Runs: {' '.join(launch.program)}")
    typer.echo(f"  Logs: {launch.log_dir / svc.BRIDGE_LOG}")
    if config.api_key_env:
        typer.echo(_api_key_hint(config.api_key_env))
    typer.echo(
        "Check it with `yantraxa-bridge service status`; remove it with "
        "`yantraxa-bridge service uninstall`."
    )


def _api_key_hint(variable: str) -> str:
    """How to give the service the API key: it never goes in the unit file."""
    if sys.platform == "darwin":
        how = (
            f'launchctl setenv {variable} "${variable}", then `yantraxa-bridge service install` '
            "again; it lasts until you log out"
        )
    elif os.name == "nt":
        how = f'setx {variable} "<key>", then `yantraxa-bridge service install` in a new terminal'
    else:
        how = (
            "`systemctl --user edit yantraxa-bridge`, add the two lines "
            f"[Service] and Environment={variable}=<key>, then `chmod 600` the override file"
        )
    return f"API-key mode: the service needs {variable} in its own environment: {how}."


@service_app.command("install")
def service_install() -> None:
    """Run the bridge in the background now and whenever you log in. Needs no admin."""
    try:
        _install_service()
    except (svc.ServiceError, ConfigError, OSError) as error:
        raise _fail(str(error)) from None


@service_app.command("uninstall")
def service_uninstall() -> None:
    """Stop the background bridge and remove it. Your pairing stays."""
    try:
        lines = svc.current_backend().uninstall()
    except (svc.ServiceError, OSError) as error:
        raise _fail(str(error)) from None
    for line in lines:
        typer.echo(line)
    typer.echo("Removed the background service." if lines else "No background service installed.")


@service_app.command("status")
def service_status() -> None:
    """Whether the background bridge is installed and running, and when it last took a turn."""
    try:
        status = svc.current_backend().status()
    except svc.ServiceError as error:
        raise _fail(str(error)) from None
    home = yantraxa_home()
    typer.echo(f"Service:     {status.kind}")
    typer.echo(f"Installed:   {'yes, ' + status.where if status.installed else 'no'}")
    running = "yes" + (f" (pid {status.pid})" if status.pid else "") if status.running else "no"
    typer.echo(f"Running:     {running}")
    if status.detail:
        typer.echo(f"State:       {status.detail}")
    typer.echo(f"Last turn:   {svc.last_turn(home) or 'none logged yet'}")
    if status.installed and not status.running:
        line = svc.last_log_line(home)
        if line:
            typer.echo(f"Last log:    {line}")
    if not status.installed:
        typer.echo("Install it with: yantraxa-bridge service install")


@service_app.command("supervise", hidden=True)
def service_supervise(
    home: Path = typer.Option(..., "--home", help="The YANTRAXA_HOME to run."),
) -> None:
    """Windows: what the sign-in entry starts. Runs the bridge, restarting it after a crash."""
    raise typer.Exit(svc.supervise(home))


@app.command("log")
def show_log(
    lines: int = typer.Option(20, "--lines", "-n", min=1, help="How many recent events."),
    tail: bool = typer.Option(False, "--tail", "-f", help="Keep printing new events."),
    raw: bool = typer.Option(False, "--json", help="Print the JSON lines as stored."),
) -> None:
    """Show the transparency log: what the bridge sent and received for each turn."""
    home = yantraxa_home()

    def show(event: dict) -> None:
        typer.echo(printable(json.dumps(event, ensure_ascii=False)) if raw else format_event(event))

    events = read_events(home, lines)
    if not events and not tail:
        typer.echo(f"No turns logged yet ({home / 'log' / 'turns.jsonl'}).")
        return
    for event in events:
        show(event)
    if tail:
        with contextlib.suppress(KeyboardInterrupt):
            for event in follow(home):
                show(event)


@app.command("revoke-local")
def revoke_local(
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation."),
) -> None:
    """Delete this machine's stored credential. Revoke the agent in Yantraxa too."""
    path = credentials_path()
    if not path.exists():
        typer.echo("No stored credential.")
        return
    if not yes:
        typer.confirm(f"Delete {path}?", abort=True)
    delete_credentials()
    typer.echo(
        f"Deleted {path}. To cut the agent off on the server too, revoke or "
        "disconnect it in Yantraxa."
    )


if __name__ == "__main__":
    app()
