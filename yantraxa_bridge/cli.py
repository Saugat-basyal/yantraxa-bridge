"""The `yantraxa-bridge` command: pair, run, service, status, diagnose, doctor, log,
revoke-local."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from enum import StrEnum
from logging.handlers import RotatingFileHandler
from pathlib import Path

import httpx
import typer
from pydantic import ValidationError

from . import __version__, runstate
from . import service as svc
from .adapters import cli_probe, make_cli_adapter, make_turn_handler
from .adapters.cli_runner import PreflightError, kill_active, sweep_stale_turn_dirs
from .client import Backoff, BridgeClient, StopReason
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
    ws_url,
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
# `run --service` after the server refuses the credential: 10 s, doubling to 5 min.
SERVICE_RETRY_INITIAL_S = 10.0
SERVICE_RETRY_MAX_S = 300.0
# How long `pair` and `service install` wait for the service's first connection.
CONNECT_WAIT_S = 20.0
CONNECT_POLL_S = 0.25


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


def _command(*args: str) -> str:
    """A bridge command to print, with this bridge's full path so it works when pasted into
    a shell that doesn't have it on the PATH: `/Users/o/.local/bin/yantraxa-bridge run`, or
    on Windows `& "C:\\Users\\O B\\.local\\bin\\yantraxa-bridge.exe" run`."""
    program, *rest = [*svc.bridge_command(), *args]
    if os.name != "nt":
        return shlex.join([program, *rest])
    first = f'& "{program}"' if " " in program else program
    return " ".join([first, *(subprocess.list2cmdline([word]) for word in rest)])


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
        help="Ask nothing: with both Claude Code and Codex installed, pass --adapter too.",
    ),
    no_service: bool = typer.Option(
        False, "--no-service", help="Don't start the bridge in the background after pairing."
    ),
) -> None:
    """Pair this machine with an agent: prints a code to enter in Yantraxa, then starts the
    bridge in the background (when you log in, too) and waits for it to connect."""
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
    _after_pair(no_service=no_service)


def _after_pair(*, no_service: bool) -> None:
    """Start the background service and wait for it to connect (0.2.1: no question, so a
    person watching the browser never misses one), or say how to start the bridge."""
    if no_service:
        if _service_for_this_home():
            typer.echo(
                "A background bridge is installed; restart it with the new pairing: "
                + _command("service", "install")
            )
        else:
            typer.echo("Start the bridge with: " + _command("run"))
        return
    _start_service()


def _start_service() -> bool:
    """Install (or reinstall) the service, wait for its first connection and say how it
    went. False when no service could be set up; the bridge is still paired then."""
    started = datetime.now(UTC).replace(microsecond=0)
    try:
        _install_service()
    except (svc.ServiceError, ConfigError, OSError) as error:
        typer.secho(f"Couldn't set up the background service: {error}", fg="red", err=True)
        typer.echo("Start the bridge with: " + _command("run"))
        return False
    typer.echo(f"Waiting up to {CONNECT_WAIT_S:.0f} s for the bridge to connect…")
    handle, reason = _wait_for_connection(started)
    if handle is not None:
        typer.secho(
            f"Connected as @{handle}. It starts automatically when you log in.",
            fg=typer.colors.GREEN,
        )
    else:
        typer.secho(f"Paired, but not connected yet: {reason}.", fg=typer.colors.YELLOW)
        if _running_service_pid() is not None:
            # A second bridge started with `run` would replace the service (4009, final).
            typer.echo("The background bridge is running and keeps trying.")
            typer.echo("If it is still not connected in a minute, run: " + _command("doctor"))
        else:
            typer.echo("If it is still not connected in a minute, run: " + _command("run"))
        typer.echo(f"Log: {yantraxa_home() / svc.LOG_DIR / svc.BRIDGE_LOG}")
    return True


def _running_service_pid() -> int | None:
    """The pid of the background service running this home, or None (0 when it runs but
    reports no pid)."""
    state = runstate.read_state()
    if state.get("mode") == "service" and runstate.is_running(state):
        return int(state["pid"])
    try:
        status = svc.current_backend().status()
    except svc.ServiceError:
        return None
    if status.installed and status.running and _is_this_home(status.home):
        return status.pid or 0
    return None


def _wait_for_connection(started: datetime) -> tuple[str | None, str]:
    """(handle, "") once the service started after `started` records a connection, else
    (None, why not) after CONNECT_WAIT_S: its last error, or that it stopped or isn't
    running. A record from the run it replaced (stopped by the reinstall) doesn't count."""
    deadline = time.monotonic() + CONNECT_WAIT_S
    while True:
        state = runstate.read_state()
        begun = runstate.parse_time(state.get("started_at"))
        if begun is None or begun < started:
            state = {}
        connected = runstate.parse_time(state.get("connected_at"))
        if connected is not None and connected >= started:
            return str(state.get("handle")), ""
        stopped = runstate.parse_time(state.get("stopped_at"))
        if stopped is not None and stopped >= started:
            return None, f"the bridge stopped: {state.get('stopped')}"
        if time.monotonic() >= deadline:
            break
        time.sleep(CONNECT_POLL_S)
    failed = runstate.parse_time(state.get("last_error_at"))
    if failed is not None and failed >= started:
        return None, str(state.get("last_error"))
    try:
        status = svc.current_backend().status()
    except svc.ServiceError as error:
        return None, str(error)
    if not status.running:
        exit_code = f" (last exit code {status.last_exit})" if status.last_exit else ""
        return None, f"the background bridge isn't running{exit_code}"
    return None, f"no connection within {CONNECT_WAIT_S:.0f} s"


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


class _Stopper:
    """What a stop signal stops: the client running now, and `run --service`'s retry wait."""

    def __init__(self) -> None:
        self.event = asyncio.Event()
        self.client: BridgeClient | None = None

    async def stop(self) -> None:
        self.event.set()
        if self.client is not None:
            await self.client.stop()

    async def wait(self, delay: float) -> bool:
        """Sleeps `delay`; True when a stop signal came first."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.event.wait(), delay)
        return self.event.is_set()


def _stop_signals() -> tuple[signal.Signals, ...]:
    """The signals `run` handles. SIGHUP is left alone when it is ignored already, as under
    `nohup yantraxa-bridge run &`: handling it would end the bridge when the terminal closes."""
    hup = getattr(signal, "SIGHUP", None)
    return tuple(
        sig for sig in _STOP_SIGNALS if not (sig == hup and signal.getsignal(sig) == signal.SIG_IGN)
    )


def _on_stop_signal(sig: signal.Signals, stopper: _Stopper, log: logging.Logger) -> None:
    """Kills any running CLI turn with its process group and deletes its folders straight
    away, then stops the client."""
    if stopped := kill_active():
        log.info("%s: stopped %d running CLI turn(s)", signal.Signals(sig).name, stopped)
    task = asyncio.ensure_future(stopper.stop())
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
        help="Run as the background service: log to $YANTRAXA_HOME/log/bridge.log, and keep "
        "retrying anything a person can fix instead of stopping.",
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
    state = runstate.StateWriter(home, mode="service" if service else "terminal")
    if service:
        log.info("starting as the background service (yantraxa-bridge %s)", __version__)
        state.started()
        raise typer.Exit(asyncio.run(_serve(state, log)))

    if (pid := _running_service_pid()) is not None:
        # Two bridges for one agent knock each other off: the newer one wins (4009) and the
        # service would stop for good.
        running = f" (pid {pid})" if pid else ""
        raise _fail(
            f"The background service already runs this bridge{running}; a second one would "
            f"replace it. See `{_command('status')}`, or stop it first with "
            f"`{_command('service', 'uninstall')}`."
        )
    try:
        config, credentials, turns = _load_run()
    except ConfigError as error:
        raise _fail(str(error)) from None
    state.started()
    _sweep(log)

    async def main() -> StopReason:
        stopper = _Stopper()
        _handle_stop_signals(stopper, log)
        return await _run_client(config, credentials, turns, state, stopper)

    reason = asyncio.run(main())
    state.stopped(reason.value)
    if reason is StopReason.DISCONNECTED:
        typer.echo("The owner disconnected this agent in Yantraxa. Stopped.")
    elif reason is StopReason.SUPERSEDED:
        raise _fail("Another bridge connected as this agent, so this one stopped.")
    elif reason is StopReason.UNAUTHORIZED:
        typer.secho(
            "The server refused this bridge's credential. Diagnosing:",
            fg=typer.colors.RED,
            err=True,
        )
        _print_checks(asyncio.run(run_diagnose()))
        raise typer.Exit(1)


def _load_run():
    """The config, credential and turn handler `run` needs; raises ConfigError saying what
    to run when one is missing."""
    config = load_config()
    credentials = load_credentials()
    if credentials is None or not config.server_url:
        raise ConfigError(f"not paired yet: run `{_command('pair')}`")
    ws_url(config.server_url)  # a bad or plain-http remote URL is a ConfigError here, not later
    secret_filter.add_secret(credentials.secret_value())
    return config, credentials, make_turn_handler(config)


def _sweep(log: logging.Logger) -> None:
    if swept := sweep_stale_turn_dirs():
        log.info("removed %d turn folder(s) left by an earlier bridge", len(swept))


def _handle_stop_signals(stopper: _Stopper, log: logging.Logger) -> None:
    loop = asyncio.get_running_loop()
    for sig in _stop_signals():
        with contextlib.suppress(NotImplementedError):  # Windows
            loop.add_signal_handler(sig, _on_stop_signal, sig, stopper, log)


async def _run_client(config, credentials, turns, state, stopper: _Stopper) -> StopReason:
    """One client until it stops; what it does is recorded in `run-state.json`."""
    client = BridgeClient(
        config,
        credentials,
        turn_handler=turns,
        on_connected=state.connected,
        on_error=state.error,
    )
    stopper.client = client
    try:
        if stopper.event.is_set():
            return StopReason.STOPPED
        return await client.run()
    finally:
        stopper.client = None
        aclose = getattr(turns, "aclose", None)
        if aclose is not None:
            await aclose()


async def _serve(state: runstate.StateWriter, log: logging.Logger) -> int:
    """`run --service`: stays up through a refused credential (retrying 10 s, doubling to
    5 min; the credential is read again each time, so pairing again fixes it in place) and
    through network trouble (the client retries that itself). Exits 0 when the owner
    disconnected the agent or another bridge took over, and 1 when there is nothing to run,
    so launchd, systemd or the Windows supervisor starts it again 10 s later."""
    stopper = _Stopper()
    _handle_stop_signals(stopper, log)
    backoff = Backoff(SERVICE_RETRY_INITIAL_S, SERVICE_RETRY_MAX_S)
    swept = False
    while True:
        try:
            config, credentials, turns = _load_run()
        except ConfigError as error:
            message = str(error)
            log.error("stopped: %s", message)
            state.error(message)
            state.stopped(message)
            typer.echo(message, err=True)
            return 1
        if not swept:
            _sweep(log)
            swept = True
        reason = await _run_client(config, credentials, turns, state, stopper)
        if reason is StopReason.STOPPED:
            state.stopped("stopped by a signal")
            return 0
        if reason is StopReason.DISCONNECTED:
            log.info("stopped: the owner disconnected this agent in Yantraxa")
            state.stopped("the owner disconnected this agent in Yantraxa")
            typer.echo("The owner disconnected this agent in Yantraxa. Stopped.")
            return 0
        if reason is StopReason.SUPERSEDED:
            message = "another bridge connected as this agent, so this one stopped"
            log.error("stopped: %s", message)
            state.stopped(message)
            typer.echo(message, err=True)
            return 0
        delay = backoff.next()
        message = "the server refused this bridge's credential"
        log.error(
            "%s; retrying in %.0fs (check with `%s`; if the agent was revoked, pair again)",
            message,
            delay,
            _command("diagnose"),
        )
        state.error(message)
        if await stopper.wait(delay):
            state.stopped("stopped by a signal")
            return 0


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
    state = runstate.read_state()
    if state.get("mode") == "terminal" and runstate.is_running(state):
        detail = runstate.describe(state)
        typer.echo(f"Terminal:    running (pid {state['pid']})" + (f", {detail}" if detail else ""))
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
        typer.echo(f"Server view: HTTP {response.status_code} (run `{_command('diagnose')}`)")


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
    if (running := _running_check()) is not None:
        checks.append(running)
    if not _print_checks(checks):
        raise typer.Exit(1)


def _running_check() -> Check | None:
    """Whether anything runs this paired bridge: FAIL with the fix when nothing does, or
    when it runs but its last attempt failed. None when it isn't paired."""
    try:
        if load_credentials() is None:
            return None
    except ConfigError:
        return None
    home = yantraxa_home()
    log_path = home / svc.LOG_DIR / svc.BRIDGE_LOG
    state = runstate.read_state()
    if runstate.is_running(state):
        where = "the background service" if state.get("mode") == "service" else "a terminal"
        running = f"running in {where} (pid {state['pid']})"
        if runstate.connected_now(state):
            return Check("running", True, f"{running}, {runstate.describe(state)}")
        if state.get("last_error"):
            return Check(
                "running",
                False,
                f"{running}, not connected: {runstate.describe(state)}; it keeps retrying "
                f"(log: {log_path}). Don't also start `run`: a second bridge replaces this one",
            )
        return Check("running", None, f"{running}, not connected yet")
    fix_service = f"run `{_command('service', 'install')}`"
    try:
        status = svc.current_backend().status()
    except svc.ServiceError as error:
        return Check(
            "running",
            False,
            f"nothing runs the bridge, and {error}: run `{_command('run')}` and leave it open",
        )
    if status.installed and not _is_this_home(status.home):
        return Check(
            "running",
            False,
            f"the background service runs another home ({status.home}), not {home}: " + fix_service,
        )
    if status.installed and status.running:
        pid = f" (pid {status.pid})" if status.pid else ""
        return Check("running", True, f"the background service is running{pid}")
    stopped = (
        f"; it stopped at {runstate.local_time(state['stopped_at'])}: {state.get('stopped')}"
        if state.get("stopped_at")
        else ""
    )
    if status.installed:
        exit_code = f" (last exit code {status.last_exit})" if status.last_exit else ""
        line = svc.last_log_line(home)
        last = f"; last log: {line}" if line else ""
        return Check(
            "running",
            False,
            f"the background service is installed but not running{exit_code}{stopped}{last}. "
            f"Fix: {fix_service}",
        )
    return Check(
        "running",
        False,
        f"paired, but nothing runs the bridge{stopped}: {fix_service} "
        f"(or `{_command('run')}` in a terminal left open)",
    )


service_app = typer.Typer(
    no_args_is_help=True,
    help="Keep the bridge running in the background, started when you log in.",
)
app.add_typer(service_app, name="service")


def _is_this_home(home: Path | None) -> bool:
    """Whether a service's YANTRAXA_HOME is this one (an unreadable one counts as this)."""
    if home is None:
        return True
    return home.expanduser().absolute() == yantraxa_home().expanduser().absolute()


def _service_for_this_home() -> bool:
    """Whether the background service is installed and runs this YANTRAXA_HOME: there is
    one service per user, and it may belong to another home."""
    try:
        status = svc.current_backend().status()
    except svc.ServiceError:
        return False
    return status.installed and _is_this_home(status.home)


def _service_summary() -> str:
    try:
        status = svc.current_backend().status()
    except svc.ServiceError:
        return "not available on this system"
    if not status.installed:
        return f"not installed (`{_command('service', 'install')}`)"
    if not _is_this_home(status.home):
        return f"installed for another home ({status.home})"
    summary = "installed, " + ("running" if status.running else "not running")
    state = runstate.read_state()
    if state.get("mode") != "service":
        return summary
    if status.running:
        detail = runstate.describe(state)
        return summary + (f", {detail}" if detail else "")
    if state.get("stopped_at"):
        when = runstate.local_time(state["stopped_at"])
        return summary + f", stopped at {when}: {state.get('stopped')}"
    return summary


def _install_service() -> None:
    """Install (or reinstall, which restarts) the background service; raises ServiceError."""
    config = load_config()
    if load_credentials() is None or not config.server_url:
        raise svc.ServiceError(f"not paired yet: run `{_command('pair')}` first")
    backend = svc.current_backend()
    launch = svc.make_launch()
    try:
        current = backend.status()
    except svc.ServiceError:
        current = None
    if current is not None and current.installed and not _is_this_home(current.home):
        typer.echo(f"Replacing the background bridge for {current.home} (one per user).")
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
        f"Check it with `{_command('service', 'status')}`; remove it with "
        f"`{_command('service', 'uninstall')}`."
    )


def _api_key_hint(variable: str) -> str:
    """How to give the service the API key: it never goes in the unit file."""
    if sys.platform == "darwin":
        how = (
            f'launchctl setenv {variable} "${variable}", then `{_command("service", "install")}` '
            "again; it lasts until you log out"
        )
    elif os.name == "nt":
        how = f'setx {variable} "<key>", then `{_command("service", "install")}` in a new terminal'
    else:
        how = (
            "`systemctl --user edit yantraxa-bridge`, add the two lines "
            f"[Service] and Environment={variable}=<key>, then `chmod 600` the override file"
        )
    return f"API-key mode: the service needs {variable} in its own environment: {how}."


@service_app.command("install")
def service_install() -> None:
    """Run the bridge in the background now and whenever you log in, and wait for it to
    connect. Needs no admin."""
    if not _start_service():
        raise typer.Exit(1)


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
    state = runstate.read_state()
    if state.get("mode") == "service" and (detail := runstate.describe(state)):
        typer.echo(f"Connection:  {detail}")
    if not status.installed:
        typer.echo("Install it with: " + _command("service", "install"))


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
