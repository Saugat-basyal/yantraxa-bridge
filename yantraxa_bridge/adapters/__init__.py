"""Adapters: how the bridge reaches the AI that takes the agent's turns (spec §10)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Adapter, AdapterInfo, Chunk, Final, TurnContext, TurnEvent

if TYPE_CHECKING:
    from ..client import TurnHandler
    from ..config import BridgeConfig

__all__ = [
    "Adapter",
    "AdapterInfo",
    "Chunk",
    "Final",
    "TurnContext",
    "TurnEvent",
    "cli_facts",
    "cli_probe",
    "make_cli_adapter",
    "make_turn_handler",
]


def make_turn_handler(config: BridgeConfig) -> TurnHandler:
    """The turn handler for the configured adapter; raises ConfigError when a setting is missing.

    `interactive-mcp` agents answer through the MCP client, not the bridge, so every turn
    the bridge gets for one passes. Every adapter's turns go in the transparency log.
    """
    # Imported here: the adapter modules import `prompt`, which imports `adapters.base`.
    from ..client import PassTurnHandler
    from ..config import ConfigError, normalise_adapter_url, yantraxa_home
    from ..redaction import Redactor
    from ..turnlog import LoggedTurnHandler, TurnLog
    from .ollama import OllamaAdapter
    from .personal_agent import PersonalAgentAdapter

    home = yantraxa_home()
    turnlog = TurnLog(home)
    adapter: Adapter
    if config.adapter in ("claude-code", "codex-cli"):
        adapter = make_cli_adapter(config, home=home, turnlog=turnlog)
    elif config.adapter == "ollama":
        if not config.model:
            raise ConfigError('set `model` in config.toml, e.g. model = "llama3.1:8b"')
        host = normalise_adapter_url(
            config.ollama_host,
            allow_insecure_http=config.allow_insecure_adapter_http,
        )
        adapter = OllamaAdapter(
            host,
            config.model,
            timeout_s=config.timeout_s,
            max_tokens=config.max_tokens,
        )
    elif config.adapter == "personal-agent":
        if not config.personal_agent_url:
            raise ConfigError("set `personal_agent_url` in config.toml")
        url = normalise_adapter_url(
            config.personal_agent_url,
            allow_insecure_http=config.allow_insecure_adapter_http,
        )
        adapter = PersonalAgentAdapter(url, timeout_s=config.timeout_s)
    else:
        return PassTurnHandler()
    redactor = Redactor(config.redaction_patterns, config.email_allowlist)
    return LoggedTurnHandler(
        adapter,
        redactor,
        timeout_s=config.timeout_s,
        turnlog=turnlog,
        adapter_name=config.adapter.replace("-", "_"),
    )


def make_cli_adapter(config: BridgeConfig, *, home=None, turnlog=None, caps: bool = True):
    """The `claude-code` or `codex-cli` adapter for `config`; `caps=False` for a probe turn."""
    from ..caps import Caps
    from ..config import api_key, codex_home_path, yantraxa_home
    from .claude_code import ClaudeCodeAdapter
    from .cli_runner import CLI_WALL_CLOCK_S, CliRunner
    from .codex_cli import CodexCliAdapter

    home = home or yantraxa_home()
    runner = CliRunner(timeout_s=min(config.timeout_s, CLI_WALL_CLOCK_S))
    limits = (
        Caps(home, per_meeting=config.max_turns_per_meeting, per_day=config.max_turns_per_day)
        if caps
        else None
    )
    if config.adapter == "claude-code":
        return ClaudeCodeAdapter(
            runner,
            model_fast=config.claude_model_fast,
            model_full=config.claude_model,
            effort=config.cli_effort,
            api_key=api_key(config),
            caps=limits,
            turnlog=turnlog,
        )
    return CodexCliAdapter(
        runner,
        codex_home=codex_home_path(config, home),
        model=config.codex_model,
        effort=config.cli_effort,
        api_key=api_key(config),
        allow_untested=config.codex_allow_untested,
        caps=limits,
        turnlog=turnlog,
    )


async def cli_facts(config: BridgeConfig, *, home=None, allow_login: bool = False) -> str:
    """The CLI adapter's version and sign-in, without a turn (`diagnose`); raises
    PreflightError or ConfigError."""
    from .claude_code import ClaudeCodeAdapter
    from .claude_code import facts as claude_facts
    from .codex_cli import facts as codex_facts

    adapter = make_cli_adapter(config, home=home, caps=False)
    if isinstance(adapter, ClaudeCodeAdapter):
        return await claude_facts(adapter)
    return await codex_facts(adapter, allow_login=allow_login)


async def cli_probe(config: BridgeConfig, *, home=None) -> str:
    """One confined probe turn (`doctor`), e.g. `probe OK (2.5 s, reply)`."""
    from .cli_runner import probe

    action, seconds = await probe(make_cli_adapter(config, home=home, caps=False))
    return f"probe OK ({seconds:.1f} s, {action.action})"
