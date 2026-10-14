"""`yantraxa-bridge diagnose`: config, credentials, server, clock skew, credential, adapter."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx

from .config import CLI_ADAPTERS, ConfigError, config_path, load_config, normalise_server_url
from .credentials import credentials_path, load_credentials

MAX_SKEW_S = 30


@dataclass
class Check:
    name: str
    ok: bool | None  # None: a warning or not applicable
    detail: str


async def diagnose(home: Path | None = None, http: httpx.AsyncClient | None = None) -> list[Check]:
    checks: list[Check] = []
    try:
        config = load_config(home)
    except ConfigError as error:
        return [Check("config", False, str(error))]
    if not config_path(home).exists() or not config.server_url:
        checks.append(Check("config", False, "not paired yet: run `yantraxa-bridge pair`"))
        return checks
    server = normalise_server_url(config.server_url)
    checks.append(Check("config", True, f"server {server}, adapter {config.adapter}"))

    try:
        credentials = load_credentials(home)
    except ConfigError as error:
        credentials = None
        checks.append(Check("credentials", False, str(error)))
    else:
        if credentials is None:
            checks.append(Check("credentials", False, "missing: run `yantraxa-bridge pair`"))
        else:
            mode = stat.S_IMODE(os.stat(credentials_path(home)).st_mode)
            checks.append(Check("credentials", True, f"{credentials.agent_id}, mode {mode:o}"))

    owned = http is None
    client = http or httpx.AsyncClient(timeout=10)
    try:
        checks += await _server_checks(client, server, credentials)
        checks.append(await _adapter_check(client, config, home))
    finally:
        if owned:
            await client.aclose()
    return checks


async def _server_checks(client: httpx.AsyncClient, server: str, credentials) -> list[Check]:
    try:
        health = await client.get(f"{server}/health")
    except httpx.HTTPError as error:
        return [Check("server", False, f"can't reach {server}: {error}")]
    checks = [Check("server", health.is_success, f"{server}/health → HTTP {health.status_code}")]

    date = health.headers.get("date")
    if date:
        skew = (datetime.now(UTC) - parsedate_to_datetime(date)).total_seconds()
        ok = abs(skew) <= MAX_SKEW_S
        checks.append(Check("clock", ok if ok else None, f"{skew:+.0f}s against the server"))

    if credentials is not None:
        try:
            me = await client.get(
                f"{server}/v1/agent/me", headers={"Authorization": credentials.bearer()}
            )
        except httpx.HTTPError as error:
            checks.append(Check("credential", False, f"request failed: {error}"))
        else:
            checks.append(_credential_check(me))
    return checks


def _credential_check(response: httpx.Response) -> Check:
    if response.status_code == 200:
        agent = response.json().get("agent", {})
        return Check("credential", True, f"@{agent.get('handle')} ({agent.get('state')})")
    if response.status_code == 401:
        return Check(
            "credential", False, "rejected: revoked, re-paired elsewhere or wrong; pair again"
        )
    if response.status_code == 403:
        return Check("credential", False, "the agent is paused by its owner")
    return Check("credential", False, f"HTTP {response.status_code}")


async def _adapter_check(client: httpx.AsyncClient, config, home: Path | None = None) -> Check:
    if config.adapter in CLI_ADAPTERS:
        # The preflight's facts without its probe turn; `doctor` adds the probe.
        from .adapters import cli_facts
        from .adapters.cli_runner import PreflightError

        try:
            return Check("adapter", True, await cli_facts(config, home=home))
        except (PreflightError, ConfigError) as error:
            return Check("adapter", False, str(error))
    if config.adapter == "ollama":
        try:
            response = await client.get(f"{config.ollama_host.rstrip('/')}/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as error:
            return Check("adapter", False, f"Ollama at {config.ollama_host}: {error}")
        models = [m.get("name") for m in response.json().get("models", [])]
        if config.model and config.model not in models:
            return Check(
                "adapter",
                False,
                f"model {config.model!r} not in Ollama ({', '.join(models) or 'none'})",
            )
        detail = f"Ollama at {config.ollama_host}, models: {', '.join(models) or 'none'}"
        return Check(
            "adapter",
            True if config.model else None,
            detail if config.model else detail + "; no `model` set in config.toml",
        )
    if config.adapter == "personal-agent":
        if not config.personal_agent_url:
            return Check("adapter", False, "personal_agent_url is not set in config.toml")
        try:
            await client.get(config.personal_agent_url)
        except httpx.HTTPError as error:
            return Check("adapter", False, f"{config.personal_agent_url}: {error}")
        return Check("adapter", True, f"personal agent at {config.personal_agent_url}")
    return Check("adapter", None, f"{config.adapter}: nothing to check")
