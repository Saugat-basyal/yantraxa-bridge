"""Device-style pairing (spec §3.3): start, show the code, poll until approved."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx

from . import __version__
from .config import load_config, normalise_server_url, save_config
from .credentials import Credentials, load_or_create_public_key, save_credentials
from .protocol import adapter_type_wire

POLL_INTERVAL_S = 3.0
# The server's window is 5 minutes and restarts on approval; this only stops a
# bridge that can't reach the server from waiting forever.
MAX_WAIT_S = 15 * 60


class PairingError(Exception):
    """Pairing could not start or finish."""


class PairingExpired(PairingError):
    """The code expired, was used, or was never approved."""


def fingerprint(public_key: str) -> str:
    """`SHA256:<base64>` of the public key string: the server's `fingerprint()`."""
    digest = base64.b64encode(hashlib.sha256(public_key.encode()).digest()).decode()
    return "SHA256:" + digest.rstrip("=")


def _echo(line: str) -> None:
    # Flushed, so the code shows at once even when stdout is a pipe or a log file.
    print(line, flush=True)


def format_code(code: str) -> str:
    return f"{code[:3]}-{code[3:]}" if len(code) == 6 else code


async def pair(
    server_url: str,
    adapter: str,
    host_label: str,
    *,
    home: Path | None = None,
    http: httpx.AsyncClient | None = None,
    poll_interval_s: float = POLL_INTERVAL_S,
    max_wait_s: float = MAX_WAIT_S,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    echo: Callable[[str], None] | None = None,
    now: Callable[[], float] = time.monotonic,
) -> Credentials:
    """Pair this bridge with an agent; stores credentials and config on success."""
    echo = echo or _echo
    server = normalise_server_url(server_url)
    public_key = load_or_create_public_key(home)
    owned = http is None
    client = http or httpx.AsyncClient(timeout=15)
    try:
        start = await _start(client, server, adapter, host_label, public_key)
        echo(f"Pairing code:  {format_code(start['code'])}")
        echo(f"Fingerprint:   {fingerprint(public_key)}")
        echo(
            "Enter the code in Yantraxa (agent setup, Local AI step) and check the "
            "fingerprint matches. Waiting for approval; the code expires in 5 minutes."
        )
        agent_id, secret = await _poll(
            client,
            server,
            start["pairing_id"],
            start["poll_token"],
            poll_interval_s=poll_interval_s,
            max_wait_s=max_wait_s,
            sleep=sleep,
            now=now,
        )
    finally:
        if owned:
            await client.aclose()

    credentials = Credentials(agent_id=agent_id, secret=secret)
    save_credentials(credentials, home)
    config = load_config(home).model_copy(
        update={"server_url": server, "adapter": adapter, "host_label": host_label}
    )
    save_config(config, home)
    echo(f"Paired as {agent_id}.")
    return credentials


async def _start(
    client: httpx.AsyncClient, server: str, adapter: str, host_label: str, public_key: str
) -> dict[str, str]:
    body = {
        "public_key": public_key,
        "bridge_version": __version__,
        "adapter_type": adapter_type_wire(adapter),
        "host_label": host_label,
    }
    try:
        response = await client.post(f"{server}/v1/pairing/start", json=body)
    except httpx.HTTPError as error:
        raise PairingError(f"can't reach {server}: {error}") from error
    if response.status_code == 201:
        return response.json()
    if response.status_code == 400:
        raise PairingError("the server refused the pairing request (check --adapter)")
    if response.status_code == 429:
        raise PairingError("too many pairing attempts from this address; try again later")
    raise PairingError(f"pairing start failed: HTTP {response.status_code}")


async def _poll(
    client: httpx.AsyncClient,
    server: str,
    pairing_id: str,
    poll_token: str,
    *,
    poll_interval_s: float,
    max_wait_s: float,
    sleep: Callable[[float], Awaitable[None]],
    now: Callable[[], float],
) -> tuple[str, str]:
    deadline = now() + max_wait_s
    headers = {"Authorization": f"Bearer {poll_token}"}
    while now() < deadline:
        await sleep(poll_interval_s)
        try:
            response = await client.get(f"{server}/v1/pairing/{pairing_id}", headers=headers)
        except httpx.HTTPError:
            continue  # a blip; keep polling until the deadline
        if response.status_code == 404:
            raise PairingExpired("the pairing code expired or was already used; pair again")
        if response.status_code == 429:
            await sleep(poll_interval_s)
            continue
        if response.status_code != 200:
            continue
        body = response.json()
        if body.get("status") == "approved":
            return body["agent_id"], body["secret"]
    raise PairingExpired("gave up waiting for approval; pair again")
