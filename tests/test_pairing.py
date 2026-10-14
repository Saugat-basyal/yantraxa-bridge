import base64
import hashlib
import json
import os
import stat

import httpx
import pytest
import respx

from yantraxa_bridge import __version__
from yantraxa_bridge.config import load_config
from yantraxa_bridge.credentials import load_credentials, load_or_create_public_key
from yantraxa_bridge.pairing import PairingError, PairingExpired, fingerprint, pair

SERVER = "http://localhost:4000"
PAIRING_ID = "0f8e1c2a-3b4d-4e5f-8a9b-0c1d2e3f4a5b"
START = {
    "pairing_id": PAIRING_ID,
    "poll_token": "poll-tok",
    "code": "K7QM2X",
    "expires_at": "2026-10-11T20:05:00.000Z",
}


async def no_sleep(_seconds):
    return None


def server_fingerprint(public_key: str) -> str:
    """apps/server/src/gateway/pairing.ts `fingerprint`."""
    digest = base64.b64encode(hashlib.sha256(public_key.encode()).digest()).decode()
    return "SHA256:" + digest.rstrip("=")


def test_fingerprint_matches_the_server():
    assert fingerprint("abc") == server_fingerprint("abc")


def test_public_key_is_stable_and_private_key_is_0600(home):
    first = load_or_create_public_key()
    second = load_or_create_public_key()
    assert first == second
    assert len(base64.b64decode(first)) == 32  # raw Ed25519
    assert stat.S_IMODE(os.stat(home / "bridge_key.pem").st_mode) == 0o600


@respx.mock
async def test_pair_happy_path(home):
    start = respx.post(f"{SERVER}/v1/pairing/start").mock(
        return_value=httpx.Response(201, json=START)
    )
    poll = respx.get(f"{SERVER}/v1/pairing/{PAIRING_ID}").mock(
        side_effect=[
            httpx.Response(200, json={"status": "pending"}),
            httpx.Response(200, json={"status": "pending"}),
            httpx.Response(
                200,
                json={"status": "approved", "agent_id": "ag_abc234", "secret": "the-secret_123"},
            ),
        ]
    )
    lines: list[str] = []
    creds = await pair(SERVER + "/", "personal-agent", "lab-box", sleep=no_sleep, echo=lines.append)

    body = json.loads(start.calls[0].request.content)
    public_key = load_or_create_public_key()
    assert body == {
        "public_key": public_key,
        "bridge_version": __version__,
        "adapter_type": "personal_agent",
        "host_label": "lab-box",
    }
    assert poll.call_count == 3
    assert all(c.request.headers["authorization"] == "Bearer poll-tok" for c in poll.calls)

    output = "\n".join(lines)
    assert "K7Q-M2X" in output
    assert server_fingerprint(public_key) in output
    assert "the-secret_123" not in output

    assert creds.agent_id == "ag_abc234"
    stored = load_credentials()
    assert stored is not None and stored.secret_value() == "the-secret_123"
    assert stat.S_IMODE(os.stat(home / "credentials").st_mode) == 0o600
    config = load_config()
    assert config.server_url == SERVER
    assert config.adapter == "personal-agent"
    assert config.host_label == "lab-box"


@respx.mock
async def test_expired_pairing_stores_nothing(home):
    respx.post(f"{SERVER}/v1/pairing/start").mock(return_value=httpx.Response(201, json=START))
    respx.get(f"{SERVER}/v1/pairing/{PAIRING_ID}").mock(
        side_effect=[
            httpx.Response(200, json={"status": "pending"}),
            httpx.Response(404, json={"error": "pairing_invalid"}),
        ]
    )
    with pytest.raises(PairingExpired):
        await pair(SERVER, "ollama", "box", sleep=no_sleep, echo=lambda _: None)
    assert load_credentials() is None


@respx.mock
async def test_poll_survives_a_network_blip_and_a_429(home):
    respx.post(f"{SERVER}/v1/pairing/start").mock(return_value=httpx.Response(201, json=START))
    respx.get(f"{SERVER}/v1/pairing/{PAIRING_ID}").mock(
        side_effect=[
            httpx.ConnectError("down"),
            httpx.Response(429, json={"error": "rate_limited"}),
            httpx.Response(
                200, json={"status": "approved", "agent_id": "ag_abc234", "secret": "s"}
            ),
        ]
    )
    creds = await pair(SERVER, "ollama", "box", sleep=no_sleep, echo=lambda _: None)
    assert creds.agent_id == "ag_abc234"


@respx.mock
async def test_poll_gives_up_after_max_wait(home):
    respx.post(f"{SERVER}/v1/pairing/start").mock(return_value=httpx.Response(201, json=START))
    respx.get(f"{SERVER}/v1/pairing/{PAIRING_ID}").mock(
        return_value=httpx.Response(200, json={"status": "pending"})
    )
    clock = iter(range(0, 10_000, 3))
    with pytest.raises(PairingExpired):
        await pair(
            SERVER,
            "ollama",
            "box",
            sleep=no_sleep,
            echo=lambda _: None,
            max_wait_s=30,
            now=lambda: float(next(clock)),
        )


@pytest.mark.parametrize(
    ("status", "message"),
    [(400, "refused"), (429, "too many"), (500, "500")],
)
@respx.mock
async def test_start_errors_are_explained(home, status, message):
    respx.post(f"{SERVER}/v1/pairing/start").mock(
        return_value=httpx.Response(status, json={"error": "x"})
    )
    with pytest.raises(PairingError, match=message):
        await pair(SERVER, "ollama", "box", sleep=no_sleep, echo=lambda _: None)


@respx.mock
async def test_unreachable_server(home):
    respx.post(f"{SERVER}/v1/pairing/start").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(PairingError, match="reach"):
        await pair(SERVER, "ollama", "box", sleep=no_sleep, echo=lambda _: None)


@respx.mock
async def test_repairing_keeps_other_config(home):
    from yantraxa_bridge.config import BridgeConfig, save_config

    save_config(
        BridgeConfig(
            server_url="http://old:4000",
            adapter="ollama",
            model="llama3.1:8b",
            auto_accept_pending=True,
        )
    )
    respx.post(f"{SERVER}/v1/pairing/start").mock(return_value=httpx.Response(201, json=START))
    respx.get(f"{SERVER}/v1/pairing/{PAIRING_ID}").mock(
        return_value=httpx.Response(
            200, json={"status": "approved", "agent_id": "ag_abc234", "secret": "s"}
        )
    )
    await pair(SERVER, "ollama", "box", sleep=no_sleep, echo=lambda _: None)
    config = load_config()
    assert config.server_url == SERVER
    assert config.model == "llama3.1:8b"
    assert config.auto_accept_pending is True
