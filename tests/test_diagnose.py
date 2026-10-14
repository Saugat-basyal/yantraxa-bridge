from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import respx

from yantraxa_bridge.config import BridgeConfig, save_config
from yantraxa_bridge.credentials import Credentials, save_credentials
from yantraxa_bridge.diagnose import diagnose

SERVER = "http://localhost:4000"
OLLAMA = "http://127.0.0.1:11434"


def by_name(checks):
    return {c.name: c for c in checks}


async def test_not_paired(home):
    checks = await diagnose()
    assert [(c.name, c.ok) for c in checks] == [("config", False)]


def paired(model="llama3.1:8b", adapter="ollama", **extra):
    save_config(BridgeConfig(server_url=SERVER, adapter=adapter, model=model, **extra))
    save_credentials(Credentials(agent_id="ag_abc234", secret="s3cret"))


@respx.mock
async def test_all_good(home):
    paired()
    respx.get(f"{SERVER}/health").mock(
        return_value=httpx.Response(
            200,
            json={"ok": True},
            headers={"date": format_datetime(datetime.now(UTC), usegmt=True)},
        )
    )
    me = respx.get(f"{SERVER}/v1/agent/me").mock(
        return_value=httpx.Response(
            200, json={"agent": {"handle": "saugat.ai", "state": "active"}, "permissions": {}}
        )
    )
    respx.get(f"{OLLAMA}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})
    )
    checks = by_name(await diagnose())
    assert all(c.ok for c in checks.values()), checks
    assert "saugat.ai" in checks["credential"].detail
    assert me.calls[0].request.headers["authorization"] == "Bearer ag_abc234:s3cret"
    assert "s3cret" not in " ".join(c.detail for c in checks.values())


@respx.mock
async def test_revoked_credential_skew_and_missing_model(home):
    paired(model="qwen3:8b")
    skewed = datetime.now(UTC) - timedelta(minutes=5)
    respx.get(f"{SERVER}/health").mock(
        return_value=httpx.Response(200, headers={"date": format_datetime(skewed, usegmt=True)})
    )
    respx.get(f"{SERVER}/v1/agent/me").mock(return_value=httpx.Response(401))
    respx.get(f"{OLLAMA}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})
    )
    checks = by_name(await diagnose())
    assert checks["credential"].ok is False and "pair again" in checks["credential"].detail
    assert checks["clock"].ok is None
    assert checks["adapter"].ok is False and "qwen3:8b" in checks["adapter"].detail


@respx.mock
async def test_unreachable_server_and_ollama(home):
    paired()
    respx.get(f"{SERVER}/health").mock(side_effect=httpx.ConnectError("refused"))
    respx.get(f"{OLLAMA}/api/tags").mock(side_effect=httpx.ConnectError("refused"))
    checks = by_name(await diagnose())
    assert checks["server"].ok is False
    assert checks["adapter"].ok is False
    assert "credential" not in checks


@respx.mock
async def test_personal_agent_without_url(home):
    paired(adapter="personal-agent", model=None)
    respx.get(f"{SERVER}/health").mock(return_value=httpx.Response(200))
    respx.get(f"{SERVER}/v1/agent/me").mock(return_value=httpx.Response(403))
    checks = by_name(await diagnose())
    assert checks["adapter"].ok is False
    assert "paused" in checks["credential"].detail
