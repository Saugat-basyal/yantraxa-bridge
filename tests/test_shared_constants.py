"""The bridge's close codes and timeouts against the server's, read from the TypeScript
source in packages/shared (the bridge ships alone, so its copies must not drift)."""

import re
from pathlib import Path

import pytest

from yantraxa_bridge import protocol
from yantraxa_bridge.client import Timing

SHARED = Path(__file__).resolve().parents[2] / "packages" / "shared" / "src"

pytestmark = pytest.mark.skipif(
    not SHARED.is_dir(), reason="packages/shared is not here (the public bridge repo)"
)


def constant(file: str, name: str) -> int:
    text = (SHARED / file).read_text(encoding="utf-8")
    found = re.search(rf"export const {name} = ([\d_]+);", text)
    assert found, f"{name} is not in {file}"
    return int(found.group(1).replace("_", ""))


@pytest.mark.parametrize(
    ("file", "name", "ours"),
    [
        ("realtime.ts", "WS_CLOSE_UNAUTHORIZED", protocol.CLOSE_UNAUTHORIZED),
        ("agent-transport.ts", "WS_CLOSE_AUTH_INCOMPLETE", protocol.CLOSE_AUTH_INCOMPLETE),
        ("agent-transport.ts", "WS_CLOSE_AGENT_PAUSED", protocol.CLOSE_AGENT_PAUSED),
        ("agent-transport.ts", "WS_CLOSE_SUPERSEDED", protocol.CLOSE_SUPERSEDED),
        ("agent-transport.ts", "WS_CLOSE_NORMAL", protocol.CLOSE_NORMAL),
    ],
)
def test_close_codes_match_the_server(file, name, ours):
    assert constant(file, name) == ours


def test_auth_timeout_matches_and_the_bridge_waits_longer():
    server_ms = constant("agent-transport.ts", "AGENT_AUTH_TIMEOUT_MS")
    assert server_ms == protocol.SERVER_AUTH_TIMEOUT_S * 1000
    assert Timing().auth_timeout_s * 1000 > server_ms
