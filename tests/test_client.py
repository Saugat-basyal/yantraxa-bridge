import asyncio
import json
import logging
import uuid

import httpx
import pytest
import respx

from yantraxa_bridge import __version__
from yantraxa_bridge.client import (
    Backoff,
    BridgeClient,
    PassTurnHandler,
    RateLimiter,
    StopReason,
    Timing,
    TurnIO,
)
from yantraxa_bridge.config import BridgeConfig
from yantraxa_bridge.credentials import Credentials
from yantraxa_bridge.protocol import TurnAction

from .fake_server import AGENT, FakeServer, frame

SECRET = "sup3r-secret_value"
MEETING = "3f2c6a5e-8a0e-4f7a-9d55-0d3c2a1b4c5d"
SEAT = "6b1e9d2a-1c3f-4e5a-8b7c-9d0e1f2a3b4c"
OTHER_SEAT = "7c2f0e3b-2d4a-4f6b-9c8d-0e1f2a3b4c5d"

FAST = Timing(
    heartbeat_s=0.05,
    pong_timeout_s=0.4,
    ack_timeout_s=0.1,
    max_resends=3,
    auth_timeout_s=1.0,
    backoff_initial_s=0.01,
    backoff_max_s=0.05,
    rate_frames=1000,
    rate_window_s=10,
)


def make_client(server: FakeServer, home, **config) -> BridgeClient:
    cfg = BridgeConfig(server_url=server.url, adapter="ollama", **config)
    return BridgeClient(
        cfg,
        Credentials(agent_id="ag_abc234", secret=SECRET),
        home=home,
        timing=FAST,
    )


class Running:
    """Runs a client against the fake server for the duration of a test."""

    def __init__(self, client: BridgeClient):
        self.client = client
        self.task: asyncio.Task[StopReason] | None = None

    async def __aenter__(self):
        self.task = asyncio.create_task(self.client.run())
        return self

    async def __aexit__(self, *exc):
        await self.client.stop()
        assert self.task is not None
        await asyncio.wait_for(self.task, 2)

    async def result(self, timeout=3.0) -> StopReason:
        assert self.task is not None
        return await asyncio.wait_for(self.task, timeout)


def joined_payload(profile=None):
    return {
        "meeting_id": MEETING,
        "code": "AB-1234",
        "seat_id": SEAT,
        "knowledge_profile": profile
        or {
            "approved_facts": [],
            "restricted_categories": [],
            "ask_when_uncertain": True,
            "return_summaries_to_agent": False,
        },
    }


def message_row(seat, body="hello"):
    return {
        "id": str(uuid.uuid4()),
        "meeting_id": MEETING,
        "seat_id": seat,
        "agenda_item_id": None,
        "kind": "message",
        "reply_to": None,
        "body": body,
        "created_at": "2026-10-11T20:00:00Z",
    }


def invitation_payload(auto_accepted: bool, invitation_id: str | None = None):
    return {
        "invitation_id": invitation_id or str(uuid.uuid4()),
        "purpose": "Join us",
        "meeting": {
            "id": MEETING,
            "code": "AB-1234",
            "title": "Plan",
            "purpose": "Plan Q4",
            "agenda": [],
            "scheduled_for": None,
            "organiser_handle": "saugat",
        },
        "auto_accepted": auto_accepted,
    }


async def test_authenticates_then_heartbeats(home):
    async with FakeServer() as server, Running(make_client(server, home)) as run:
        await server.wait_for(lambda: len(server.of_type("ping")) >= 3)
        auth = server.auth_frames[0]
        assert auth["type"] == "auth"
        assert auth["payload"] == {
            "agent_id": "ag_abc234",
            "secret": SECRET,
            "bridge_version": __version__,
            "adapter_type": "ollama",
            "capabilities": ["projects"],
        }
        uuid.UUID(auth["id"])
        assert run.client.agent is not None and run.client.agent["handle"] == "saugat.ai"
        assert run.client.connected


async def test_resends_until_acked_with_the_same_id(home):
    async with FakeServer() as server, Running(make_client(server, home)) as run:
        await server.authed.wait()
        server.withhold["message"] = 2
        frame_id = await run.client.send(
            "message", {"meeting_id": MEETING, "body": "hi"}, meeting_id=MEETING
        )
        await server.wait_for(lambda: len(server.of_type("message")) == 3)
        await server.wait_for(lambda: not run.client.pending)
        await asyncio.sleep(0.3)
        sent = server.of_type("message")
        assert len(sent) == 3
        assert {f["id"] for f in sent} == {frame_id}


async def test_gives_up_after_max_resends(home, caplog):
    async with FakeServer() as server, Running(make_client(server, home)) as run:
        await server.authed.wait()
        server.withhold["message"] = 100
        with caplog.at_level(logging.WARNING, logger="yantraxa_bridge"):
            await run.client.send("message", {"meeting_id": MEETING, "body": "hi"})
            await server.wait_for(lambda: not run.client.pending)
        await asyncio.sleep(0.3)
        assert len(server.of_type("message")) == 4  # first send + 3 resends
        assert "giving up" in caplog.text


async def test_an_error_with_ref_id_is_final(home):
    async with FakeServer() as server, Running(make_client(server, home)) as run:
        await server.authed.wait()
        server.withhold["message"] = 100
        frame_id = await run.client.send("message", {"meeting_id": MEETING, "body": "hi"})
        await server.wait_for(lambda: len(server.of_type("message")) == 1)
        await server.push("error", {"code": "forbidden", "ref_id": frame_id})
        await server.wait_for(lambda: not run.client.pending)
        await asyncio.sleep(0.3)
        assert len(server.of_type("message")) == 1
        assert run.client.errors[-1] == ("forbidden", frame_id)


async def test_inbound_frames_are_deduped_and_important_ones_acked(home):
    invitation_id = str(uuid.uuid4())
    async with FakeServer() as server:
        with respx.mock() as mock:
            route = mock.post(f"{server.url}/v1/agent/invitations/{invitation_id}/accept").mock(
                return_value=httpx.Response(200)
            )
            async with Running(make_client(server, home, auto_accept_pending=True)):
                await server.authed.wait()
                same = str(uuid.uuid4())
                payload = invitation_payload(False, invitation_id)
                await server.push("invitation", payload, id_=same)
                await server.push("invitation", payload, id_=same)
                await server.wait_for(lambda: len(server.of_type("ack")) == 2)
                await asyncio.sleep(0.1)
            assert route.call_count == 1
            assert {f["payload"]["id"] for f in server.of_type("ack")} == {same}


async def test_pending_invitation_is_accepted_when_configured(home):
    invitation_id = str(uuid.uuid4())
    async with FakeServer() as server:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(f"{server.url}/v1/agent/invitations/{invitation_id}/accept").mock(
                return_value=httpx.Response(200, json={"status": "accepted"})
            )
            client = make_client(server, home, auto_accept_pending=True)
            async with Running(client):
                await server.authed.wait()
                await server.push("invitation", invitation_payload(False, invitation_id))
                await server.wait_for(lambda: route.called)
            request = route.calls[0].request
            assert request.headers["authorization"] == f"Bearer ag_abc234:{SECRET}"


async def test_pending_invitation_is_left_alone_by_default(home):
    async with FakeServer() as server:
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(url__regex=r".*/accept").mock(return_value=httpx.Response(200))
            async with Running(make_client(server, home)) as run:
                await server.authed.wait()
                await server.push("invitation", invitation_payload(False))
                await server.wait_for(lambda: len(run.client.invitations) == 1)
                await asyncio.sleep(0.1)
            assert not route.called


async def test_meeting_state_and_own_messages_filtered(home):
    async with FakeServer() as server, Running(make_client(server, home)) as run:
        await server.authed.wait()
        await server.push("meeting_joined", joined_payload(), meeting_id=MEETING)
        await server.push("message_posted", {"message": message_row(SEAT, "mine")})
        await server.push("message_posted", {"message": message_row(OTHER_SEAT, "theirs")})
        await server.wait_for(
            lambda: MEETING in run.client.meetings and run.client.meetings[MEETING].recent
        )
        await asyncio.sleep(0.05)
        state = run.client.meetings[MEETING]
        assert state.code == "AB-1234"
        assert state.seat_id == SEAT
        assert [m.body for m in state.recent] == ["theirs"]


async def test_summary_written_only_when_profile_allows(home):
    async with FakeServer() as server, Running(make_client(server, home)) as run:
        await server.authed.wait()
        await server.push("meeting_joined", joined_payload(), meeting_id=MEETING)
        await server.push("summary", {"meeting_id": MEETING, "markdown": "# Nope"})
        await server.wait_for(lambda: len(server.of_type("ack")) == 2)
        assert not (home / "summaries").exists()

        allow = joined_payload()["knowledge_profile"] | {"return_summaries_to_agent": True}
        await server.push("meeting_joined", joined_payload(allow), meeting_id=MEETING)
        await server.push("summary", {"meeting_id": MEETING, "markdown": "# Notes\n- done"})
        path = home / "summaries" / "AB-1234.md"
        await server.wait_for(path.exists)
        assert path.read_text() == "# Notes\n- done"
        assert run.client.meetings[MEETING].code == "AB-1234"


async def test_summary_carries_the_representation_review(home):
    async with FakeServer() as server, Running(make_client(server, home)):
        await server.authed.wait()
        allow = joined_payload()["knowledge_profile"] | {"return_summaries_to_agent": True}
        await server.push("meeting_joined", joined_payload(allow), meeting_id=MEETING)
        summary = {
            "meeting_id": MEETING,
            "code": "AB-1234",
            "version": 2,
            "author_kind": "system",
            "markdown": "# Notes\n- done\n",
            "document": {"schema_version": 1},
            "representation_review": {
                "rating": "partially",
                "correction": "I'd only pay $300.\nAsk me first next time.",
            },
        }
        await server.push("summary", summary)
        path = home / "summaries" / "AB-1234.md"
        await server.wait_for(path.exists)
        assert path.read_text() == (
            "# Notes\n- done\n\n## Representation review\n\n"
            "Did the agent represent its owner? Partially.\n\n"
            "Owner's correction:\n\n> I'd only pay $300.\n> Ask me first next time.\n"
        )


async def test_summary_without_a_review_is_just_the_markdown(home):
    async with FakeServer() as server, Running(make_client(server, home)):
        await server.authed.wait()
        allow = joined_payload()["knowledge_profile"] | {"return_summaries_to_agent": True}
        await server.push("meeting_joined", joined_payload(allow), meeting_id=MEETING)
        await server.push(
            "summary",
            {"meeting_id": MEETING, "markdown": "# Notes", "representation_review": None},
        )
        path = home / "summaries" / "AB-1234.md"
        await server.wait_for(path.exists)
        assert path.read_text() == "# Notes"


async def test_summary_for_a_meeting_the_bridge_forgot_is_kept_under_its_code(home):
    # The server only sends allowed summaries; after a restart the bridge has no state.
    async with FakeServer() as server, Running(make_client(server, home)):
        await server.authed.wait()
        await server.push(
            "summary", {"meeting_id": MEETING, "code": "AB-1234", "markdown": "# Later"}
        )
        path = home / "summaries" / "AB-1234.md"
        await server.wait_for(path.exists)
        assert path.read_text() == "# Later"
        await server.push("summary", {"meeting_id": MEETING, "code": "../etc", "markdown": "# Odd"})
        odd = home / "summaries" / f"{MEETING}.md"
        await server.wait_for(odd.exists)
        assert odd.read_text() == "# Odd"


async def test_default_turn_handler_passes(home):
    async with FakeServer() as server, Running(make_client(server, home)) as run:
        await server.authed.wait()
        await server.push("meeting_joined", joined_payload(), meeting_id=MEETING)
        turn_id = str(uuid.uuid4())
        await server.push(
            "turn_request", {"meeting_id": MEETING, "turn_id": turn_id}, meeting_id=MEETING
        )
        await server.wait_for(lambda: server.of_type("turn_response"))
        response = server.of_type("turn_response")[0]
        assert response["meeting_id"] == MEETING
        assert response["payload"] == {
            "meeting_id": MEETING,
            "turn_id": turn_id,
            "actions": [{"action": "pass"}],
        }
        assert isinstance(run.client.turn_handler, PassTurnHandler)


class SlowHandler:
    def __init__(self, delay: float):
        self.delay = delay
        self.calls = 0

    async def take_turn(self, request, io: TurnIO) -> list[TurnAction]:
        self.calls += 1
        await io.activity("preparing")
        await io.chunk("thinking out loud")
        await asyncio.sleep(self.delay)
        return [TurnAction(action="message", text="done")]


async def test_turn_io_and_duplicate_turns(home):
    async with FakeServer() as server:
        client = make_client(server, home)
        handler = SlowHandler(0.2)
        client.turn_handler = handler
        async with Running(client):
            await server.authed.wait()
            turn_id = str(uuid.uuid4())
            for _ in range(2):  # a resent turn_request carries a new frame id
                await server.push(
                    "turn_request", {"meeting_id": MEETING, "turn_id": turn_id}, meeting_id=MEETING
                )
            await server.wait_for(lambda: server.of_type("turn_response"))
            await asyncio.sleep(0.1)
        assert handler.calls == 1
        assert len(server.of_type("turn_response")) == 1
        assert server.of_type("activity")[0]["payload"] == {
            "meeting_id": MEETING,
            "state": "preparing",
        }
        chunk = server.of_type("turn_chunk")[0]["payload"]
        assert chunk == {"meeting_id": MEETING, "turn_id": turn_id, "text": "thinking out loud"}
        actions = server.of_type("turn_response")[0]["payload"]["actions"]
        assert actions == [{"action": "message", "text": "done"}]


async def test_turn_timeout_passes(home):
    async with FakeServer() as server:
        client = make_client(server, home, timeout_s=0.1)
        client.turn_handler = SlowHandler(10)
        async with Running(client):
            await server.authed.wait()
            await server.push("turn_request", {"meeting_id": MEETING, "turn_id": str(uuid.uuid4())})
            await server.wait_for(lambda: server.of_type("turn_response"))
        actions = server.of_type("turn_response")[0]["payload"]["actions"]
        assert actions == [{"action": "pass"}]


def agent_me(server: FakeServer, status: int):
    """`GET /v1/agent/me` on the fake server's address, answering `status`."""
    mock = respx.mock(assert_all_called=False)
    route = mock.get(f"{server.url}/v1/agent/me")
    if status == 200:
        route.mock(return_value=httpx.Response(200, json={"agent": AGENT, "permissions": {}}))
    else:
        route.mock(return_value=httpx.Response(status, json={"error": {"code": "unauthorized"}}))
    return mock, route


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        (1000, StopReason.DISCONNECTED),
        (4001, StopReason.UNAUTHORIZED),
        (4009, StopReason.SUPERSEDED),
    ],
)
async def test_final_close_codes_stop_the_bridge(home, code, reason):
    async with FakeServer() as server:
        mock, _ = agent_me(server, 401)
        with mock:
            run = Running(make_client(server, home))
            await run.__aenter__()
            await server.authed.wait()
            assert server.ws is not None
            await server.ws.close(code, "bye")
            assert await run.result() == reason
        assert server.connections == 1


async def test_unauthorized_at_auth_stops_once_the_credential_check_says_401(home):
    async with FakeServer() as server:
        server.auth_plan = [4001]
        mock, route = agent_me(server, 401)
        with mock:
            run = Running(make_client(server, home))
            await run.__aenter__()
            assert await run.result() == StopReason.UNAUTHORIZED
            assert route.call_count == 1
            authorization = route.calls[0].request.headers["Authorization"]
        assert authorization == f"Bearer ag_abc234:{SECRET}"


async def test_a_4001_with_a_valid_credential_is_retried(home, caplog):
    """The owner's bridge stopped for good on 2026-10-15 after one 4001 that wasn't a verdict
    (the old 5 s auth timer): now the credential is checked, and a 200 means reconnect."""
    errors: list[str] = []
    async with FakeServer() as server:
        server.auth_plan = [4001, 4001]
        mock, route = agent_me(server, 200)
        with mock, caplog.at_level(logging.WARNING, logger="yantraxa_bridge"):
            client = make_client(server, home)
            client._on_error = errors.append
            async with Running(client) as run:
                await server.authed.wait()
                assert server.connections == 3
                assert not run.task.done()
            assert route.call_count == 2
    assert "the server closed with 4001 but the credential is valid; retrying" in caplog.text
    assert errors.count("the server closed with 4001 but the credential is valid; retrying") == 2
    # Auth is still the first frame of every connection.
    assert [f["type"] for f in server.auth_frames] == ["auth", "auth", "auth"]


@pytest.mark.parametrize("answer", ["network", 500, 403])
async def test_a_4001_whose_credential_check_is_not_a_401_is_retried(home, answer):
    async with FakeServer() as server:
        server.auth_plan = [4001]
        mock = respx.mock(assert_all_called=False)
        route = mock.get(f"{server.url}/v1/agent/me")
        if answer == "network":
            route.mock(side_effect=httpx.ConnectError("no route"))
        else:
            route.mock(return_value=httpx.Response(answer))
        with mock:
            async with Running(make_client(server, home)):
                await server.authed.wait()
                assert server.connections == 2


@pytest.mark.parametrize("code", [4008, 4000, 4011, 4999])
async def test_auth_incomplete_and_unknown_codes_reconnect(home, code):
    errors: list[str] = []
    async with FakeServer() as server:
        server.auth_plan = [code]
        client = make_client(server, home)
        client._on_error = errors.append
        async with Running(client):
            await server.authed.wait()
            assert server.connections == 2
    if code == 4008:
        assert errors == ["the server closed before authentication finished (4008); reconnecting"]
    else:
        assert errors == [f"connection closed ({code}); reconnecting"]


async def test_hooks_hear_each_connection_and_each_failure(home):
    seen: list = []
    async with FakeServer() as server:
        server.auth_plan = [4003]
        cfg = BridgeConfig(server_url=server.url, adapter="ollama")
        client = BridgeClient(
            cfg,
            Credentials(agent_id="ag_abc234", secret=SECRET),
            home=home,
            timing=FAST,
            on_connected=lambda agent: seen.append(("connected", agent["handle"])),
            on_error=lambda reason: seen.append(("error", reason)),
        )
        async with Running(client):
            await server.authed.wait()
            await server.wait_for(lambda: len(seen) == 2)
    assert seen == [
        ("error", "agent is paused by its owner; retrying with backoff"),
        ("connected", "saugat.ai"),
    ]


def test_the_bridge_waits_longer_for_auth_ok_than_the_server_waits_for_auth():
    from yantraxa_bridge.protocol import SERVER_AUTH_TIMEOUT_S

    assert Timing().auth_timeout_s == 20
    assert Timing().auth_timeout_s > SERVER_AUTH_TIMEOUT_S


async def test_paused_backs_off_and_retries(home):
    async with FakeServer() as server:
        server.auth_plan = [4003, 4003]
        async with Running(make_client(server, home)):
            await server.authed.wait()
            assert server.connections == 3


async def test_reconnect_resends_unacked_frames_with_the_same_id(home):
    async with FakeServer() as server, Running(make_client(server, home)) as run:
        await server.authed.wait()
        server.withhold["message"] = 1
        frame_id = await run.client.send("message", {"meeting_id": MEETING, "body": "hi"})
        await server.wait_for(lambda: len(server.of_type("message")) == 1)
        server.authed.clear()
        assert server.ws is not None
        await server.ws.close(1011, "crash")
        await server.authed.wait()
        await server.wait_for(lambda: not run.client.pending)
        assert server.connections == 2
        assert {f["id"] for f in server.of_type("message")} == {frame_id}


async def test_missing_pongs_force_a_reconnect(home):
    async with FakeServer() as server, Running(make_client(server, home)):
        await server.authed.wait()
        server.mute_pong = True
        await server.wait_for(lambda: server.connections >= 2)


async def test_unreachable_server_keeps_retrying(home):
    async with FakeServer() as server:
        port = server.url
    client = BridgeClient(
        BridgeConfig(server_url=port, adapter="ollama"),
        Credentials(agent_id="ag_abc234", secret=SECRET),
        home=home,
        timing=FAST,
    )
    async with Running(client) as run:
        await asyncio.sleep(0.2)
        assert not run.task.done()
        assert client.attempts >= 2


async def test_garbage_frames_are_ignored(home):
    async with FakeServer() as server, Running(make_client(server, home)) as run:
        await server.authed.wait()
        await server.push_raw("not json at all")
        await server.push_raw(frame("teleport", {}))
        await server.push("meeting_joined", joined_payload(), meeting_id=MEETING)
        await server.wait_for(lambda: MEETING in run.client.meetings)
        assert server.connections == 1


async def test_secret_never_reaches_the_logs(home, caplog):
    with caplog.at_level(logging.DEBUG, logger="yantraxa_bridge"):
        async with FakeServer() as server, Running(make_client(server, home)) as run:
            await server.authed.wait()
            await run.client.send("message", {"meeting_id": MEETING, "body": "hi"})
            await server.wait_for(lambda: not run.client.pending)
            mock, _ = agent_me(server, 401)
            with mock:
                await server.ws.close(4001, "revoked")
                await run.result()
    assert SECRET not in caplog.text


def test_backoff_grows_caps_and_resets():
    backoff = Backoff(1, 60, rng=lambda: 0.5)
    delays = [backoff.next() for _ in range(9)]
    assert delays[:7] == [1, 2, 4, 8, 16, 32, 60]
    assert delays[7:] == [60, 60]
    backoff.reset()
    assert backoff.next() == 1


def test_backoff_jitter_bounds():
    low = Backoff(1, 60, rng=lambda: 0.0)
    high = Backoff(1, 60, rng=lambda: 0.999999)
    for _ in range(10):
        lo, hi = low.next(), high.next()
    assert lo == pytest.approx(48)
    assert hi == pytest.approx(72, rel=1e-3)


async def test_rate_limiter_waits_for_the_window():
    limiter = RateLimiter(3, 0.2)
    loop = asyncio.get_running_loop()
    start = loop.time()
    for _ in range(4):
        await limiter.acquire()
    assert loop.time() - start >= 0.19


def test_frames_are_json_serialisable():
    json.dumps(json.loads(frame("pong", {})))
