import json
import re
import uuid

import pytest

from yantraxa_bridge import protocol
from yantraxa_bridge.protocol import (
    ProtocolError,
    TurnAction,
    adapter_type_wire,
    make_frame,
    parse_server_frame,
)

MEETING = "3f2c6a5e-8a0e-4f7a-9d55-0d3c2a1b4c5d"
SEAT = "6b1e9d2a-1c3f-4e5a-8b7c-9d0e1f2a3b4c"


def server(type_, payload, **extra):
    return json.dumps(
        {
            "id": str(uuid.uuid4()),
            "type": type_,
            "ts": "2026-10-11T20:00:00.000Z",
            "payload": payload,
            **extra,
        }
    )


def test_make_frame_builds_the_envelope():
    frame = make_frame(
        "activity", {"meeting_id": MEETING, "state": "preparing"}, meeting_id=MEETING
    )
    assert uuid.UUID(frame["id"]).version == 4
    assert frame["type"] == "activity"
    assert frame["meeting_id"] == MEETING
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", frame["ts"])
    assert frame["payload"] == {"meeting_id": MEETING, "state": "preparing"}


def test_make_frame_leaves_out_meeting_id_when_none():
    assert "meeting_id" not in make_frame("ping", {})


def test_make_frame_validates_the_payload():
    with pytest.raises(ProtocolError):
        make_frame("activity", {"meeting_id": MEETING, "state": "thinking"})
    with pytest.raises(ProtocolError):
        make_frame(
            "auth",
            {"agent_id": "nope", "secret": "x", "bridge_version": "1", "adapter_type": "ollama"},
        )
    with pytest.raises(ProtocolError):
        make_frame("message", {"meeting_id": MEETING, "body": "   "})
    with pytest.raises(ProtocolError):
        make_frame("teleport", {})


def test_make_frame_drops_unset_optionals():
    frame = make_frame("message", {"meeting_id": MEETING, "body": "hi"})
    assert frame["payload"] == {"meeting_id": MEETING, "body": "hi", "kind": "message"}


def test_auth_frame_accepts_a_real_credential():
    frame = make_frame(
        "auth",
        {
            "agent_id": "ag_abc234",
            "secret": "s3cr3t_-X",
            "bridge_version": "0.1.0",
            "adapter_type": "personal_agent",
        },
    )
    assert frame["payload"]["agent_id"] == "ag_abc234"


def test_turn_response_carries_actions():
    frame = make_frame(
        "turn_response",
        {
            "meeting_id": MEETING,
            "turn_id": str(uuid.uuid4()),
            "actions": [TurnAction(action="pass")],
        },
    )
    assert frame["payload"]["actions"] == [{"action": "pass"}]


def test_parses_every_server_frame_type():
    message = {
        "id": str(uuid.uuid4()),
        "meeting_id": MEETING,
        "seat_id": SEAT,
        "agenda_item_id": None,
        "kind": "message",
        "reply_to": None,
        "body": "hello",
        "created_at": "2026-10-11T20:00:00Z",
    }
    event = {
        "id": 7,
        "meeting_id": MEETING,
        "type": "agent_activity",
        "actor_seat_id": SEAT,
        "payload": {"state": "preparing"},
        "created_at": "2026-10-11T20:00:00Z",
    }
    invitation = {
        "invitation_id": str(uuid.uuid4()),
        "purpose": "Join us",
        "meeting": {
            "id": MEETING,
            "code": "AB-1234",
            "title": "Plan",
            "purpose": "Plan Q4",
            "agenda": [{"position": 1, "title": "Budget"}],
            "scheduled_for": None,
            "organiser_handle": "saugat",
        },
        "auto_accepted": False,
    }
    frames = {
        "auth_ok": {
            "agent": {"id": "ag_abc234", "handle": "saugat.ai", "state": "active"},
            "permissions": {"post_messages": True},
        },
        "ack": {"id": str(uuid.uuid4())},
        "pong": {},
        "error": {"code": "forbidden", "ref_id": str(uuid.uuid4())},
        "invitation": invitation,
        "meeting_joined": {
            "meeting_id": MEETING,
            "code": "AB-1234",
            "seat_id": SEAT,
            "knowledge_profile": {"approved_facts": []},
        },
        "turn_request": {"meeting_id": MEETING, "turn_id": str(uuid.uuid4()), "deadline": "x"},
        "message_posted": {"message": message},
        "event": {"event": event},
        "summary": {"meeting_id": MEETING, "markdown": "# Notes"},
    }
    for type_, payload in frames.items():
        frame = parse_server_frame(server(type_, payload))
        assert frame.type == type_

    turn = parse_server_frame(server("turn_request", frames["turn_request"]))
    assert turn.payload.model_extra == {"deadline": "x"}
    joined = parse_server_frame(server("meeting_joined", frames["meeting_joined"]))
    assert joined.payload.seat_id == SEAT


def test_parse_rejects_bad_frames():
    for raw in [
        "not json",
        json.dumps({"type": "pong", "payload": {}}),
        server("teleport", {}),
        server("error", {"code": "made_up"}),
        server("ack", {"id": "not-a-uuid"}),
        b"\x00\x01",
    ]:
        with pytest.raises(ProtocolError):
            parse_server_frame(raw)


def test_parse_accepts_the_not_started_error_code():
    frame = parse_server_frame(
        server("error", {"code": "not_started", "ref_id": str(uuid.uuid4())})
    )
    assert frame.payload.code == "not_started"


def test_unknown_auth_ok_fields_pass_through():
    frame = parse_server_frame(
        server(
            "auth_ok", {"agent": {"id": "ag_abc234", "handle": "x", "bio": "hi"}, "permissions": {}}
        )
    )
    assert frame.payload.agent["bio"] == "hi"


def test_adapter_names_map_to_the_wire():
    assert adapter_type_wire("ollama") == "ollama"
    assert adapter_type_wire("personal-agent") == "personal_agent"
    assert adapter_type_wire("interactive-mcp") == "interactive_mcp"


def test_constants_mirror_the_server():
    assert protocol.CLOSE_UNAUTHORIZED == 4001
    assert protocol.CLOSE_AGENT_PAUSED == 4003
    assert protocol.CLOSE_SUPERSEDED == 4009
    assert protocol.CLOSE_NORMAL == 1000
    assert protocol.HEARTBEAT_INTERVAL_S == 20
    assert protocol.MAX_FRAME_BYTES == 64 * 1024
    assert protocol.RATE_LIMIT_FRAMES == 30
    assert protocol.RATE_LIMIT_WINDOW_S == 10
