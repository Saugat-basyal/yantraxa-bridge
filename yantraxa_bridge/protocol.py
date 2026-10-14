"""Frames on `/ws/agent` (spec §5).

Mirrors `packages/shared/src/agent-transport.ts`; change both together. Every frame,
both ways, is the envelope `{ id, type, meeting_id?, ts, payload }`.

Bridge → server: `auth` is answered by `auth_ok` (or a close), `ping` by `pong`, and
`ack` by nothing. Every other bridge frame gets `ack { id }` once handled, with an
`error { code, ref_id }` before it when handling failed, so the bridge stops resending
either way.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
)

# Close codes the gateway uses, and what the bridge does on each (see client.py).
CLOSE_NORMAL = 1000  # the owner disconnected the agent: stop
CLOSE_UNAUTHORIZED = 4001  # bad, revoked or missing credential: stop, run diagnose
CLOSE_AGENT_PAUSED = 4003  # paused by the owner: back off and retry
CLOSE_SUPERSEDED = 4009  # a newer bridge authenticated as this agent: stop

HEARTBEAT_INTERVAL_S = 20
MAX_FRAME_BYTES = 64 * 1024
MAX_MESSAGE_BYTES = 8 * 1024
RATE_LIMIT_FRAMES = 30
RATE_LIMIT_WINDOW_S = 10
DEDUPE_IDS = 1_000

ADAPTER_TYPES = (
    "ollama",
    "personal_agent",
    "demo",
    "interactive_mcp",
    "claude_code",
    "codex_cli",
)
AGENT_ID_PATTERN = r"^ag_[a-z2-7]{6}$"
SECRET_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"
MEETING_CODE_PATTERN = r"^[A-Z]{2}-\d{4}$"

ActivityState = Literal["reading_agenda", "preparing", "responding"]
ActionName = Literal["message", "question", "reply", "proposal", "pass"]
# Every action, in the order the prompt lists them. `allowed_actions` defaults to all of
# them, for servers that don't send it.
ACTION_NAMES: tuple[ActionName, ...] = ("message", "question", "reply", "proposal", "pass")
PostableKind = Literal["message", "question", "reply", "proposal"]
ErrorCode = Literal[
    "bad_request",
    "rate_limited",
    "agent_mismatch",
    "forbidden",
    "permission_denied",
    "not_found",
    "meeting_closed",
    "not_started",
    "message_too_large",
    "duplicate_id",
    "no_turn",
    "internal_error",
]


def _check_uuid(value: str) -> str:
    try:
        uuid.UUID(value)
    except ValueError as error:
        raise ValueError("not a uuid") from error
    return value


Uuid = Annotated[str, AfterValidator(_check_uuid)]


class ProtocolError(ValueError):
    """A frame that doesn't match the protocol."""


def adapter_type_wire(adapter: str) -> str:
    """CLI adapter names use dashes; the server's `adapter_type` uses underscores."""
    return adapter.replace("-", "_")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ bridge → server


class _Named(_Strict):
    """Any payload may name an `agent_id`; the server refuses another agent's."""

    agent_id: str | None = Field(default=None, max_length=64)


class AuthPayload(_Strict):
    agent_id: str = Field(pattern=AGENT_ID_PATTERN)
    secret: str = Field(pattern=SECRET_PATTERN, repr=False)
    bridge_version: str = Field(min_length=1, max_length=40)
    adapter_type: Literal[
        "ollama", "personal_agent", "demo", "interactive_mcp", "claude_code", "codex_cli"
    ]


class PingPayload(_Named):
    pass


class ActivityPayload(_Named):
    meeting_id: Uuid
    state: ActivityState


class TurnChunkPayload(_Named):
    meeting_id: Uuid
    turn_id: Uuid
    text: str = Field(max_length=8192)


class Vote(_Strict):
    """An advisory vote. `motion_id` is the motion's id or its proposal message's id.

    `choice` is `support`, `oppose` or `abstain`. The server also takes `yes`/`no` and
    records anything else as an `unknown_choice` rejection, so it is kept as a string.
    """

    motion_id: Uuid
    choice: str = Field(min_length=1, max_length=40)
    reason: str | None = Field(default=None, max_length=2000)


class TurnAction(_Strict):
    """One step of a turn's answer; adapters (Task 4) produce this shape."""

    action: ActionName
    text: str | None = Field(default=None, max_length=8192)
    reply_to: Uuid | None = None
    vote: Vote | None = None


class TurnResponsePayload(_Named):
    meeting_id: Uuid
    turn_id: Uuid
    actions: list[TurnAction] = Field(max_length=5)


class MessagePayload(_Named):
    meeting_id: Uuid
    body: str
    kind: PostableKind = "message"
    reply_to: Uuid | None = None
    agenda_item_id: Uuid | None = None

    @field_validator("body")
    @classmethod
    def _not_blank(cls, body: str) -> str:
        if not body.strip():
            raise ValueError("empty message")
        return body


class AckPayload(_Named):
    id: Uuid


BRIDGE_PAYLOADS: dict[str, type[_Strict]] = {
    "auth": AuthPayload,
    "ping": PingPayload,
    "activity": ActivityPayload,
    "turn_chunk": TurnChunkPayload,
    "turn_response": TurnResponsePayload,
    "message": MessagePayload,
    "ack": AckPayload,
}
# Frames the server acks; everything else the bridge sends is answered differently.
ACKED_BRIDGE_TYPES = frozenset({"activity", "turn_chunk", "turn_response", "message"})


def _now() -> str:
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def make_frame(
    type_: str, payload: dict[str, Any] | BaseModel, *, meeting_id: str | None = None
) -> dict[str, Any]:
    """A validated bridge frame, ready for `json.dumps`, with a fresh UUID id."""
    model = BRIDGE_PAYLOADS.get(type_)
    if model is None:
        raise ProtocolError(f"unknown bridge frame type {type_!r}")
    try:
        raw = payload.model_dump() if isinstance(payload, BaseModel) else payload
        if isinstance(raw, dict) and isinstance(raw.get("actions"), list):
            raw = {
                **raw,
                "actions": [
                    a.model_dump() if isinstance(a, BaseModel) else a for a in raw["actions"]
                ],
            }
        parsed = model.model_validate(raw)
    except ValidationError as error:
        raise ProtocolError(f"bad {type_} payload: {_summarise(error)}") from None
    frame: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "type": type_,
        "ts": _now(),
        "payload": parsed.model_dump(exclude_none=True),
    }
    if meeting_id is not None:
        frame["meeting_id"] = meeting_id
    return frame


def _summarise(error: Exception) -> str:
    """Field paths only: a validation error must never echo the secret back."""
    if isinstance(error, ValidationError):
        return ", ".join(".".join(str(p) for p in e["loc"]) or "payload" for e in error.errors())
    return str(error)


# ------------------------------------------------------------------ server → bridge


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")


class AuthOkPayload(BaseModel):
    agent: dict[str, Any]
    permissions: dict[str, Any]

    @field_validator("agent")
    @classmethod
    def _agent_shape(cls, agent: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(agent.get("id"), str) or not isinstance(agent.get("handle"), str):
            raise ValueError("agent needs id and handle")
        return agent


class ServerAckPayload(BaseModel):
    id: Uuid


class PongPayload(BaseModel):
    pass


class ErrorPayload(BaseModel):
    code: ErrorCode
    ref_id: Uuid | None = None


class AgendaEntry(BaseModel):
    position: int
    title: str


class InvitationMeeting(BaseModel):
    id: Uuid
    code: str = Field(pattern=MEETING_CODE_PATTERN)
    title: str
    purpose: str
    agenda: list[AgendaEntry]
    scheduled_for: str | None
    organiser_handle: str | None


class InvitationPayload(BaseModel):
    invitation_id: Uuid
    purpose: str
    meeting: InvitationMeeting
    auto_accepted: bool


class MeetingJoinedPayload(BaseModel):
    meeting_id: Uuid
    code: str = Field(pattern=MEETING_CODE_PATTERN)
    seat_id: Uuid
    knowledge_profile: Any = None


class TurnRequestPayload(_Open):
    """M4: the conductor's request for a turn; extra fields pass through.

    The M5 fields are optional, so an M4 server's requests still parse, and loosely
    typed, so a malformed one is ignored (adapters/context.py) instead of failing the
    frame. `model_fields_set` tells a field the server left out from one it sent as null.
    """

    meeting_id: Uuid
    turn_id: Uuid
    # M5: what the agent may be told in this meeting, resolved for every turn; null when
    # nothing. Preferred over the `meeting_joined` profile.
    knowledge: Any = None
    # M5: the pre-meeting briefing, until the seat has answered or passed a turn.
    briefing: Any = None
    # M5: `[{id, proposal_message_id, text, closes_at}]`, the motions open for a vote.
    open_motions: Any = None
    # What the seat may do this turn, from its permissions; the prompt and the answer
    # schema offer only these. Absent (an older server): every action.
    allowed_actions: list[ActionName] = Field(default_factory=lambda: list(ACTION_NAMES))

    @field_validator("allowed_actions", mode="before")
    @classmethod
    def _known_actions(cls, value: Any) -> list[str]:
        """Known actions in the usual order, plus `pass`, which a turn may always do.
        Anything that isn't a list means every action; unknown names are dropped."""
        if not isinstance(value, list):
            return list(ACTION_NAMES)
        named = {v for v in value if isinstance(v, str)}
        return [a for a in ACTION_NAMES if a in named or a == "pass"]


class MessageRow(BaseModel):
    id: str
    meeting_id: str
    seat_id: str
    agenda_item_id: str | None
    kind: Literal["message", "question", "reply", "proposal", "system"]
    reply_to: str | None
    body: str
    created_at: str


class EventRow(BaseModel):
    id: int
    meeting_id: str
    type: str
    actor_seat_id: str | None
    payload: Any = None
    created_at: str


class MessagePostedPayload(BaseModel):
    message: MessageRow


class EventPayload(BaseModel):
    event: EventRow


class SummaryPayload(_Open):
    """M5: the meeting summary; extra fields pass through."""

    meeting_id: Uuid


def _server_frame(type_: str, payload: type[BaseModel]) -> type[BaseModel]:
    return type(
        f"{type_.title().replace('_', '')}Frame",
        (BaseModel,),
        {
            "__annotations__": {
                "id": Uuid,
                "type": Literal[type_],  # type: ignore[valid-type]
                "meeting_id": Uuid | None,
                "ts": str,
                "payload": payload,
            },
            "meeting_id": None,
        },
    )


AuthOkFrame = _server_frame("auth_ok", AuthOkPayload)
AckFrame = _server_frame("ack", ServerAckPayload)
PongFrame = _server_frame("pong", PongPayload)
ErrorFrame = _server_frame("error", ErrorPayload)
InvitationFrame = _server_frame("invitation", InvitationPayload)
MeetingJoinedFrame = _server_frame("meeting_joined", MeetingJoinedPayload)
TurnRequestFrame = _server_frame("turn_request", TurnRequestPayload)
MessagePostedFrame = _server_frame("message_posted", MessagePostedPayload)
EventFrame = _server_frame("event", EventPayload)
SummaryFrame = _server_frame("summary", SummaryPayload)

ServerFrame = Annotated[
    AuthOkFrame  # type: ignore[valid-type]
    | AckFrame  # type: ignore[valid-type]
    | PongFrame  # type: ignore[valid-type]
    | ErrorFrame  # type: ignore[valid-type]
    | InvitationFrame  # type: ignore[valid-type]
    | MeetingJoinedFrame  # type: ignore[valid-type]
    | TurnRequestFrame  # type: ignore[valid-type]
    | MessagePostedFrame  # type: ignore[valid-type]
    | EventFrame  # type: ignore[valid-type]
    | SummaryFrame,  # type: ignore[valid-type]
    Field(discriminator="type"),
]
_server_adapter: TypeAdapter[Any] = TypeAdapter(ServerFrame)

# Server frames the bridge acks back. Fan-out (`message_posted`, `event`) is not acked:
# every bridge frame counts against the 30-per-10-s limit, and a busy room would spend
# it all on acks.
ACKED_SERVER_TYPES = frozenset({"invitation", "meeting_joined", "turn_request", "summary"})


def parse_server_frame(raw: str | bytes) -> Any:
    """Parse one server frame; raises ProtocolError for anything off-protocol."""
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise ProtocolError("frame is not JSON") from error
    if not isinstance(data, dict):
        raise ProtocolError("frame is not an object")
    try:
        frame = _server_adapter.validate_python(data)
    except ValidationError as error:
        raise ProtocolError(f"bad {data.get('type')!r} frame: {_summarise(error)}") from None
    return frame
