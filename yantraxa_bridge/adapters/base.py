"""The adapter interface (spec §10): what a turn looks like to the AI behind the bridge."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, model_serializer

from ..protocol import ACTION_NAMES, ActionName, AgendaEntry, TurnAction

# The most a single action's text may hold on the wire (`turnActionSchema`).
MAX_TEXT_CHARS = 8192
DEFAULT_MAX_OUTPUT_CHARS = 1200


@dataclass(frozen=True)
class AdapterInfo:
    runtime_label: str
    capabilities: tuple[str, ...]
    streams: bool


class Me(BaseModel):
    handle: str | None = None
    display_name: str | None = None
    seat_id: str


class MeetingInfo(BaseModel):
    id: str
    code: str
    title: str | None = None
    purpose: str | None = None
    agenda: list[AgendaEntry] = Field(default_factory=list)


class AgendaItem(BaseModel):
    id: str | None = None
    position: int | None = None
    title: str


class ContextMessage(BaseModel):
    id: str
    seat_id: str
    author: str
    kind: Literal["message", "question", "reply", "proposal", "system"]
    body: str
    reply_to: str | None = None
    created_at: str


class OpenProposal(BaseModel):
    id: str  # the motion a vote names
    text: str
    author: str | None = None
    message_id: str | None = None  # the message that proposed it, when known


class OpenMotion(BaseModel):
    """M5: a motion open for this seat's advisory vote."""

    id: str  # the motion a vote names
    text: str
    proposal_message_id: str | None = None
    closes_at: str | None = None


class BriefingAgendaItem(BaseModel):
    position: int | None = None
    title: str
    notes: str = ""


class BriefingPerson(BaseModel):
    kind: str | None = None  # `human` or `agent`; absent for the organiser
    handle: str | None = None
    display_name: str | None = None


class PreviousSummary(BaseModel):
    code: str
    title: str = ""
    completed_at: str | None = None
    overview: str = ""


class Briefing(BaseModel):
    """M5: what the agent is told before its first turn of a meeting."""

    purpose: str = ""
    agenda: list[BriefingAgendaItem] = Field(default_factory=list)
    organiser: BriefingPerson | None = None
    participants: list[BriefingPerson] = Field(default_factory=list)
    previous_summaries: list[PreviousSummary] = Field(default_factory=list)


class Knowledge(BaseModel):
    """The approved part of the server's knowledge profile; nothing else reaches a model."""

    approved_facts: list[str] = Field(default_factory=list)
    restricted_categories: list[str] = Field(default_factory=list)
    ask_when_uncertain: bool = True

    @classmethod
    def from_profile(cls, profile: Any) -> Knowledge:
        if not isinstance(profile, dict):
            return cls()

        def strings(key: str) -> list[str]:
            value = profile.get(key)
            if not isinstance(value, list):
                return []
            return [v.strip() for v in value if isinstance(v, str) and v.strip()]

        ask = profile.get("ask_when_uncertain")
        return cls(
            approved_facts=strings("approved_facts"),
            restricted_categories=strings("restricted_categories"),
            ask_when_uncertain=ask if isinstance(ask, bool) else True,
        )


class TurnContext(BaseModel):
    """Everything an adapter may use for one turn. `personal_agent` receives it as JSON."""

    turn_id: str
    me: Me
    meeting: MeetingInfo
    agenda_item: AgendaItem | None = None
    recent_messages: list[ContextMessage] = Field(default_factory=list)
    open_proposals: list[OpenProposal] = Field(default_factory=list)
    # M5. None when the server didn't send `open_motions` (an M4 server): votes then
    # name open proposals instead.
    open_motions: list[OpenMotion] | None = None
    briefing: Briefing | None = None
    knowledge: Knowledge = Field(default_factory=Knowledge)
    deadline_at: str
    max_output_chars: int = Field(default=DEFAULT_MAX_OUTPUT_CHARS, gt=0, le=MAX_TEXT_CHARS)
    # What the seat may do this turn (`turn_request.allowed_actions`); always has `pass`.
    allowed_actions: list[ActionName] = Field(default_factory=lambda: list(ACTION_NAMES))

    @model_serializer(mode="wrap")
    def _drop_absent_m5(self, handler: Any) -> dict[str, Any]:
        """Leave the M5 fields out when unset, and `allowed_actions` when it is every
        action, so an M4 turn dumps as it always did (the demo parity vectors and
        `personal_agent`'s JSON depend on it)."""
        data = handler(self)
        for key in ("open_motions", "briefing"):
            if data.get(key) is None:
                data.pop(key, None)
        if list(data.get("allowed_actions") or ()) == list(ACTION_NAMES):
            data.pop("allowed_actions", None)
        return data


@dataclass(frozen=True)
class Chunk:
    """Part of the answer's text, as the model produces it (streaming adapters only)."""

    text: str


@dataclass(frozen=True)
class Final:
    """The turn's answer; an adapter yields exactly one, last."""

    actions: list[TurnAction]


TurnEvent = Chunk | Final


def pass_turn() -> Final:
    return Final([TurnAction(action="pass")])


class Adapter(Protocol):
    def describe(self) -> AdapterInfo: ...

    def take_turn(self, ctx: TurnContext) -> AsyncIterator[TurnEvent]: ...

    async def aclose(self) -> None: ...
