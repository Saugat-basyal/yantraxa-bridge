"""A `TurnContext` from a `turn_request` and what the bridge knows about the meeting.

The conductor's `turn_request` arrives in M4; today its payload is only guaranteed to
carry `meeting_id` and `turn_id`. These optional extras are used when present and
well-formed, and ignored otherwise:

- `agenda_item`: `{id?, position?, title}`
- `participants`: `[{seat_id, display_name?, handle?}]`, for naming message authors
- `recent_messages`: message rows, instead of the ones the bridge collected
- `open_proposals`: `[{id, text, author?, message_id?}]`; without it, proposal
  messages the bridge has seen count as open, with their message id as the motion id
- `deadline_at`, `max_output_chars`

M5 adds three, also optional (`TurnRequestPayload` declares them):

- `knowledge`: the knowledge resolved for this turn. When the server sends it (null
  included, meaning "nothing to share") it replaces the `meeting_joined` profile, which
  can be stale by the time a turn comes; only an M4 server's requests fall back to it.
- `briefing`: purpose, agenda, organiser, participants and earlier summaries, sent until
  the seat has answered or passed a turn.
- `open_motions`: `[{id, proposal_message_id, text, closes_at}]`. When sent, votes name
  these motions; otherwise (M4) they name open proposals as before.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import ValidationError

from ..client import MeetingState
from ..protocol import MessageRow, TurnRequestPayload
from .base import (
    DEFAULT_MAX_OUTPUT_CHARS,
    MAX_TEXT_CHARS,
    AgendaItem,
    Briefing,
    ContextMessage,
    Knowledge,
    Me,
    MeetingInfo,
    OpenMotion,
    OpenProposal,
    TurnContext,
)


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def _agenda_item(value: Any) -> AgendaItem | None:
    if not isinstance(value, dict):
        return None
    try:
        return AgendaItem.model_validate(value)
    except ValidationError:
        return None


def _names(value: Any) -> dict[str, str]:
    if not isinstance(value, list):
        return {}
    names = {}
    for entry in value:
        if not isinstance(entry, dict) or not isinstance(entry.get("seat_id"), str):
            continue
        name = entry.get("display_name") or (
            f"@{entry['handle']}" if isinstance(entry.get("handle"), str) else None
        )
        if isinstance(name, str) and name.strip():
            names[entry["seat_id"]] = name.strip()
    return names


def _rows(value: Any, fallback: list[MessageRow]) -> list[MessageRow]:
    if not isinstance(value, list):
        return fallback
    try:
        return [MessageRow.model_validate(row) for row in value]
    except ValidationError:
        return fallback


def _proposals(value: Any, messages: list[ContextMessage]) -> list[OpenProposal]:
    if not isinstance(value, list):
        return [
            OpenProposal(id=m.id, text=m.body, author=m.author, message_id=m.id)
            for m in messages
            if m.kind == "proposal"
        ]
    proposals = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        entry = {**entry}
        if "text" not in entry and isinstance(entry.get("body"), str):
            entry["text"] = entry.pop("body")
        try:
            proposals.append(
                OpenProposal.model_validate(
                    {k: entry[k] for k in ("id", "text", "author", "message_id") if k in entry}
                )
            )
        except ValidationError:
            continue
    return proposals


def _motions(value: Any) -> list[OpenMotion] | None:
    if not isinstance(value, list):
        return None
    motions = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        try:
            motions.append(
                OpenMotion.model_validate(
                    {
                        k: entry[k]
                        for k in ("id", "text", "proposal_message_id", "closes_at")
                        if k in entry
                    }
                )
            )
        except ValidationError:
            continue
    return motions


def _briefing(value: Any) -> Briefing | None:
    if not isinstance(value, dict):
        return None
    try:
        return Briefing.model_validate(value)
    except ValidationError:
        return None


def _knowledge(request: TurnRequestPayload, meeting: MeetingState) -> Knowledge:
    if "knowledge" in request.model_fields_set:
        return Knowledge.from_profile(request.knowledge)
    return Knowledge.from_profile(meeting.knowledge_profile)


def build_context(
    request: TurnRequestPayload,
    meeting: MeetingState,
    client: Any,
    *,
    timeout_s: float,
    now: datetime | None = None,
) -> TurnContext:
    """`client` is the `BridgeClient` (its `agent` and `invitations` are read)."""
    extra = request.model_extra or {}
    agent = getattr(client, "agent", None) or {}
    invited = next(
        (
            invitation.meeting
            for invitation in (getattr(client, "invitations", None) or {}).values()
            if invitation.meeting.id == meeting.meeting_id
        ),
        None,
    )
    names = _names(extra.get("participants"))

    def author(seat_id: str) -> str:
        if seat_id == meeting.seat_id:
            return "you"
        return names.get(seat_id) or f"participant {seat_id[:8]}"

    messages = [
        ContextMessage(
            id=row.id,
            seat_id=row.seat_id,
            author=author(row.seat_id),
            kind=row.kind,
            body=row.body,
            reply_to=row.reply_to,
            created_at=row.created_at,
        )
        for row in _rows(extra.get("recent_messages"), list(meeting.recent))
    ]

    deadline = extra.get("deadline_at")
    if not isinstance(deadline, str):
        deadline = _iso((now or datetime.now(UTC)) + timedelta(seconds=timeout_s))
    limit = extra.get("max_output_chars")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        limit = DEFAULT_MAX_OUTPUT_CHARS

    return TurnContext(
        turn_id=request.turn_id,
        me=Me(
            handle=agent.get("handle"),
            display_name=agent.get("display_name"),
            seat_id=meeting.seat_id,
        ),
        meeting=MeetingInfo(
            id=meeting.meeting_id,
            code=meeting.code,
            title=invited.title if invited else None,
            purpose=invited.purpose if invited else None,
            agenda=list(invited.agenda) if invited else [],
        ),
        agenda_item=_agenda_item(extra.get("agenda_item")),
        recent_messages=messages,
        open_proposals=_proposals(extra.get("open_proposals"), messages),
        open_motions=_motions(request.open_motions),
        briefing=_briefing(request.briefing),
        knowledge=_knowledge(request, meeting),
        deadline_at=deadline,
        max_output_chars=min(limit, MAX_TEXT_CHARS),
    )
