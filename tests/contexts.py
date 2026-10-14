"""A small, fixed meeting context shared by the adapter tests."""

from yantraxa_bridge.adapters.base import (
    AgendaItem,
    Briefing,
    BriefingAgendaItem,
    BriefingPerson,
    ContextMessage,
    Knowledge,
    Me,
    MeetingInfo,
    OpenMotion,
    OpenProposal,
    PreviousSummary,
    TurnContext,
)
from yantraxa_bridge.protocol import AgendaEntry

MEETING_ID = "3f2c6a5e-8a0e-4f7a-9d55-0d3c2a1b4c5d"
TURN_ID = "0b6f3c1e-5a4d-4e2f-9c8b-7a6d5e4f3a2b"
SEAT = "6b1e9d2a-1c3f-4e5a-8b7c-9d0e1f2a3b4c"
OTHER_SEAT = "7c2f0e3b-2d4a-4f6b-9c8d-0e1f2a3b4c5d"
M1 = "11111111-1111-4111-8111-111111111111"
M2 = "22222222-2222-4222-8222-222222222222"
P1 = "33333333-3333-4333-8333-333333333333"
PM1 = "44444444-4444-4444-8444-444444444444"  # the message that proposed P1
V1 = "55555555-5555-4555-8555-555555555555"  # the motion opened on it


def make_motion(**overrides) -> OpenMotion:
    fields = {
        "id": V1,
        "proposal_message_id": PM1,
        "text": "Spend $400 on a new shed roof",
        "closes_at": "2026-10-11T09:10:00.000Z",
    }
    fields.update(overrides)
    return OpenMotion(**fields)


def make_briefing(**overrides) -> Briefing:
    fields = {
        "purpose": "Agree what the garden spends this spring",
        "agenda": [
            BriefingAgendaItem(position=0, title="Tool shed repairs", notes="Quotes attached"),
            BriefingAgendaItem(position=1, title="Seed order", notes=""),
        ],
        "organiser": BriefingPerson(handle="priya", display_name="Priya"),
        "participants": [
            BriefingPerson(kind="human", handle="priya", display_name="Priya"),
            BriefingPerson(kind="agent", handle="saugat.ai", display_name="Saugat's agent"),
        ],
        "previous_summaries": [
            PreviousSummary(
                code="AB-1000",
                title="Winter wrap-up",
                completed_at="2026-09-01T10:00:00.000Z",
                overview="The shed roof was flagged as leaking.",
            )
        ],
    }
    fields.update(overrides)
    return Briefing(**fields)


def make_context(**overrides) -> TurnContext:
    fields = {
        "turn_id": TURN_ID,
        "me": Me(handle="saugat.ai", display_name="Saugat's agent", seat_id=SEAT),
        "meeting": MeetingInfo(
            id=MEETING_ID,
            code="AB-1234",
            title="Community garden spring budget",
            purpose="Agree what the garden spends this spring",
            agenda=[
                AgendaEntry(position=1, title="Tool shed repairs"),
                AgendaEntry(position=2, title="Seed order"),
            ],
        ),
        "agenda_item": AgendaItem(position=1, title="Tool shed repairs"),
        "recent_messages": [
            ContextMessage(
                id=M1,
                seat_id=OTHER_SEAT,
                author="Priya",
                kind="message",
                body="The shed roof leaks onto the seed trays.",
                reply_to=None,
                created_at="2026-10-11T09:00:00.000Z",
            ),
            ContextMessage(
                id=M2,
                seat_id=OTHER_SEAT,
                author="Priya",
                kind="question",
                body="Can we afford $400 for a new roof this month?",
                reply_to=None,
                created_at="2026-10-11T09:01:00.000Z",
            ),
        ],
        "open_proposals": [
            OpenProposal(id=P1, text="Spend $400 on a new shed roof", author="Priya")
        ],
        "knowledge": Knowledge(
            approved_facts=["Saugat can volunteer on Saturday mornings."],
            restricted_categories=["health", "personal finances"],
            ask_when_uncertain=True,
        ),
        "deadline_at": "2026-10-11T09:02:40.000Z",
        "max_output_chars": 600,
    }
    fields.update(overrides)
    return TurnContext(**fields)
