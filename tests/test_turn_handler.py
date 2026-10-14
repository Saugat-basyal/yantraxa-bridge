import asyncio
import uuid
from collections import deque
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from yantraxa_bridge.adapters import make_turn_handler
from yantraxa_bridge.adapters.base import AdapterInfo, Chunk, Final
from yantraxa_bridge.adapters.context import build_context
from yantraxa_bridge.adapters.demo import DemoAdapter
from yantraxa_bridge.adapters.handler import AdapterTurnHandler
from yantraxa_bridge.adapters.ollama import OllamaAdapter
from yantraxa_bridge.adapters.personal_agent import PersonalAgentAdapter
from yantraxa_bridge.client import BridgeClient, MeetingState, PassTurnHandler
from yantraxa_bridge.config import BridgeConfig, ConfigError
from yantraxa_bridge.credentials import Credentials
from yantraxa_bridge.protocol import (
    InvitationMeeting,
    MessageRow,
    TurnAction,
    TurnRequestPayload,
)
from yantraxa_bridge.redaction import REDACTED, Redactor

from .fake_server import FakeServer
from .test_client import FAST, Running

MEETING = "3f2c6a5e-8a0e-4f7a-9d55-0d3c2a1b4c5d"
SEAT = "6b1e9d2a-1c3f-4e5a-8b7c-9d0e1f2a3b4c"
OTHER = "7c2f0e3b-2d4a-4f6b-9c8d-0e1f2a3b4c5d"
KEY = "sk-abcdefghijklmnopqrstuvwxyz123456"

PROFILE = {
    "approved_facts": ["Saugat can volunteer on Saturdays."],
    "restricted_categories": ["health"],
    "ask_when_uncertain": True,
    "return_summaries_to_agent": False,
}


def row(body, kind="message", seat=OTHER, id_=None):
    return MessageRow(
        id=id_ or str(uuid.uuid4()),
        meeting_id=MEETING,
        seat_id=seat,
        agenda_item_id=None,
        kind=kind,
        reply_to=None,
        body=body,
        created_at="2026-10-11T09:00:00.000Z",
    )


def meeting_state(messages=()):
    state = MeetingState(MEETING, "AB-1234", SEAT, PROFILE)
    state.recent = deque(messages, maxlen=50)
    return state


def invitation_meeting():
    return InvitationMeeting(
        id=MEETING,
        code="AB-1234",
        title="Garden budget",
        purpose="Spring spend",
        agenda=[{"position": 1, "title": "Tool shed"}, {"position": 2, "title": "Seeds"}],
        scheduled_for=None,
        organiser_handle="priya",
    )


class FakeIO:
    def __init__(self, meeting=None, **extra):
        self.request = TurnRequestPayload(meeting_id=MEETING, turn_id=str(uuid.uuid4()), **extra)
        self.meeting = meeting
        self.client = SimpleNamespace(
            agent={"id": "ag_abc234", "handle": "saugat.ai", "display_name": "Saugat's agent"},
            invitations={
                "inv": SimpleNamespace(meeting=invitation_meeting()),
            },
        )
        self.sent: list[tuple[str, str]] = []

    async def activity(self, state):
        self.sent.append(("activity", state))

    async def chunk(self, text):
        self.sent.append(("chunk", text))

    @property
    def chunks(self):
        return [text for kind, text in self.sent if kind == "chunk"]

    @property
    def activities(self):
        return [state for kind, state in self.sent if kind == "activity"]


class Scripted:
    """An adapter that replays fixed events and records the context it got."""

    def __init__(self, *events, streams=True):
        self.events = events
        self.streams = streams
        self.contexts = []

    def describe(self):
        return AdapterInfo("Scripted", ("message",), self.streams)

    async def take_turn(self, ctx):
        self.contexts.append(ctx)
        for event in self.events:
            yield event

    async def aclose(self):
        pass


def handler(adapter, **kw):
    kw.setdefault("chunk_interval_s", 0)
    return AdapterTurnHandler(adapter, Redactor(), timeout_s=40, **kw)


def say(text, action="message"):
    return Final([TurnAction(action=action, text=text)])


async def run_turn(adapter, io=None, **kw):
    io = io or FakeIO(meeting_state([row("How do we fix the shed?", "question")]))
    actions = await handler(adapter, **kw).take_turn(io.request, io)
    return io, actions


# ------------------------------------------------------------------ activity and streaming


async def test_activity_order_and_streamed_chunks():
    io, actions = await run_turn(
        Scripted(
            Chunk("Borrow "), Chunk("ladders "), Chunk("Saturday."), say("Borrow ladders Saturday.")
        )
    )
    assert io.activities == ["reading_agenda", "preparing", "responding"]
    first_chunk = io.sent.index(("chunk", io.chunks[0]))
    assert io.sent.index(("activity", "responding")) < first_chunk
    assert "".join(io.chunks) == "Borrow ladders Saturday."
    assert [a.model_dump(exclude_none=True) for a in actions] == [
        {"action": "message", "text": "Borrow ladders Saturday."}
    ]


async def test_a_non_streaming_answer_still_says_responding_before_it_lands():
    io, actions = await run_turn(Scripted(say("Fix the roof first."), streams=False))
    assert io.activities == ["reading_agenda", "preparing", "responding"]
    assert io.chunks == []
    assert actions[0].text == "Fix the roof first."


async def test_a_pass_never_claims_to_be_responding():
    io, actions = await run_turn(Scripted(Final([TurnAction(action="pass")])))
    assert io.activities == ["reading_agenda", "preparing"]
    assert [a.action for a in actions] == ["pass"]


async def test_chunks_are_coalesced():
    pieces = [Chunk(f"w{i} ") for i in range(30)]
    text = "".join(c.text for c in pieces)
    io, _ = await run_turn(Scripted(*pieces, say(text)), chunk_interval_s=60)
    assert len(io.chunks) == 1
    assert "".join(io.chunks) == text


# ------------------------------------------------------------------ redaction


async def test_final_text_is_redacted():
    _, actions = await run_turn(
        Scripted(say(f"Use {KEY} or AKIAABCDEFGHIJKLMNOP, mail bob@example.com"), streams=False)
    )
    text = actions[0].text
    assert KEY not in text and "AKIA" not in text and "bob@example.com" not in text
    assert text.count(REDACTED) == 3


async def test_streamed_chunks_never_carry_a_key_split_across_deltas():
    io, actions = await run_turn(
        Scripted(
            Chunk("the key is sk-abcdefghij"),
            Chunk("klmnopqrstuvwxyz123456 and "),
            Chunk("-----BEGIN RSA "),
            Chunk("PRIVATE KEY----- done"),
            say(f"the key is {KEY} and -----BEGIN RSA PRIVATE KEY----- done"),
        )
    )
    streamed = "".join(io.chunks)
    assert "sk-" not in streamed
    assert "abcdefghij" not in streamed
    assert "PRIVATE" not in streamed
    assert streamed == f"the key is {REDACTED} and {REDACTED} done"
    assert actions[0].text == streamed


async def test_allowlisted_emails_survive():
    adapter = Scripted(say("write to team@garden.test"), streams=False)
    h = AdapterTurnHandler(
        adapter, Redactor(email_allowlist=["team@garden.test"]), timeout_s=40, chunk_interval_s=0
    )
    io = FakeIO(meeting_state())
    actions = await h.take_turn(io.request, io)
    assert actions[0].text == "write to team@garden.test"


# ------------------------------------------------------------------ malformed output


@pytest.mark.parametrize(
    "events",
    [
        (),  # the adapter ended without a final answer
        (say("   "),),
        (Final([]),),
    ],
)
async def test_unusable_answers_pass(events):
    _, actions = await run_turn(Scripted(*events))
    assert [a.action for a in actions] == ["pass"]


async def test_a_turn_for_an_unknown_meeting_passes_quietly():
    adapter = Scripted(say("hi"))
    io, actions = await run_turn(adapter, FakeIO(None))
    assert [a.action for a in actions] == ["pass"]
    assert io.sent == []
    assert adapter.contexts == []


# ------------------------------------------------------------------ context


def test_context_from_meeting_state_invitation_and_request_extras():
    proposal = row("Spend $400 on the roof", "proposal")
    question = row("Who has the key?", "question")
    mine = row("I can bring a ladder.", seat=SEAT)
    io = FakeIO(
        meeting_state([proposal, mine, question]),
        agenda_item={"id": str(uuid.uuid4()), "position": 1, "title": "Tool shed"},
        participants=[{"seat_id": OTHER, "display_name": "Priya"}],
        max_output_chars=300,
    )
    now = datetime(2026, 10, 11, 9, 0, tzinfo=UTC)
    ctx = build_context(io.request, io.meeting, io.client, timeout_s=40, now=now)
    assert ctx.meeting.title == "Garden budget"
    assert [a.title for a in ctx.meeting.agenda] == ["Tool shed", "Seeds"]
    assert ctx.agenda_item is not None and ctx.agenda_item.title == "Tool shed"
    assert [m.author for m in ctx.recent_messages] == ["Priya", "you", "Priya"]
    assert [p.id for p in ctx.open_proposals] == [proposal.id]
    assert ctx.open_proposals[0].message_id == proposal.id
    assert ctx.knowledge.approved_facts == PROFILE["approved_facts"]
    assert ctx.knowledge.restricted_categories == ["health"]
    assert ctx.me.handle == "saugat.ai"
    assert ctx.deadline_at == "2026-10-11T09:00:40.000Z"
    assert ctx.max_output_chars == 300


def test_context_tolerates_odd_extras_and_missing_invitation():
    io = FakeIO(
        meeting_state(),
        agenda_item="Tool shed",
        participants="nope",
        open_proposals=[{"bad": 1}],
        max_output_chars=10**9,
    )
    io.client.invitations = {}
    ctx = build_context(io.request, io.meeting, io.client, timeout_s=40)
    assert ctx.meeting.code == "AB-1234"
    assert ctx.meeting.title is None
    assert ctx.agenda_item is None
    assert ctx.open_proposals == []
    assert ctx.max_output_chars <= 8192


def test_a_missing_profile_means_no_knowledge():
    state = MeetingState(MEETING, "AB-1234", SEAT, None)
    ctx = build_context(FakeIO(state).request, state, FakeIO().client, timeout_s=40)
    assert ctx.knowledge.approved_facts == []
    assert ctx.knowledge.ask_when_uncertain is True


KNOWLEDGE = {
    "scope_kind": "meeting",
    "approved_facts": ["Saugat prefers Tuesday evenings."],
    "restricted_categories": ["health", "family"],
    "ask_when_uncertain": False,
    "return_summaries_to_agent": True,
    "scope": "meeting:3f2c6a5e-8a0e-4f7a-9d55-0d3c2a1b4c5d",
    "facts_withheld_reason": None,
}
MOTION = "55555555-5555-4555-8555-555555555555"


def test_the_turns_knowledge_beats_the_joined_profile():
    io = FakeIO(meeting_state(), knowledge=KNOWLEDGE)
    ctx = build_context(io.request, io.meeting, io.client, timeout_s=40)
    assert ctx.knowledge.approved_facts == ["Saugat prefers Tuesday evenings."]
    assert ctx.knowledge.restricted_categories == ["health", "family"]
    assert ctx.knowledge.ask_when_uncertain is False
    assert "scope" not in ctx.model_dump()["knowledge"]


def test_null_knowledge_on_the_turn_means_share_nothing():
    io = FakeIO(meeting_state(), knowledge=None)
    ctx = build_context(io.request, io.meeting, io.client, timeout_s=40)
    assert ctx.knowledge.approved_facts == []
    assert ctx.knowledge.restricted_categories == []


def test_briefing_and_open_motions_reach_the_context():
    io = FakeIO(
        meeting_state(),
        briefing={
            "purpose": "Spring spend",
            "agenda": [{"position": 0, "title": "Tool shed", "notes": "Two quotes"}],
            "organiser": {"handle": "priya", "display_name": "Priya"},
            "participants": [{"kind": "human", "handle": "priya", "display_name": "Priya"}],
            "previous_summaries": [],
        },
        open_motions=[
            {
                "id": MOTION,
                "proposal_message_id": str(uuid.uuid4()),
                "text": "Spend $400 on the roof",
                "closes_at": None,
            },
            {"id": str(uuid.uuid4())},  # no text: dropped
            "junk",
        ],
    )
    ctx = build_context(io.request, io.meeting, io.client, timeout_s=40)
    assert ctx.briefing is not None
    assert ctx.briefing.agenda[0].notes == "Two quotes"
    assert ctx.briefing.organiser is not None and ctx.briefing.organiser.handle == "priya"
    assert ctx.open_motions is not None
    assert [m.id for m in ctx.open_motions] == [MOTION]


def test_m4_requests_have_no_briefing_or_motions_and_dump_as_before():
    io = FakeIO(meeting_state())
    ctx = build_context(io.request, io.meeting, io.client, timeout_s=40)
    assert ctx.briefing is None and ctx.open_motions is None
    dumped = ctx.model_dump(mode="json")
    assert "briefing" not in dumped and "open_motions" not in dumped
    assert ctx.knowledge.approved_facts == PROFILE["approved_facts"]


def test_malformed_m5_fields_are_ignored():
    io = FakeIO(meeting_state(), briefing="hi", open_motions={"id": MOTION}, knowledge=["x"])
    ctx = build_context(io.request, io.meeting, io.client, timeout_s=40)
    assert ctx.briefing is None and ctx.open_motions is None
    assert ctx.knowledge.approved_facts == []


# ------------------------------------------------------------------ choosing the adapter


def test_make_turn_handler_per_adapter():
    ollama = make_turn_handler(BridgeConfig(adapter="ollama", model="llama3.1:8b"))
    assert isinstance(ollama, AdapterTurnHandler)
    assert isinstance(ollama.adapter, OllamaAdapter)
    agent = make_turn_handler(
        BridgeConfig(adapter="personal-agent", personal_agent_url="http://127.0.0.1:8765")
    )
    assert isinstance(agent.adapter, PersonalAgentAdapter)
    assert isinstance(make_turn_handler(BridgeConfig(adapter="interactive-mcp")), PassTurnHandler)


@pytest.mark.parametrize(
    "config",
    [BridgeConfig(adapter="ollama"), BridgeConfig(adapter="personal-agent")],
)
def test_make_turn_handler_needs_its_settings(config):
    with pytest.raises(ConfigError):
        make_turn_handler(config)


def test_make_turn_handler_rejects_cleartext_remote_adapter_without_opt_in():
    config = BridgeConfig(adapter="ollama", model="x", ollama_host="http://198.51.100.109:11434")
    with pytest.raises(ConfigError, match="https"):
        make_turn_handler(config)

    allowed = config.model_copy(update={"allow_insecure_adapter_http": True})
    assert isinstance(make_turn_handler(allowed).adapter, OllamaAdapter)


# ------------------------------------------------------------------ through the client


async def test_a_turn_request_through_the_client(home):
    async with FakeServer() as server:
        config = BridgeConfig(server_url=server.url, adapter="ollama", model="x")
        turns = AdapterTurnHandler(
            DemoAdapter("pragmatist"), Redactor(), timeout_s=5, chunk_interval_s=0
        )
        client = BridgeClient(
            config,
            Credentials(agent_id="ag_abc234", secret="sup3r-secret_value"),
            home=home,
            timing=FAST,
            turn_handler=turns,
        )
        async with Running(client):
            await server.authed.wait()
            await server.push(
                "meeting_joined",
                {
                    "meeting_id": MEETING,
                    "code": "AB-1234",
                    "seat_id": SEAT,
                    "knowledge_profile": PROFILE,
                },
            )
            question = row("Can we fix the roof this week?", "question").model_dump()
            await server.push("message_posted", {"message": question})
            await server.wait_for(
                lambda: MEETING in client.meetings and client.meetings[MEETING].recent
            )
            turn_id = str(uuid.uuid4())
            await server.push(
                "turn_request",
                {"meeting_id": MEETING, "turn_id": turn_id, "agenda_item": {"title": "Tool shed"}},
                meeting_id=MEETING,
            )
            await server.wait_for(lambda: server.of_type("turn_response"))
            await asyncio.sleep(0.05)
        states = [f["payload"]["state"] for f in server.of_type("activity")]
        assert states == ["reading_agenda", "preparing", "responding"]
        [response] = server.of_type("turn_response")
        [action] = response["payload"]["actions"]
        assert action["action"] == "reply"
        assert action["reply_to"] == question["id"]
        assert "Tool shed" in action["text"]


async def test_the_turns_knowledge_is_kept_for_the_meeting(home):
    async with FakeServer() as server:
        config = BridgeConfig(server_url=server.url, adapter="ollama", model="x")
        client = BridgeClient(
            config,
            Credentials(agent_id="ag_abc234", secret="sup3r-secret_value"),
            home=home,
            timing=FAST,
            turn_handler=PassTurnHandler(),
        )
        async with Running(client):
            await server.authed.wait()
            await server.push(
                "meeting_joined",
                {
                    "meeting_id": MEETING,
                    "code": "AB-1234",
                    "seat_id": SEAT,
                    "knowledge_profile": PROFILE,
                },
            )
            await server.wait_for(lambda: MEETING in client.meetings)
            assert client.meetings[MEETING].returns_summaries is False
            await server.push(
                "turn_request",
                {"meeting_id": MEETING, "turn_id": str(uuid.uuid4()), "knowledge": KNOWLEDGE},
                meeting_id=MEETING,
            )
            await server.wait_for(lambda: server.of_type("turn_response"))
            state = client.meetings[MEETING]
            assert state.turn_knowledge == KNOWLEDGE
            assert state.returns_summaries is True


# ------------------------------------------------------------------ allowed_actions


def test_allowed_actions_reach_the_context_and_its_json_only_when_limited():
    io = FakeIO(meeting_state())
    ctx = build_context(io.request, io.meeting, io.client, timeout_s=40)
    assert ctx.allowed_actions == ["message", "question", "reply", "proposal", "pass"]
    assert "allowed_actions" not in ctx.model_dump(mode="json")

    io = FakeIO(meeting_state(), allowed_actions=["message", "reply", "pass"])
    ctx = build_context(io.request, io.meeting, io.client, timeout_s=40)
    assert ctx.allowed_actions == ["message", "reply", "pass"]
    assert ctx.model_dump(mode="json")["allowed_actions"] == ["message", "reply", "pass"]


async def test_a_disallowed_proposal_goes_out_as_a_message_with_the_same_text():
    io = FakeIO(meeting_state(), allowed_actions=["message", "question", "reply", "pass"])
    _, actions = await run_turn(
        Scripted(say("Let's buy the roof.", "proposal"), streams=False), io=io
    )
    assert [(a.action, a.text) for a in actions] == [("message", "Let's buy the roof.")]


async def test_a_disallowed_action_passes_when_messages_are_off_too():
    io = FakeIO(meeting_state(), allowed_actions=["pass"])
    _, actions = await run_turn(Scripted(say("Hello all."), streams=False), io=io)
    assert actions == [TurnAction(action="pass")]
    assert "responding" not in io.activities


async def test_allowed_actions_go_out_unchanged():
    io = FakeIO(meeting_state(), allowed_actions=["message", "reply", "pass"])
    _, actions = await run_turn(Scripted(say("Agreed.", "reply"), streams=False), io=io)
    assert [(a.action, a.text) for a in actions] == [("reply", "Agreed.")]
