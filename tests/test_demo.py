import json
import re
from pathlib import Path

import pytest

from yantraxa_bridge import demo_personas
from yantraxa_bridge.adapters.base import ContextMessage, Final, OpenProposal
from yantraxa_bridge.adapters.demo import DemoAdapter

from .contexts import M1, M2, OTHER_SEAT, P1, SEAT, make_context


async def take(persona, ctx):
    events = [event async for event in DemoAdapter(persona).take_turn(ctx)]
    assert len(events) == 1 and isinstance(events[0], Final)
    [action] = events[0].actions
    return action


def message(id_, body, kind="message"):
    return ContextMessage(
        id=id_,
        seat_id=OTHER_SEAT,
        author="Priya",
        kind=kind,
        body=body,
        reply_to=None,
        created_at="2026-10-11T09:00:00.000Z",
    )


@pytest.mark.parametrize("persona", demo_personas.PERSONAS)
async def test_same_context_same_answer(persona):
    ctx = make_context()
    answers = [(await take(persona, ctx)).model_dump() for _ in range(3)]
    assert answers[0] == answers[1] == answers[2]


async def test_personas_answer_differently():
    ctx = make_context()
    texts = {(await take(p, ctx)).text for p in demo_personas.PERSONAS}
    assert len(texts) == len(demo_personas.PERSONAS)


@pytest.mark.parametrize("persona", demo_personas.PERSONAS)
async def test_an_open_proposal_gets_a_vote(persona):
    action = await take(persona, make_context())
    assert action.vote is not None
    assert action.vote.motion_id == P1
    assert action.vote.choice in {"yes", "no", "abstain"}
    # A vote doesn't quote the proposal: it's one of the persona's lines for its choice.
    assert action.text in demo_personas.VOTE_LINES[persona][action.vote.choice]
    # The proposal's message isn't known here, so the vote rides on a plain message.
    assert action.action == "message" and action.reply_to is None


async def test_a_vote_replies_to_the_proposal_message_when_known():
    proposal = message(M1, "Spend $400 on a new shed roof", "proposal")
    ctx = make_context(
        recent_messages=[proposal],
        open_proposals=[OpenProposal(id=P1, text=proposal.body, message_id=M1)],
    )
    action = await take("budget_hawk", ctx)
    assert action.action == "reply" and action.reply_to == M1
    assert action.vote is not None and action.vote.choice == "no"


@pytest.mark.parametrize("persona", demo_personas.PERSONAS)
async def test_a_question_gets_a_reply(persona):
    action = await take(persona, make_context(open_proposals=[]))
    assert action.action == "reply"
    assert action.reply_to == M2
    assert action.text in demo_personas.REPLIES[persona]


@pytest.mark.parametrize("persona", demo_personas.PERSONAS)
async def test_an_empty_room_gets_an_opener_naming_the_agenda_item(persona):
    action = await take(persona, make_context(open_proposals=[], recent_messages=[]))
    assert action.action in {"message", "question"}
    assert action.reply_to is None
    assert "\u201cTool shed repairs\u201d" in action.text


@pytest.mark.parametrize("persona", demo_personas.PERSONAS)
async def test_a_statement_quotes_a_person_or_names_them(persona):
    ctx = make_context(
        open_proposals=[], recent_messages=[message(M1, "We could borrow ladders from the school.")]
    )
    action = await take(persona, ctx)
    assert action.action in {"message", "question", "proposal"}
    assert '"' not in action.text
    if "\u201c" in action.text:
        assert "\u201cWe could borrow ladders from the school\u201d" in action.text


async def test_long_last_messages_are_shortened():
    ctx = make_context(open_proposals=[], recent_messages=[message(M1, "word " * 200)])
    action = await take("pragmatist", ctx)
    assert len(action.text) < 400


CHAIN = (
    'Re ""Building on "We have the library\'s community room on 14 March for 40 people": '
    "for Tool shed repairs, let's run it as a two-week trial with one owner, then "
    'review."...": for Tool shed repairs I\'d want a costed quote first.'
)


@pytest.mark.parametrize(
    ("body", "quote"),
    [
        (CHAIN, "I'd want a costed quote first"),
        ('Re "Spend $400 on the roof": doable if we pick one option now.', None),
        ('On Seeds, re "x": only if it fits the budget we already have.', None),
        ('I like "a plan". For Seeds, let\'s invite the wider group to help.', None),
        ('For Seeds: we go with "Fix the roof first and the shed later" now.', None),
        ('We call it "the hub" now.', "We call it the hub now"),
        ('Re "x":', None),
        ('"..."', None),
    ],
)
def test_quotes_drop_earlier_quotes(body, quote):
    got = demo_personas.quote_of(body, "Tool shed repairs")
    if quote is not None:
        assert got == quote
    assert got is None or not any(mark in got for mark in '"\u201c\u201d')
    assert got is None or len(got.removesuffix("...").split()) <= demo_personas.QUOTE_WORDS


def test_a_message_with_nothing_left_to_quote_names_the_speaker():
    ctx = make_context(open_proposals=[], recent_messages=[message(M1, 'Re "x":')])
    for persona in demo_personas.PERSONAS:
        text = demo_personas.respond(persona, ctx.model_dump(mode="json"))["text"]
        if "point" in text:
            assert "Priya's point" in text
        assert not re.search('["\u201c\u201d]', text), text


def test_a_person_is_quoted_in_curly_quotes_cut_with_an_ellipsis_and_an_agent_only_named():
    def said(author_kind):
        return demo_personas.respond(
            "pragmatist",
            {
                "me": {"seat_id": "me"},
                "agenda_item": {"title": "Venue and date"},
                "recent_messages": [
                    {
                        "id": "m1",
                        "seat_id": "s1",
                        "author": "Priya",
                        "author_kind": author_kind,
                        "kind": "message",
                        "body": "Town Hall on 14 March with a $2,000 cap, for 120 people",
                    }
                ],
            },
        )["text"]

    assert "\u201cTown Hall on 14 March with a $2,000\u2026\u201d" in said("human")
    assert "Priya's point" in said("agent")
    assert not re.search("[\u201c\u201d]", said("agent"))


async def test_the_adapter_names_another_agents_point_instead_of_quoting_it():
    agent_message = message(M1, "We could borrow ladders from the school.").model_copy(
        update={"author": "Ana's agent", "author_kind": "agent"}
    )
    action = await take(
        "pragmatist", make_context(open_proposals=[], recent_messages=[agent_message])
    )
    assert "Ana's agent's point" in action.text
    assert "borrow ladders" not in action.text


def test_a_short_message_is_quoted_back_whole():
    ctx = {
        "me": {"seat_id": "me"},
        "agenda_item": {"title": "Spring fair"},
        "recent_messages": [
            {
                "id": "m7",
                "seat_id": "s1",
                "author": "Priya",
                "kind": "message",
                "body": "Let's book the library hall.",
            },
        ],
    }
    for persona in demo_personas.PERSONAS:
        for n in range(4):
            mine_now = {
                "id": f"x{n}",
                "seat_id": "me",
                "author": "you",
                "kind": "message",
                "body": f"turn {n}",
            }
            turn = {**ctx, "recent_messages": [*ctx["recent_messages"], mine_now]}
            text = demo_personas.respond_fresh(persona, turn)["text"]
            if "\u201c" in text:
                assert "\u201cLet's book the library hall\u201d" in text


def test_the_topic_is_named_once_in_curly_quotes_or_this_when_there_is_none():
    for persona in demo_personas.PERSONAS:
        named = demo_personas.respond(persona, {"agenda_item": {"title": "Topic shortlist"}})
        assert "\u201cTopic shortlist\u201d" in named["text"]
        unnamed = demo_personas.respond(persona, {})["text"]
        assert re.search(r"\bthis\b", unnamed)
        assert not re.search('["\u201c\u201d]', unnamed)


def test_a_persona_backs_its_own_proposal_without_replying_to_itself():
    ctx = {
        "me": {"seat_id": "me"},
        "agenda_item": {"title": "Topic shortlist"},
        "recent_messages": [
            {
                "id": "m1",
                "seat_id": "s1",
                "author": "Priya",
                "kind": "message",
                "body": "We could start with the hall.",
            },
            {
                "id": "m2",
                "seat_id": "me",
                "author": "you",
                "kind": "proposal",
                "body": "I propose we book the hall.",
            },
        ],
        "open_proposals": [
            {"id": "mo1", "text": "I propose we book the hall.", "message_id": "m2"}
        ],
    }
    for persona in demo_personas.PERSONAS:
        answer = demo_personas.respond(persona, ctx)
        assert answer["vote"] == {"motion_id": "mo1", "choice": "yes"}
        assert answer["reply_to"] is None
        assert answer["action"] == "message"
        assert re.search(r"my (own )?proposal", answer["text"])
    # Someone else's proposal still gets a vote, replying to it.
    theirs = {
        **ctx,
        "recent_messages": [
            *ctx["recent_messages"],
            {
                "id": "m3",
                "seat_id": "s1",
                "author": "Priya",
                "kind": "proposal",
                "body": "Book it for 14 March.",
            },
        ],
        "open_proposals": [
            *ctx["open_proposals"],
            {"id": "mo2", "text": "Book it for 14 March.", "message_id": "m3"},
        ],
    }
    for persona in demo_personas.PERSONAS:
        answer = demo_personas.respond(persona, theirs)
        assert answer["vote"]["motion_id"] == "mo2"
        assert answer["reply_to"] == "m3"


def test_demo_agents_replying_to_each_other_never_nest_quotes():
    # Three demo agents in one room, each answering the room as the server would show it.
    seats = {
        p: f"{n:08x}-aaaa-4aaa-8aaa-aaaaaaaaaaaa" for n, p in enumerate(demo_personas.PERSONAS)
    }
    base = make_context(open_proposals=[], recent_messages=[]).model_dump(mode="json")
    messages = [
        {
            "id": "f0000000-0000-4000-8000-000000000000",
            "seat_id": OTHER_SEAT,
            "author": "Priya",
            "kind": "message",
            "body": "We have the library's community room on 14 March for 40 people.",
            "reply_to": None,
            "created_at": "2026-10-11T09:00:00.000Z",
        }
    ]
    proposals: list[dict] = []
    texts = []
    for turn in range(12):
        persona = demo_personas.PERSONAS[turn % 3]
        ctx = {
            **base,
            "me": {**base["me"], "seat_id": seats[persona]},
            "recent_messages": messages[-10:],
            "open_proposals": proposals[-1:],
        }
        answer = demo_personas.respond(persona, ctx)
        mid = f"f{turn + 1:07x}-0000-4000-8000-000000000000"
        kind = {"reply": "message"}.get(answer["action"], answer["action"])
        messages.append(
            {
                "id": mid,
                "seat_id": seats[persona],
                "author": demo_personas.NAMES[persona],
                "kind": kind,
                "body": answer["text"],
                "reply_to": answer["reply_to"],
                "created_at": "2026-10-11T09:00:00.000Z",
            }
        )
        if answer["vote"]:
            proposals.clear()  # one round of votes, then the motion closes
        if kind == "proposal":
            proposals.append({"id": f"p{mid[1:]}", "text": answer["text"], "message_id": mid})
        texts.append(answer["text"])
    assert len(texts) >= 6
    for text in texts:
        assert len(re.findall('["\u201c\u201d]', text)) <= 2, text
        assert not re.search('(Re|Building on|re) ["\u201c]+(Re|Building on|On|["\u201c])', text)


@pytest.mark.parametrize("persona", demo_personas.PERSONAS)
def test_a_lone_demo_agent_never_repeats_its_own_last_line(persona):
    meeting = {"title": "Garden day", "agenda": [{"title": "Tools"}]}
    ctx = {"me": {"seat_id": "me"}, "meeting": meeting, "agenda_item": {"title": "Tools"}}
    opener = demo_personas.respond(persona, ctx)
    # Nobody else speaks, so `respond` gives the same opener every turn.
    again = {
        **ctx,
        "recent_messages": [
            {
                "id": "m1",
                "seat_id": "me",
                "author": "you",
                "kind": "message",
                "body": opener["text"],
            }
        ],
    }
    assert demo_personas.respond(persona, again)["text"] == opener["text"]
    fresh = demo_personas.respond_fresh(persona, again)
    assert fresh["text"] != opener["text"]
    assert "\u201cTools\u201d" in fresh["text"]
    # With nothing of its own said yet, it answers exactly as `respond` does.
    assert demo_personas.respond_fresh(persona, ctx) == opener
    # A third turn, after the second line, may go back to the first.
    third = {
        **ctx,
        "recent_messages": [
            *again["recent_messages"],
            {
                "id": "m2",
                "seat_id": "me",
                "author": "you",
                "kind": "message",
                "body": fresh["text"],
            },
        ],
    }
    assert demo_personas.respond_fresh(persona, third)["text"] != fresh["text"]


@pytest.mark.parametrize("persona", demo_personas.PERSONAS)
@pytest.mark.parametrize("own", [False, True])
def test_a_vote_line_is_never_repeated_across_a_meetings_proposals(persona, own):
    meeting = {"title": "Launch", "agenda": [{"title": "Plan"}]}
    recent: list[dict] = []
    said = []
    for n in range(1, 5):
        # Pragmatist and Budget hawk each answer a costed, dated proposal one way.
        proposal = {"id": f"motion-{n}", "message_id": f"p{n}", "text": f"Spend ${n}00 on day {n}"}
        recent.append(
            {
                "id": f"p{n}",
                "seat_id": "me" if own else "them",
                "author": "Ana",
                "kind": "proposal",
                "body": proposal["text"],
            }
        )
        ctx = {
            "me": {"seat_id": "me"},
            "meeting": meeting,
            "recent_messages": [*recent],
            "open_proposals": [proposal],
        }
        answer = demo_personas.respond(persona, ctx)
        assert answer["vote"]["motion_id"] == proposal["id"]
        said.append(answer["text"])
        recent.append(
            {
                "id": f"a{n}",
                "seat_id": "me",
                "author": "you",
                "kind": answer["action"],
                "body": answer["text"],
            }
        )
    assert len(set(said)) == len(said)


async def test_the_adapter_never_sends_its_own_last_line_again():
    first = await take("pragmatist", make_context(open_proposals=[], recent_messages=[]))
    mine_before = message(M1, first.text).model_copy(update={"seat_id": SEAT, "author": "you"})
    again = await take("pragmatist", make_context(open_proposals=[], recent_messages=[mine_before]))
    assert again.text != first.text


def test_unknown_persona_is_refused():
    with pytest.raises(ValueError):
        DemoAdapter("pirate")


def test_describe():
    info = DemoAdapter("budget_hawk").describe()
    assert "Demo" in info.runtime_label
    assert info.streams is False


# ---------------------------------------------------------------- server parity vectors
# The server's in-process demo agents (`apps/server/src/demo/personas.ts`) speak with the
# same lines and rules. This writes every persona's answer to a spread of contexts to
# `vectors/demo_personas.json`, regenerated on every run, so commit the file whenever it
# changes. Inside the Yantraxa repository, the server's own vectors
# (`fixtures/persona-vectors.json`, rewritten there with UPDATE_PERSONA_VECTORS=1) are
# checked here too, answer for answer; the public bridge repository doesn't have them.

VECTORS = Path(__file__).parent / "vectors" / "demo_personas.json"
SERVER_VECTORS = (
    Path(__file__).resolve().parents[2]
    / "apps"
    / "server"
    / "src"
    / "demo"
    / "fixtures"
    / "persona-vectors.json"
)
# Whitespace other than a plain space, all of which `str.split()` splits on.
ODD_WHITESPACE = "\tLadders\u00a0from\u2028the\x1cschool,\u3000please \x85now\n\n"


def mine(id_, body, kind="message"):
    return message(id_, body, kind).model_copy(update={"seat_id": SEAT, "author": "you"})


def vector_contexts():
    long_word = "x" * 120
    cases = {
        "default": make_context(),
        "proposal_with_message": make_context(
            recent_messages=[message(M1, "Spend $400 on a new shed roof", "proposal")],
            open_proposals=[
                OpenProposal(id=P1, text="Spend $400 on a new shed roof", message_id=M1)
            ],
        ),
        "proposal_plain_words": make_context(
            open_proposals=[OpenProposal(id=P1, text="Ask the council for help", message_id=M1)]
        ),
        "proposal_money_word_only": make_context(
            open_proposals=[OpenProposal(id=P1, text="Split the COST with the school")]
        ),
        "proposal_word_boundary": make_context(
            open_proposals=[OpenProposal(id=P1, text="A costly, overpaid, unbudgeted plan")]
        ),
        "proposal_unicode_digit_and_case": make_context(
            open_proposals=[OpenProposal(id=P1, text="Fix it by week \u00b2 and PR\u0130CE it")]
        ),
        "proposal_arabic_digit": make_context(
            open_proposals=[OpenProposal(id=P1, text="Meet on day \u0663 about the caf\u00e9")]
        ),
        "proposal_long_s": make_context(
            open_proposals=[OpenProposal(id=P1, text="Who pays the co\u017fts?")]
        ),
        "proposal_empty_message_id": make_context(
            open_proposals=[OpenProposal(id=P1, text="Paint it green", message_id="")]
        ),
        "two_proposals_latest_wins": make_context(
            open_proposals=[
                OpenProposal(id=P1, text="Spend $400 on a new shed roof", message_id=M1),
                OpenProposal(id=M2, text="Patch the roof with tarp", message_id=M2),
            ]
        ),
        "question": make_context(open_proposals=[]),
        "empty_room": make_context(open_proposals=[], recent_messages=[]),
        "only_my_messages": make_context(
            open_proposals=[], recent_messages=[mine(M1, "I said something earlier.")]
        ),
        "my_message_after_theirs": make_context(
            open_proposals=[],
            recent_messages=[
                message(M1, "We could borrow ladders from the school."),
                mine(M2, "Good idea?", "question"),
            ],
        ),
        "statement": make_context(
            open_proposals=[],
            recent_messages=[message(M1, "We could borrow ladders from the school.")],
        ),
        "system_last": make_context(
            open_proposals=[],
            recent_messages=[message(M2, "Agenda item started.", "system")],
        ),
        "long_message": make_context(
            open_proposals=[], recent_messages=[message(M1, "word " * 200)]
        ),
        "long_unbroken_word": make_context(
            open_proposals=[], recent_messages=[message(M1, long_word)]
        ),
        "exactly_eighty": make_context(
            open_proposals=[], recent_messages=[message(M1, "a" * 79 + "!")]
        ),
        "trailing_punctuation_cut": make_context(
            open_proposals=[],
            recent_messages=[message(M1, ("We need more hands, " * 3) + "and; more: tools?! " * 4)],
        ),
        "odd_whitespace": make_context(
            open_proposals=[],
            recent_messages=[message(M1, ODD_WHITESPACE)],
        ),
        "astral_characters": make_context(
            open_proposals=[],
            recent_messages=[message(M1, "🌱🌻 " * 30 + "seeds for the spring beds")],
        ),
        "quote_chain": make_context(open_proposals=[], recent_messages=[message(M1, CHAIN)]),
        "quote_chain_question": make_context(
            open_proposals=[], recent_messages=[message(M1, CHAIN + " Who pays?", "question")]
        ),
        "nothing_left_to_quote": make_context(
            open_proposals=[], recent_messages=[message(M1, 'Re "x":')]
        ),
        "nothing_left_no_author": make_context(
            open_proposals=[],
            recent_messages=[message(M1, '"..."').model_copy(update={"author": " "})],
        ),
        "curly_and_short_quotes": make_context(
            open_proposals=[],
            recent_messages=[
                message(
                    M1,
                    "We call it \u201cthe hub\u201d, says \u201cthe whole street, every one\u201d.",
                )
            ],
        ),
        "proposal_quoting_a_proposal": make_context(
            open_proposals=[
                OpenProposal(
                    id=P1,
                    text='Building on "Spend $400 on a new shed roof": for Tool shed repairs, '
                    "let's run it as a two-week trial with one owner, then review.",
                    message_id=M1,
                )
            ]
        ),
        "proposal_nothing_left": make_context(
            open_proposals=[OpenProposal(id=P1, text='Re "x":', message_id=M1)]
        ),
        "no_agenda_item": make_context(agenda_item=None, open_proposals=[], recent_messages=[]),
        "no_agenda_at_all": make_context(
            agenda_item=None,
            meeting=make_context().meeting.model_copy(update={"agenda": []}),
            open_proposals=[],
            recent_messages=[],
        ),
        "no_titles": make_context(
            agenda_item=None,
            meeting=make_context().meeting.model_copy(update={"agenda": [], "title": None}),
            open_proposals=[],
            recent_messages=[],
        ),
    }
    cases["own_proposal_open"] = make_context(
        recent_messages=[
            message(M1, "We have the library room in March."),
            mine(M2, "I propose we pick one option today.", "proposal"),
        ],
        open_proposals=[
            OpenProposal(id=P1, text="I propose we pick one option today.", message_id=M2)
        ],
    )
    cases["short_human_message"] = make_context(
        open_proposals=[],
        recent_messages=[
            message(M1, "Let's book the library hall.").model_copy(update={"author_kind": "human"})
        ],
    )
    cases["agent_message"] = make_context(
        open_proposals=[],
        recent_messages=[
            message(M1, "We could borrow ladders from the school.").model_copy(
                update={"author": "Ana's agent", "author_kind": "agent"}
            )
        ],
    )
    # A spread of topics and message ids, so both wordings of every pick show up.
    for n in range(12):
        topic = f"Item {n}: {['Compost', 'Water butts', 'Fences', 'Bees'][n % 4]}"
        item = make_context().agenda_item.model_copy(update={"title": topic})
        mid = f"{n:08x}-1111-4111-8111-111111111111"
        cases[f"spread_{n:02d}_opener"] = make_context(
            agenda_item=item, open_proposals=[], recent_messages=[]
        )
        cases[f"spread_{n:02d}_statement"] = make_context(
            agenda_item=item,
            open_proposals=[],
            recent_messages=[message(mid, f"Option {n}: we share the {topic.lower()} rota.")],
        )
        cases[f"spread_{n:02d}_question"] = make_context(
            agenda_item=item,
            open_proposals=[],
            recent_messages=[message(mid, f"Who leads {topic}?", "question")],
        )
    return cases


def vectors():
    cases = []
    for name, ctx in vector_contexts().items():
        context = ctx.model_dump(mode="json")
        answers = {p: demo_personas.respond(p, context) for p in demo_personas.PERSONAS}
        cases.append({"name": name, "context": context, "answers": answers})
    return cases


def test_parity_vectors_are_written():
    content = json.dumps({"cases": vectors()}, ensure_ascii=False, indent=2) + "\n"
    if not VECTORS.exists() or VECTORS.read_text(encoding="utf-8") != content:
        VECTORS.parent.mkdir(exist_ok=True)
        VECTORS.write_text(content, encoding="utf-8")
    cases = json.loads(VECTORS.read_text(encoding="utf-8"))["cases"]
    assert len(cases) == len(vector_contexts())
    # Every rule and both wordings of every pick appear somewhere.
    answers = [(p, a) for case in cases for p, a in case["answers"].items()]
    actions = {(p, a["action"]) for p, a in answers}
    assert ("pragmatist", "proposal") in actions and ("budget_hawk", "question") in actions
    assert {a["vote"]["choice"] for _, a in answers if a["vote"]} == {"yes", "no", "abstain"}
    for templates in (demo_personas.REPLIES, demo_personas.STATEMENTS, demo_personas.OPENERS):
        for persona, options in templates.items():
            texts = [a["text"] for p, a in answers if p == persona]
            for option in options:
                template = option[1] if isinstance(option, tuple) else option
                pattern = re.escape(template)
                pattern = pattern.replace(r"\{topic\}", ".*").replace(r"\{quote\}", ".*")
                assert any(re.fullmatch(pattern, t, re.S) for t in texts), template


def test_every_vector_sounds_like_a_person():
    for case in vectors():
        ctx = case["context"]
        topic = (ctx.get("agenda_item") or {}).get("title") or ""
        for persona, answer in case["answers"].items():
            text = answer["text"]
            label = f"{case['name']} / {persona}"
            # Curly quotes only, never nested; never a cut-off excerpt.
            assert '"' not in text, label
            assert len(re.findall("[\u201c\u201d]", text)) <= 2, label
            assert "..." not in text, label
            # An ellipsis only ends a quote.
            if "\u2026" in text:
                assert "\u2026\u201d" in text, label
            # The agenda title is never a lead-in ("For Topic shortlist: ...").
            if topic.strip():
                assert not re.match("(For|On|Re) [^\u201c]*:", text), label


@pytest.mark.skipif(not SERVER_VECTORS.exists(), reason="outside the Yantraxa repository")
def test_the_servers_vectors_get_the_same_answers():
    cases = json.loads(SERVER_VECTORS.read_text(encoding="utf-8"))["cases"]
    assert len(cases) > 40
    for case in cases:
        assert sorted(case["answers"]) == sorted(demo_personas.PERSONAS), case["name"]
        for persona, expected in case["answers"].items():
            got = demo_personas.respond(persona, case["context"])
            assert got == expected, f"{case['name']} / {persona}"
