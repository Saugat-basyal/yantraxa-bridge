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
    assert "Spend $400 on a new shed roof" in action.text
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
async def test_a_question_gets_a_reply_naming_the_agenda_item_and_the_question(persona):
    ctx = make_context(open_proposals=[])
    action = await take(persona, ctx)
    assert action.action == "reply"
    assert action.reply_to == M2
    assert "Tool shed repairs" in action.text
    assert "afford $400" in action.text


@pytest.mark.parametrize("persona", demo_personas.PERSONAS)
async def test_an_empty_room_gets_an_opener(persona):
    action = await take(persona, make_context(open_proposals=[], recent_messages=[]))
    assert action.action in {"message", "question"}
    assert action.reply_to is None
    assert "Tool shed repairs" in action.text


@pytest.mark.parametrize("persona", demo_personas.PERSONAS)
async def test_a_statement_quotes_the_last_message(persona):
    ctx = make_context(
        open_proposals=[], recent_messages=[message(M1, "We could borrow ladders from the school.")]
    )
    action = await take(persona, ctx)
    assert action.action in {"message", "question", "proposal"}
    assert "borrow ladders" in action.text
    assert "Tool shed repairs" in action.text


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
    texts = {
        demo_personas.respond(p, ctx.model_dump(mode="json"))["text"]
        for p in demo_personas.PERSONAS
    }
    assert all("Priya's point" in t and '"' not in t for t in texts), texts


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
        assert text.count('"') <= 2, text
        assert not re.search(r'(Re|Building on|re) "+(Re|Building on|On|")', text), text
        assert "..." not in text or text.count("...") == 1, text


def test_unknown_persona_is_refused():
    with pytest.raises(ValueError):
        DemoAdapter("pirate")


def test_describe():
    info = DemoAdapter("budget_hawk").describe()
    assert "Demo" in info.runtime_label
    assert info.streams is False


# ---------------------------------------------------------------- server parity vectors
# The server's in-process demo agents (`apps/server/src/demo/personas.ts`) port
# `demo_personas` line for line. This writes every persona's answer to a spread of
# contexts to `vectors/demo_personas.json`; the server's `personas.test.ts` asserts each
# one byte for byte. Regenerated on every run, so commit the file whenever it changes.

VECTORS = Path(__file__).parent / "vectors" / "demo_personas.json"
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
    # A spread of topics and message ids, so both wordings of every `_pick` show up.
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
