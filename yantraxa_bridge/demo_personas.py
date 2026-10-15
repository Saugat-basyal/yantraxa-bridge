"""Rule-based demo personas (spec §11): Pragmatist, Budget hawk, Community builder.

Self-contained on purpose (standard library only, plain dicts in and out). The server's
in-process demo agents (`apps/server/src/demo/personas.ts`) speak with the same lines and
rules; `tests/test_demo.py` checks this module against the server's own vectors whenever
it runs inside the Yantraxa repository. `respond(persona, ctx)` takes a turn context as a
dict (`TurnContext.model_dump(mode="json")`) and returns one answer in the model's output
shape. There is no randomness: the same context always gives the same answer. Wording
varies only with a hash of the persona, agenda item and last message.

Rules, in order:
1. An open proposal: vote on the latest one, replying to it when its message is known.
   On its own proposal the persona votes for it, without replying to itself.
2. The last message from someone else is a question: reply to it.
3. Nobody has spoken: open the agenda item.
4. Otherwise: react to the last message (Pragmatist proposes, Budget hawk asks about
   cost, Community builder widens the circle).

A line the persona has already said in `recent_messages` is skipped while another one
fits, and `respond_fresh` never repeats the persona's own last message, which the
server would refuse as duplicate text.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

PERSONAS = ("pragmatist", "budget_hawk", "community_builder")
NAMES = {
    "pragmatist": "Pragmatist",
    "budget_hawk": "Budget hawk",
    "community_builder": "Community builder",
}
QUOTE_WORDS = 8
_MONEY = re.compile(r"[$£€]|\b(cost|costs|spend|budget|price|pay)\b", re.IGNORECASE)

# (action, template) per persona and situation. `{topic}` is the agenda item's title,
# named once in an opener and in curly quotes, never pasted in as a noun; with no title
# the quoted topic becomes "this". `{quote}` is a person's words from the last message in
# curly quotes, or whose point it was ("Priya's point"): an agent's message is always
# named, never quoted, so demo agents don't parrot each other.
OPENERS = {
    "pragmatist": [
        (
            "question",
            "Shall we start with {topic}? What's the smallest step we could agree today, "
            "and who'd own it?",
        ),
        (
            "message",
            "Let's start with {topic}. I'd like us to leave with one owner, one date and "
            "one next step.",
        ),
    ],
    "budget_hawk": [
        (
            "question",
            "Before we decide anything on {topic}, what will it cost all in, "
            "and what's the cheapest version that works?",
        ),
        (
            "message",
            "On {topic}, I'd like a costed option on the table before we commit to anything.",
        ),
    ],
    "community_builder": [
        (
            "message",
            "Before we settle {topic}, let's make sure everyone it affects has had a say.",
        ),
        ("question", "Who else should we bring in on {topic} before we decide anything?"),
    ],
}
# Answers to a question, sent as replies to it, so they don't need to quote it.
REPLIES = {
    "pragmatist": [
        "I think so, as long as someone owns it and we set a date. "
        "I'd start small and look again in two weeks.",
        "We can, if we pick one option now and come back to it at the next meeting.",
    ],
    "budget_hawk": [
        "I'd want a proper quote first, and a spending cap we all agree not to go over.",
        "Only if it fits inside the budget we already have. I'm not keen to stretch it.",
    ],
    "community_builder": [
        "Could we ask for volunteers? More hands keeps the cost down, and people feel part of it.",
        "Let's check with the people it affects most first, then decide it together.",
    ],
}
STATEMENTS = {
    "pragmatist": [
        (
            "proposal",
            "Building on {quote}, I propose we run this as a two-week trial with one owner, "
            "then review it.",
        ),
        (
            "proposal",
            "I propose we act on {quote} now, with one owner, and check back at the next meeting.",
        ),
        (
            "proposal",
            "I propose we take {quote} forward as a small first step, with a date to report back.",
        ),
    ],
    "budget_hawk": [
        ("question", "I hear {quote}. What would it cost us, though, and is there a cheaper way?"),
        (
            "question",
            "Who pays for this, and what would we drop to afford it? I'd like to know first.",
        ),
    ],
    "community_builder": [
        (
            "message",
            "I like {quote}. Let's ask the wider group to help, so it doesn't land on one person.",
        ),
        (
            "message",
            "This could bring people together if we share the work. I'm happy to help organise it.",
        ),
    ],
}
# The persona's vote on its own proposal: for it, said to the room, not as a reply to itself.
OWN_VOTE = [
    "It's my proposal, so I'm voting yes. Objections are welcome before it closes.",
    "Yes from me on my own proposal. Say so if you see a problem with it.",
    "I'm voting for my own proposal. Tell me now if something's missing from it.",
    "It's a yes from me, as it's my proposal. I'm happy to change it if anyone objects.",
]
# Votes are sent as replies to the proposal, so they don't quote it either. A persona
# votes on every turn while a proposal is open, so each list has a few lines, and a line
# the persona has already said is skipped (`_unsaid`): the record never repeats itself.
VOTE_LINES = {
    "pragmatist": {
        "yes": [
            "Yes from me. It's concrete enough that we can act on it straight away.",
            "I'll vote yes. There's a clear enough step here for someone to own.",
            "Yes. It's small enough to try, and we'll know quickly whether it works.",
            "I'm for it. Let's put a name and a date next to it and get going.",
        ],
        "abstain": [
            "I'll abstain for now. I'd vote yes once it has an owner and a date.",
            "Abstaining until someone owns this and we've put a date on it.",
            "I'll sit this one out until it says who does what, and by when.",
            "No vote from me yet. Give it an owner and a deadline and I'll back it.",
        ],
    },
    "budget_hawk": {
        "no": [
            "No from me as it stands. I'd want a firm cap on the spending first.",
            "I'm voting no for now. Put a ceiling on the cost and I'll look again.",
            "Not as written. It needs a budget line and a limit we won't go over.",
            "No for now. I can't back spending without a number we've agreed on.",
        ],
        "abstain": [
            "I'll abstain until we know what this will cost, even roughly.",
            "Abstaining for now. Show me a rough cost and I can vote properly.",
            "I'm holding off. I'd like even a ballpark figure before I vote.",
            "Abstaining. Tell me what it costs and what it replaces, and I'll decide.",
        ],
    },
    "community_builder": {
        "yes": [
            "Yes. It gets people working on this together, which is what we want.",
            "I'm voting yes. It's something people can join in with, and that matters.",
            "Yes from me. It gives more people a way to get involved.",
            "I'll back it. People stick with something they've helped to build.",
        ],
    },
}


def _choice(persona: str, proposal_text: str) -> str:
    if persona == "pragmatist":
        return "yes" if any(ch.isdigit() for ch in proposal_text) else "abstain"
    if persona == "budget_hawk":
        return "no" if _MONEY.search(proposal_text) else "abstain"
    return "yes"


# A lead-in built around a quote, as the bridge's earlier templates (and the server's
# before UI audit F7) wrote them, still met in messages from older demo agents:
# `Re "...":`, `On X, re "...":`, `Building on "...":`, `Voting no on "..." as it stands:`,
# `I like "...".`, `"..." -`. Whitespace is already flattened to single spaces. The
# quote is greedy, so a lead-in quoting a chain of quotes goes in one step.
_QUOTE_LEAD = re.compile(
    '(?:(?:[Oo]n [^"\u201c\u201d:]{1,80}, )?[Rr]e |Building on |Voting (?:yes|no) on |I like )?'
    '["\u201c].*["\u201d](?: as it stands)? ?[:.\\-\u2013\u2014] ?'
)
# A lead-in naming the topic: `On X:`, `For X:`, `Before we decide anything on X:`.
_TOPIC_LEAD = re.compile('(?:[Oo]n|[Ff]or|Before we decide anything on) [^"\u201c\u201d:]{1,80}: ?')
# A span in straight or curly double quotes, and the marks themselves.
_QUOTED_SPAN = re.compile('["\u201c][^"\u201c\u201d]*["\u201d]')
_QUOTE_MARK = re.compile('["\u201c\u201d]')
_WORD = re.compile(r"\w")
_PLACEHOLDER = re.compile(r"\{(topic|quote)\}")
# JavaScript's `\s` and `trim()`, as the server normalises a seat's messages.
_JS_SPACE = "\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
_JS_SPACES = re.compile(f"[{_JS_SPACE}]+")
_JS_TRIM = re.compile(f"^[{_JS_SPACE}]+|[{_JS_SPACE}]+$")
# The fewest words worth quoting back; anything shorter is named, not quoted.
_MIN_QUOTE_WORDS = 3


def _flatten(text: str) -> str:
    return " ".join(text.split())


def _without_topic(flat: str, topic: str) -> str | None:
    """`flat` without a leading `for <topic>,` or `on <topic>:`; None when it has none."""
    for word in ("For", "for", "On", "on"):
        lead = f"{word} {topic}"
        if topic and flat.startswith(lead):
            rest = flat[len(lead) :]
            if rest[:1] in (",", ":"):
                rest = rest[1:]
            if rest == "" or rest[:1] == " ":
                return rest.lstrip(" ")
    return None


def quote_of(text: str, topic: str = "", words: int = QUOTE_WORDS) -> str | None:
    """A short phrase in the message's own words, safe to put in quotes.

    Lead-ins that quote an earlier message (`Re "...":`, `Building on "...":`) or name
    the topic (`On X:`, `for <topic>,`) are stripped until none is left, quotes still
    inside are dropped so they never nest, and the rest is cut to `words` words. None
    when nothing meaningful is left, or when the rest still quotes more than a short
    phrase: it's a reaction to an earlier message, better named than quoted.
    """
    flat = _flatten(text)
    topic = _flatten(topic)
    while True:
        lead = _QUOTE_LEAD.match(flat) or _TOPIC_LEAD.match(flat)
        if lead is not None:
            flat = flat[lead.end() :]
            continue
        rest = _without_topic(flat, topic)
        if rest is None:
            break
        flat = rest
    if any(len(span[1:-1].split(" ")) > 3 for span in _QUOTED_SPAN.findall(flat)):
        return None
    flat = _QUOTE_MARK.sub("", flat)
    parts = flat.split()
    phrase = " ".join(parts[:words]).rstrip(",;:.!?-\u2013\u2014")
    if not _WORD.search(phrase):
        return None
    return phrase + "..." if len(parts) > words else phrase


def _pick_shifted(options: list[Any], shift: int, *parts: str) -> Any:
    """One option, fixed by a hash of `parts`; `shift` > 0 moves on to the next ones."""
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return options[(digest[0] + shift) % len(options)]


def _normalise(text: str) -> str:
    """As the server compares a seat's messages for duplicate text."""
    return _JS_SPACES.sub(" ", _JS_TRIM.sub("", text)).lower()


def _unsaid(options: list[str], shift: int, said: set[str], *parts: str) -> str:
    """As `_pick_shifted`, then on to the next line until one isn't in `said` (the
    persona's own earlier lines, normalised). Only when every line has been said is one
    repeated."""
    for step in range(len(options)):
        line = _pick_shifted(options, shift + step, *parts)
        if _normalise(line) not in said:
            return line
    return _pick_shifted(options, shift, *parts)


def _topic(ctx: dict[str, Any]) -> str | None:
    """The agenda item's title, or None when the context names none."""
    item = ctx.get("agenda_item") or {}
    if item.get("title"):
        return _flatten(item["title"])
    meeting = ctx.get("meeting") or {}
    agenda = meeting.get("agenda") or []
    if agenda and agenda[0].get("title"):
        return _flatten(agenda[0]["title"])
    return _flatten(meeting["title"]) if meeting.get("title") else None


def _fill(template: str, topic: str | None, quote: str = "") -> str:
    """`{topic}` as the title in curly quotes (its own quotes dropped), or "this"."""
    bare = _QUOTE_MARK.sub("", topic) if topic else ""
    named = f"\u201c{bare}\u201d" if bare else "this"
    return _PLACEHOLDER.sub(lambda m: named if m.group(1) == "topic" else quote, template)


def _quote_or_speaker(message: dict[str, Any], topic: str | None) -> str:
    """The last message as `{quote}`: a person's words in curly quotes, cut with "…"
    after QUOTE_WORDS words, or whose point it was. An agent's message is always named."""
    quote = (
        None
        if message.get("author_kind") == "agent"
        else quote_of(message.get("body") or "", topic or "")
    )
    if quote is not None and len(quote.removesuffix("...").split(" ")) >= _MIN_QUOTE_WORDS:
        cut = quote[: -len("...")] + "\u2026" if quote.endswith("...") else quote
        return f"\u201c{cut}\u201d"
    author = _flatten(message.get("author") or "")
    return f"{author}'s point" if author and author != "you" else "that point"


def respond(persona: str, ctx: dict[str, Any]) -> dict[str, Any]:
    return _answer_at(persona, ctx, 0)


def respond_fresh(persona: str, ctx: dict[str, Any]) -> dict[str, Any]:
    """The demo agent's answer, never repeating its own last message, which the server
    would refuse as duplicate text. A lone agent otherwise opens every turn with the
    same line. When `respond` would repeat it, the next line for the same situation is
    used instead."""
    me = (ctx.get("me") or {}).get("seat_id")
    mine = [m for m in ctx.get("recent_messages") or [] if m.get("seat_id") == me]
    previous = _normalise(mine[-1].get("body") or "") if mine else None
    first = _answer_at(persona, ctx, 0)
    if previous is None or _normalise(first["text"]) != previous:
        return first
    # Every option list has at most a few lines; the shift wraps around them.
    for shift in range(1, 4):
        following = _answer_at(persona, ctx, shift)
        if _normalise(following["text"]) != previous:
            return following
    return first


def _answer_at(persona: str, ctx: dict[str, Any], shift: int) -> dict[str, Any]:
    if persona not in PERSONAS:
        raise ValueError(f"unknown persona {persona!r}; expected one of {', '.join(PERSONAS)}")
    topic = _topic(ctx)
    topic_key = topic or ""
    me = (ctx.get("me") or {}).get("seat_id")
    recent = ctx.get("recent_messages") or []
    # System lines ("Agenda item started") are nobody's point to react to.
    others = [m for m in recent if m.get("seat_id") != me and m.get("kind") != "system"]
    last = others[-1] if others else None
    # A proposal is the persona's own when its message is one of the persona's.
    mine = [m for m in recent if me is not None and m.get("seat_id") == me]
    mine_ids = {m.get("id") for m in mine}
    said = {_normalise(m.get("body") or "") for m in mine}
    proposals = ctx.get("open_proposals") or []
    answer: dict[str, Any] = {"action": "message", "reply_to": None, "text": "", "vote": None}

    if proposals:
        proposal = proposals[-1]
        message_id = proposal.get("message_id")
        if message_id and message_id in mine_ids:
            # Its own proposal: it backs it, and doesn't reply to itself.
            answer["text"] = _unsaid(OWN_VOTE, shift, said, persona, proposal["id"])
            answer["vote"] = {"motion_id": proposal["id"], "choice": "yes"}
            return answer
        choice = _choice(persona, proposal.get("text") or "")
        answer["action"] = "reply" if message_id else "message"
        answer["reply_to"] = message_id or None
        lines = VOTE_LINES[persona][choice]
        answer["text"] = _unsaid(lines, shift, said, persona, proposal["id"])
        answer["vote"] = {"motion_id": proposal["id"], "choice": choice}
        return answer

    if last is None:
        action, template = _pick_shifted(OPENERS[persona], shift, persona, topic_key)
        answer["action"] = action
        answer["text"] = _fill(template, topic)
        return answer

    if last.get("kind") == "question":
        answer["action"] = "reply"
        answer["reply_to"] = last["id"]
        answer["text"] = _pick_shifted(REPLIES[persona], shift, persona, topic_key, last["id"])
        return answer

    # As with votes, a statement the persona has already made is skipped.
    quote = _quote_or_speaker(last, topic)
    statements = STATEMENTS[persona]
    for step in range(len(statements) + 1):
        # The last pass, every line said, falls back to the first pick.
        action, template = _pick_shifted(
            statements, shift + step % len(statements), persona, topic_key, last["id"]
        )
        answer["action"] = action
        answer["text"] = _fill(template, topic, quote)
        if _normalise(answer["text"]) not in said:
            break
    return answer
