"""Rule-based demo personas (spec §11): Pragmatist, Budget hawk, Community builder.

Self-contained on purpose (standard library only, plain dicts in and out) so the
server's demo agents can mirror the same rules. `respond(persona, ctx)` takes a turn
context as a dict (`TurnContext.model_dump(mode="json")`) and returns one answer in the
model's output shape. There is no randomness: the same context always gives the same
answer. Wording varies only with a hash of the persona, agenda item and last message.

Rules, in order:
1. An open proposal: vote on the latest one, replying to it when its message is known.
2. The last message from someone else is a question: reply to it.
3. Nobody has spoken: open the agenda item.
4. Otherwise: react to the last message (Pragmatist proposes, Budget hawk asks about
   cost, Community builder widens the circle).
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

# (action, template) per persona and situation. Templates use {topic} and {quote}.
OPENERS = {
    "pragmatist": [
        ("question", "On {topic}: what's the smallest step we can agree today, and who owns it?"),
        ("message", "Let's keep {topic} concrete: one owner, one date, one next step."),
    ],
    "budget_hawk": [
        (
            "question",
            "Before we decide anything on {topic}: what does it cost in total, "
            "and what's the cheapest option that works?",
        ),
        ("message", "For {topic}, I'd like a costed option on the table before we commit."),
    ],
    "community_builder": [
        (
            "message",
            "For {topic}, let's make sure everyone affected has a say before we settle it.",
        ),
        ("question", "On {topic}: who else should we bring in before we decide?"),
    ],
}
REPLIES = {
    "pragmatist": [
        'On {topic}, re "{quote}": yes, if we name an owner and a date. '
        "I'd start small and review in two weeks.",
        'Re "{quote}": doable for {topic} if we pick one option now and revisit later.',
    ],
    "budget_hawk": [
        "Re \"{quote}\": for {topic} I'd want a costed quote first, and a cap we won't exceed.",
        'On {topic}, re "{quote}": only if it fits the budget we already have.',
    ],
    "community_builder": [
        'Re "{quote}": for {topic}, could we ask for volunteers? '
        "More hands makes it cheaper and builds goodwill.",
        'On {topic}, re "{quote}": let\'s check with the people it affects most, then decide.',
    ],
}
STATEMENTS = {
    "pragmatist": [
        (
            "proposal",
            'Building on "{quote}": for {topic}, let\'s run it as a two-week trial '
            "with one owner, then review.",
        ),
        ("proposal", 'For {topic}: we go with "{quote}" now and check back at the next meeting.'),
    ],
    "budget_hawk": [
        ("question", '"{quote}" - what would that cost for {topic}, and is there a cheaper way?'),
        ("question", 'On {topic}: who pays for "{quote}", and what do we drop to afford it?'),
    ],
    "community_builder": [
        ("message", 'I like "{quote}". For {topic}, let\'s invite the wider group to help.'),
        ("message", 'On {topic}: "{quote}" could bring people together if we share the work.'),
    ],
}


def _vote(persona: str, proposal_text: str) -> tuple[str, str]:
    concrete = any(ch.isdigit() for ch in proposal_text)
    if persona == "pragmatist":
        if concrete:
            return "yes", 'Voting yes on "{quote}": it\'s concrete enough to act on for {topic}.'
        return "abstain", 'Abstaining on "{quote}" until it has an owner and a date for {topic}.'
    if persona == "budget_hawk":
        if _MONEY.search(proposal_text):
            return "no", 'Voting no on "{quote}" as it stands: for {topic} I\'d want a firm cap.'
        return "abstain", 'Abstaining on "{quote}" until we know what it costs for {topic}.'
    return "yes", 'Voting yes on "{quote}": it gets people working together on {topic}.'


# A lead-in built around a quote, as the templates write them: `Re "...":`,
# `On X, re "...":`, `Building on "...":`, `Voting no on "..." as it stands:`,
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
    flat = " ".join(text.split())
    topic = " ".join(topic.split())
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


def _fill(template: str, topic: str, quote: str | None, ref: str) -> str:
    """The template with the quote, or with `ref` (unquoted) when there's no quote."""
    if quote is None:
        if ref == "that point" and template.startswith('"{quote}"'):
            ref = "That point"
        return template.replace('"{quote}"', "{quote}").format(topic=topic, quote=ref)
    return template.format(topic=topic, quote=quote)


def _speaker(message: dict[str, Any]) -> str:
    author = " ".join((message.get("author") or "").split())
    return f"{author}'s point" if author and author != "you" else "that point"


def _pick(options: list[Any], *parts: str) -> Any:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return options[digest[0] % len(options)]


def _topic(ctx: dict[str, Any]) -> str:
    item = ctx.get("agenda_item") or {}
    if item.get("title"):
        return item["title"]
    meeting = ctx.get("meeting") or {}
    agenda = meeting.get("agenda") or []
    if agenda and agenda[0].get("title"):
        return agenda[0]["title"]
    return meeting.get("title") or "this item"


def respond(persona: str, ctx: dict[str, Any]) -> dict[str, Any]:
    if persona not in PERSONAS:
        raise ValueError(f"unknown persona {persona!r}; expected one of {', '.join(PERSONAS)}")
    topic = _topic(ctx)
    me = (ctx.get("me") or {}).get("seat_id")
    others = [m for m in ctx.get("recent_messages") or [] if m.get("seat_id") != me]
    last = others[-1] if others else None
    proposals = ctx.get("open_proposals") or []
    answer: dict[str, Any] = {"action": "message", "reply_to": None, "text": "", "vote": None}

    if proposals:
        proposal = proposals[-1]
        choice, template = _vote(persona, proposal.get("text", ""))
        answer["action"] = "reply" if proposal.get("message_id") else "message"
        answer["reply_to"] = proposal.get("message_id")
        quote = quote_of(proposal.get("text", ""), topic)
        answer["text"] = _fill(template, topic, quote, "the proposal")
        answer["vote"] = {"motion_id": proposal["id"], "choice": choice}
        return answer

    if last is None:
        action, template = _pick(OPENERS[persona], persona, topic)
        answer["action"] = action
        answer["text"] = template.format(topic=topic)
        return answer

    quote = quote_of(last.get("body", ""), topic)
    ref = _speaker(last)
    if last.get("kind") == "question":
        answer["action"] = "reply"
        answer["reply_to"] = last["id"]
        template = _pick(REPLIES[persona], persona, topic, last["id"])
        answer["text"] = _fill(template, topic, quote, ref)
        return answer

    action, template = _pick(STATEMENTS[persona], persona, topic, last["id"])
    answer["action"] = action
    answer["text"] = _fill(template, topic, quote, ref)
    return answer
