import json

import httpx
import pytest
import respx

from yantraxa_bridge.adapters.base import Chunk, Final
from yantraxa_bridge.adapters.personal_agent import PersonalAgentAdapter

from .contexts import M2, MEETING_ID, P1, TURN_ID, make_context

URL = "http://127.0.0.1:8765"


async def collect(adapter, ctx=None):
    events = [event async for event in adapter.take_turn(ctx or make_context())]
    await adapter.aclose()
    return events


@respx.mock
async def test_round_trip_posts_the_context_and_reads_one_action():
    route = respx.post(f"{URL}/turn").mock(
        return_value=httpx.Response(
            200,
            json={
                "action": "reply",
                "reply_to": M2,
                "text": "Saugat can help fix it on Saturday.",
                "vote": {"motion_id": P1, "choice": "yes", "reason": "It's leaking now."},
            },
        )
    )
    events = await collect(PersonalAgentAdapter(URL + "/", timeout_s=5))
    assert not [e for e in events if isinstance(e, Chunk)]
    [final] = events
    assert isinstance(final, Final)
    [action] = final.actions
    assert action.action == "reply"
    assert action.reply_to == M2
    assert action.vote is not None and action.vote.reason == "It's leaking now."

    sent = json.loads(route.calls[0].request.content)
    assert sent["turn_id"] == TURN_ID
    assert sent["meeting"]["id"] == MEETING_ID
    assert sent["recent_messages"][1]["id"] == M2
    assert sent["open_proposals"][0]["id"] == P1
    assert sent["knowledge"]["approved_facts"] == ["Saugat can volunteer on Saturday mornings."]
    assert sent["max_output_chars"] == 600


@respx.mock
async def test_an_actions_list_is_accepted_and_bad_entries_dropped():
    respx.post(f"{URL}/turn").mock(
        return_value=httpx.Response(
            200,
            json={
                "actions": [
                    {"action": "question", "text": "Who has the shed key?"},
                    {"action": "dance", "text": "no"},
                    {"action": "proposal", "text": "Fix the roof first."},
                ]
            },
        )
    )
    [final] = await collect(PersonalAgentAdapter(URL, timeout_s=5))
    assert [a.action for a in final.actions] == ["question", "proposal"]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"action": "message"}),
        httpx.Response(200, json={"actions": []}),
        httpx.Response(200, json=["message"]),
    ],
)
@respx.mock
async def test_bad_answers_pass(response):
    respx.post(f"{URL}/turn").mock(return_value=response)
    [final] = await collect(PersonalAgentAdapter(URL, timeout_s=5))
    assert [a.action for a in final.actions] == ["pass"]


@respx.mock
async def test_unreachable_endpoint_passes():
    respx.post(f"{URL}/turn").mock(side_effect=httpx.ConnectError("refused"))
    [final] = await collect(PersonalAgentAdapter(URL, timeout_s=5))
    assert final.actions[0].action == "pass"


def test_describe():
    info = PersonalAgentAdapter(URL, timeout_s=5).describe()
    assert info.runtime_label == "Personal agent on owner's machine"
    assert info.streams is False
