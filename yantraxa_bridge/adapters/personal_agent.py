"""The `personal_agent` adapter: the owner's own agent answers at `{url}/turn`.

The bridge POSTs the `TurnContext` as JSON and expects one answer object in the model's
output shape (`{"action", "text", "reply_to", "vote"}`, with real ids), or
`{"actions": [...]}` with up to five. No streaming. Anything unusable passes.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from urllib.parse import urlsplit

import httpx

from ..prompt import parse_action
from .base import AdapterInfo, Final, TurnContext, TurnEvent, pass_turn

log = logging.getLogger("yantraxa_bridge.adapters.personal_agent")

RUNTIME_LABEL = "Personal agent on owner's machine"
MAX_ACTIONS = 5
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class PersonalAgentAdapter:
    def __init__(
        self, url: str, *, timeout_s: float, http: httpx.AsyncClient | None = None
    ) -> None:
        self.url = url.rstrip("/") + "/turn"
        self.timeout_s = timeout_s
        self._http = http
        if urlsplit(self.url).hostname not in LOOPBACK_HOSTS:
            log.warning(
                "personal_agent_url is not on loopback: meeting content leaves this machine"
            )

    def describe(self) -> AdapterInfo:
        return AdapterInfo(
            RUNTIME_LABEL, ("message", "question", "reply", "proposal", "vote"), streams=False
        )

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def take_turn(self, ctx: TurnContext) -> AsyncIterator[TurnEvent]:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.timeout_s)
        try:
            response = await self._http.post(self.url, json=ctx.model_dump(mode="json"))
        except httpx.HTTPError as error:
            log.warning("personal agent unreachable (%s); passing", type(error).__name__)
            yield pass_turn()
            return
        if not response.is_success:
            log.warning("personal agent answered HTTP %d; passing", response.status_code)
            yield pass_turn()
            return
        try:
            data = response.json()
        except ValueError:
            log.warning("personal agent answered with something other than JSON; passing")
            yield pass_turn()
            return
        answers = data.get("actions") if isinstance(data, dict) and "actions" in data else [data]
        if not isinstance(answers, list):
            answers = []
        actions = [
            action
            for answer in answers[:MAX_ACTIONS]
            if (action := parse_action(answer, ctx)) is not None
        ]
        if not actions:
            log.warning("personal agent's answer had no usable action; passing")
            yield pass_turn()
            return
        yield Final(actions)
