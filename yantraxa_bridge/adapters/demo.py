"""The `demo` adapter: a deterministic persona from `demo_personas`, no model at all."""

from __future__ import annotations

from collections.abc import AsyncIterator

from .. import demo_personas
from ..prompt import parse_action
from .base import AdapterInfo, Final, TurnContext, TurnEvent, pass_turn

RUNTIME_LABEL = "Demo persona in the bridge"


class DemoAdapter:
    def __init__(self, persona: str) -> None:
        if persona not in demo_personas.PERSONAS:
            raise ValueError(
                f"unknown persona {persona!r}; expected one of {', '.join(demo_personas.PERSONAS)}"
            )
        self.persona = persona

    def describe(self) -> AdapterInfo:
        return AdapterInfo(
            f"{RUNTIME_LABEL} ({demo_personas.NAMES[self.persona]})",
            ("message", "question", "reply", "proposal", "vote"),
            streams=False,
        )

    async def aclose(self) -> None:
        pass

    async def take_turn(self, ctx: TurnContext) -> AsyncIterator[TurnEvent]:
        answer = demo_personas.respond(self.persona, ctx.model_dump(mode="json"))
        action = parse_action(answer, ctx)
        yield Final([action]) if action is not None else pass_turn()
