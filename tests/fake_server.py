"""A scriptable stand-in for the server's /ws/agent gateway."""

import asyncio
import json
import uuid
from collections.abc import Callable
from typing import Any

from websockets.asyncio.server import ServerConnection, serve

AGENT = {"id": "ag_abc234", "handle": "saugat.ai", "state": "active"}


def frame(
    type_: str, payload: dict[str, Any], *, id_: str | None = None, meeting_id: str | None = None
) -> str:
    out: dict[str, Any] = {
        "id": id_ or str(uuid.uuid4()),
        "type": type_,
        "ts": "2026-10-11T20:00:00.000Z",
        "payload": payload,
    }
    if meeting_id:
        out["meeting_id"] = meeting_id
    return json.dumps(out)


class FakeServer:
    def __init__(self) -> None:
        self.received: list[dict[str, Any]] = []
        self.auth_frames: list[dict[str, Any]] = []
        self.connections = 0
        # One entry per connection: a close code to answer `auth` with, or None for auth_ok.
        self.auth_plan: list[int | None] = []
        # Frame types (or ids) whose acks are withheld, with how many times to withhold.
        self.withhold: dict[str, int] = {}
        # Called for each acked frame; may send extra frames (e.g. an error) before the ack.
        self.before_ack: Callable[[dict[str, Any]], str | None] | None = None
        self.mute_pong = False
        self.ws: ServerConnection | None = None
        self.authed = asyncio.Event()
        self._server: Any = None
        self.port = 0

    async def __aenter__(self) -> "FakeServer":
        self._server = await serve(self._handler, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._server.close()
        await self._server.wait_closed()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def _handler(self, ws: ServerConnection) -> None:
        self.connections += 1
        auth = json.loads(await ws.recv())
        self.auth_frames.append(auth)
        code = self.auth_plan.pop(0) if self.auth_plan else None
        if code is not None:
            await ws.close(code, "planned")
            return
        await ws.send(frame("auth_ok", {"agent": AGENT, "permissions": {"post_messages": True}}))
        self.ws = ws
        self.authed.set()
        async for raw in ws:
            received = json.loads(raw)
            self.received.append(received)
            kind = received["type"]
            if kind == "ping":
                if not self.mute_pong:
                    await ws.send(frame("pong", {}))
                continue
            if kind == "ack":
                continue
            if self._withheld(received):
                continue
            if self.before_ack:
                extra = self.before_ack(received)
                if extra:
                    await ws.send(extra)
            await ws.send(frame("ack", {"id": received["id"]}))

    def _withheld(self, received: dict[str, Any]) -> bool:
        for key in (received["id"], received["type"]):
            left = self.withhold.get(key, 0)
            if left:
                self.withhold[key] = left - 1
                return True
        return False

    async def push(self, type_: str, payload: dict[str, Any], **kw: Any) -> str:
        id_ = kw.pop("id_", None) or str(uuid.uuid4())
        assert self.ws is not None
        await self.ws.send(frame(type_, payload, id_=id_, **kw))
        return id_

    async def push_raw(self, raw: str) -> None:
        assert self.ws is not None
        await self.ws.send(raw)

    def of_type(self, type_: str) -> list[dict[str, Any]]:
        return [f for f in self.received if f["type"] == type_]

    async def wait_for(self, predicate: Callable[[], bool], timeout: float = 3.0) -> None:
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(0.01)
