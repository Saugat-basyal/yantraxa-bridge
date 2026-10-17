"""The bridge's `/ws/agent` client: auth, heartbeats, acks and resends, reconnects.

What the client does on each close code (see protocol.py):

- 1000, the owner disconnected the agent: stop.
- 4001, the credential is bad or revoked: the client asks `GET /v1/agent/me` with the
  credential first. Only a 401 there stops it (the CLI runs `diagnose`); a 200 or no answer
  means the 4001 was not a verdict on the credential, so it reconnects with backoff (0.2.1).
- 4009, a newer bridge authenticated as this agent: stop, so two bridges don't keep
  knocking each other off.
- 4003, paused by the owner, 4008, auth didn't finish, and anything else: reconnect with
  backoff.

Each failed attempt goes to `on_error` with its reason, and each connection to
`on_connected`, so `run` can record them for `status`, `doctor` and `pair`.

Project requests (0.2.0): the client advertises `projects` in `auth`; each
`project_request` runs the turn handler's `take_project_turn` under a hard timeout inside
the request's deadline, and goes back as one `project_response` or `project_pass`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import re
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import httpx
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidURI

from . import __version__
from .config import BridgeConfig, normalise_server_url, ws_url, yantraxa_home
from .credentials import Credentials
from .protocol import (
    ACKED_BRIDGE_TYPES,
    ACKED_SERVER_TYPES,
    BRIDGE_CAPABILITIES,
    CLOSE_AGENT_PAUSED,
    CLOSE_AUTH_INCOMPLETE,
    CLOSE_NORMAL,
    CLOSE_SUPERSEDED,
    CLOSE_UNAUTHORIZED,
    DEDUPE_IDS,
    HEARTBEAT_INTERVAL_S,
    MAX_FRAME_BYTES,
    PROJECT_TIMEOUT_MAX_S,
    PROJECT_TIMEOUT_MIN_S,
    RATE_LIMIT_FRAMES,
    RATE_LIMIT_WINDOW_S,
    SERVER_AUTH_TIMEOUT_S,
    ActivityState,
    MessageRow,
    ProjectPassReason,
    ProjectRequestPayload,
    ProtocolError,
    TurnAction,
    TurnRequestPayload,
    adapter_type_wire,
    make_frame,
    parse_server_frame,
)

log = logging.getLogger("yantraxa_bridge.client")

# Frames waiting for an ack while the bridge is offline; the oldest are dropped past this.
MAX_PENDING = 200
RECENT_MESSAGES = 50


class StopReason(StrEnum):
    DISCONNECTED = "disconnected"  # 1000: the owner disconnected the agent
    UNAUTHORIZED = "unauthorized"  # 4001, and `/v1/agent/me` says 401: bad or revoked credential
    SUPERSEDED = "superseded"  # 4009: another bridge took over
    STOPPED = "stopped"  # stop() was called


FINAL_CLOSES = {
    CLOSE_NORMAL: StopReason.DISCONNECTED,
    CLOSE_UNAUTHORIZED: StopReason.UNAUTHORIZED,
    CLOSE_SUPERSEDED: StopReason.SUPERSEDED,
}


@dataclass(frozen=True)
class Timing:
    heartbeat_s: float = HEARTBEAT_INTERVAL_S
    # No pong for this long means the connection is dead: reconnect.
    pong_timeout_s: float = 2 * HEARTBEAT_INTERVAL_S + 5
    ack_timeout_s: float = 5.0
    max_resends: int = 3
    # Longer than the server's own auth timeout, so the server gives its verdict (and records
    # the attempt) before the bridge gives up.
    auth_timeout_s: float = SERVER_AUTH_TIMEOUT_S + 5
    backoff_initial_s: float = 1.0
    backoff_max_s: float = 60.0
    # A little under the server's 30 per 10 s, so the bridge never trips it.
    rate_frames: int = RATE_LIMIT_FRAMES - 4
    rate_window_s: float = RATE_LIMIT_WINDOW_S


class Backoff:
    """1 s, 2 s, 4 s … capped at 60 s, each ±20% jitter so bridges don't reconnect in step."""

    def __init__(
        self, initial: float, maximum: float, rng: Callable[[], float] = random.random
    ) -> None:
        self.initial, self.maximum, self.rng = initial, maximum, rng
        self.attempt = 0

    def next(self) -> float:
        base = min(self.maximum, self.initial * 2**self.attempt)
        self.attempt += 1
        return base * (0.8 + 0.4 * self.rng())

    def reset(self) -> None:
        self.attempt = 0


class RateLimiter:
    """At most `frames` sends in any `window_s`; `acquire` waits for room."""

    def __init__(self, frames: int, window_s: float) -> None:
        self.frames, self.window_s = frames, window_s
        self._sent: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            while True:
                now = loop.time()
                while self._sent and now - self._sent[0] >= self.window_s:
                    self._sent.popleft()
                if len(self._sent) < self.frames:
                    self._sent.append(now)
                    return
                await asyncio.sleep(self.window_s - (now - self._sent[0]))


class LruSet:
    def __init__(self, size: int) -> None:
        self.size = size
        self._items: OrderedDict[str, None] = OrderedDict()

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def add(self, key: str) -> None:
        self._items[key] = None
        self._items.move_to_end(key)
        while len(self._items) > self.size:
            self._items.popitem(last=False)


@dataclass
class Pending:
    frame: dict[str, Any]
    sent_at: float | None = None  # None: not sent on the current connection yet
    resends: int = 0


@dataclass
class MeetingState:
    meeting_id: str
    code: str
    seat_id: str
    knowledge_profile: Any
    recent: deque[MessageRow] = field(default_factory=lambda: deque(maxlen=RECENT_MESSAGES))
    # M5: the knowledge on the latest `turn_request` that carried one. It is the resolved
    # policy (mandate included), so once known it outranks the `meeting_joined` profile.
    turn_knowledge: Any = None
    has_turn_knowledge: bool = False

    @property
    def returns_summaries(self) -> bool:
        profile = self.turn_knowledge if self.has_turn_knowledge else self.knowledge_profile
        return isinstance(profile, dict) and profile.get("return_summaries_to_agent") is True


class TurnIO:
    """What a turn handler can send while it works on a turn."""

    def __init__(self, client: BridgeClient, request: TurnRequestPayload) -> None:
        self.client = client
        self.request = request
        self.meeting = client.meetings.get(request.meeting_id)

    async def activity(self, state: ActivityState) -> None:
        meeting_id = self.request.meeting_id
        await self.client.send(
            "activity", {"meeting_id": meeting_id, "state": state}, meeting_id=meeting_id
        )

    async def chunk(self, text: str) -> None:
        meeting_id = self.request.meeting_id
        await self.client.send(
            "turn_chunk",
            {"meeting_id": meeting_id, "turn_id": self.request.turn_id, "text": text},
            meeting_id=meeting_id,
        )


@dataclass(frozen=True)
class ProjectAnswer:
    """A project request's answer: the reply's `text`, or the `reason` it passes."""

    text: str | None = None
    reason: ProjectPassReason = "declined"


class TurnHandler(Protocol):
    """Answers a `turn_request` (Task 4 plugs the adapters in here), and from 0.2.0 a
    `project_request`; a handler without `take_project_turn` passes `unsupported`."""

    async def take_turn(self, request: TurnRequestPayload, io: TurnIO) -> list[TurnAction]: ...


class PassTurnHandler:
    """Every turn is a pass: the default, and `interactive-mcp`, whose agent answers
    through its MCP client instead. Project requests pass as `unsupported`."""

    async def take_turn(self, request: TurnRequestPayload, io: TurnIO) -> list[TurnAction]:
        return [TurnAction(action="pass")]

    async def take_project_turn(
        self, request: ProjectRequestPayload, agent: dict[str, Any] | None
    ) -> ProjectAnswer:
        return ProjectAnswer(reason="unsupported")


PASS = [TurnAction(action="pass")]
# A project answer must be sent this long before the request's deadline.
PROJECT_DEADLINE_MARGIN_S = 1.0
_SUMMARY_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
REVIEW_RATINGS = {
    "yes": "Yes, the agent represented them.",
    "partially": "Partially.",
    "no": "No, the agent did not represent them.",
}


def _review_section(review: Any) -> str:
    """The owner's representation review as a markdown section, or "" when there is none."""
    if not isinstance(review, dict) or review.get("rating") not in REVIEW_RATINGS:
        return ""
    lines = [
        "",
        "",
        "## Representation review",
        "",
        f"Did the agent represent its owner? {REVIEW_RATINGS[review['rating']]}",
    ]
    correction = review.get("correction")
    if isinstance(correction, str) and correction.strip():
        lines += ["", "Owner's correction:", ""]
        lines += [f"> {line}" if line else ">" for line in correction.strip().splitlines()]
    return "\n".join(lines) + "\n"


class BridgeClient:
    def __init__(
        self,
        config: BridgeConfig,
        credentials: Credentials,
        *,
        home: Path | None = None,
        turn_handler: TurnHandler | None = None,
        timing: Timing | None = None,
        http: httpx.AsyncClient | None = None,
        on_connected: Callable[[dict[str, Any]], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        if not config.server_url:
            raise ValueError("config has no server_url; run `yantraxa-bridge pair` first")
        self.config = config
        self.credentials = credentials
        self.home = home or yantraxa_home()
        self.turn_handler: TurnHandler = turn_handler or PassTurnHandler()
        self.timing = timing or Timing()
        self.server_url = normalise_server_url(config.server_url)
        self.url = ws_url(config.server_url)
        self._http = http
        self._on_connected = on_connected
        self._on_error = on_error

        self.agent: dict[str, Any] | None = None
        self.permissions: dict[str, Any] = {}
        self.meetings: dict[str, MeetingState] = {}
        self.invitations: dict[str, Any] = {}
        self.errors: list[tuple[str, str | None]] = []
        self.pending: OrderedDict[str, Pending] = OrderedDict()
        self.attempts = 0

        self._seen = LruSet(DEDUPE_IDS)
        self._turns: set[str] = set()
        self._project_turns: set[int] = set()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._limiter = RateLimiter(self.timing.rate_frames, self.timing.rate_window_s)
        self._backoff = Backoff(self.timing.backoff_initial_s, self.timing.backoff_max_s)
        self._ws: ClientConnection | None = None
        self._stopping = asyncio.Event()
        self._self_closed = False
        self._last_pong = 0.0

    @property
    def connected(self) -> bool:
        return self._ws is not None

    # ------------------------------------------------------------------ lifecycle

    async def run(self) -> StopReason:
        """Connect and stay connected until a final close code or `stop()`."""
        try:
            while not self._stopping.is_set():
                self.attempts += 1
                reason = await self._connect_once()
                if reason is StopReason.UNAUTHORIZED and not await self._credential_refused():
                    reason = None
                if reason is not None:
                    return reason
                if self._stopping.is_set():
                    break
                delay = self._backoff.next()
                log.info("reconnecting in %.1fs", delay)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stopping.wait(), delay)
            return StopReason.STOPPED
        finally:
            await self._shutdown()

    async def stop(self) -> None:
        self._stopping.set()
        if self._ws is not None:
            self._self_closed = True
            with contextlib.suppress(Exception):
                await self._ws.close(CLOSE_NORMAL, "bridge stopping")

    async def _shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def _connect_once(self) -> StopReason | None:
        """One connection; a StopReason means stop, None means reconnect."""
        self._self_closed = False
        try:
            async with connect(
                self.url,
                open_timeout=self.timing.auth_timeout_s,
                max_size=MAX_FRAME_BYTES,
                user_agent_header=f"yantraxa-bridge/{__version__}",
            ) as ws:
                return await self._session(ws)
        except ConnectionClosed as closed:
            return self._classify(closed.rcvd.code if closed.rcvd else None)
        except (OSError, TimeoutError, InvalidHandshake, InvalidURI) as error:
            self._failed(f"can't connect to {self.url}: {error or type(error).__name__}")
            return None

    def _failed(self, reason: str) -> None:
        """A failed attempt: logged, and handed to `on_error`."""
        log.warning("%s", reason)
        if self._on_error is not None:
            self._on_error(reason)

    def _classify(self, code: int | None) -> StopReason | None:
        if self._stopping.is_set():
            return StopReason.STOPPED
        if self._self_closed:
            return None
        if code in FINAL_CLOSES:
            reason = FINAL_CLOSES[code]
            self._failed(f"server closed the connection ({code}): {reason.value}")
            return reason
        if code == CLOSE_AGENT_PAUSED:
            self._failed("agent is paused by its owner; retrying with backoff")
        elif code == CLOSE_AUTH_INCOMPLETE:
            self._failed(f"the server closed before authentication finished ({code}); reconnecting")
        else:
            self._failed(f"connection closed ({code}); reconnecting")
        return None

    async def _credential_refused(self) -> bool:
        """After a 4001: whether `GET /v1/agent/me` refuses the credential too (HTTP 401).
        A 4001 can come from something other than a verdict on the credential (an older
        server's auth timeout did that), and a bridge that stops on it stays down for good."""
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=15)
        try:
            response = await self._http.get(
                f"{self.server_url}/v1/agent/me",
                headers={"Authorization": self.credentials.bearer()},
            )
        except httpx.HTTPError as error:
            self._failed(
                f"the server closed with 4001 and checking the credential failed "
                f"({error or type(error).__name__}); retrying"
            )
            return False
        if response.status_code == 401:
            return True
        if response.status_code == 200:
            self._failed("the server closed with 4001 but the credential is valid; retrying")
        else:
            self._failed(
                f"the server closed with 4001 and checking the credential gave "
                f"HTTP {response.status_code}; retrying"
            )
        return False

    async def _session(self, ws: ClientConnection) -> StopReason | None:
        auth = make_frame(
            "auth",
            {
                "agent_id": self.credentials.agent_id,
                "secret": self.credentials.secret_value(),
                "bridge_version": __version__,
                "adapter_type": adapter_type_wire(self.config.adapter),
                "capabilities": list(BRIDGE_CAPABILITIES),
            },
        )
        await self._limiter.acquire()
        await ws.send(json.dumps(auth))
        try:
            raw = await asyncio.wait_for(ws.recv(), self.timing.auth_timeout_s)
        except TimeoutError:
            self._failed(f"no auth_ok within {self.timing.auth_timeout_s:.0f}s")
            self._self_closed = True
            await ws.close()
            return None
        try:
            first = parse_server_frame(raw)
        except ProtocolError as error:
            self._failed(f"unexpected first frame: {error}")
            self._self_closed = True
            await ws.close()
            return None
        if first.type != "auth_ok":
            self._failed(f"expected auth_ok, got {first.type}")
            self._self_closed = True
            await ws.close()
            return None

        self.agent = first.payload.agent
        self.permissions = first.payload.permissions
        self._backoff.reset()
        self._ws = ws
        if self._stopping.is_set():  # stop() ran before the socket was registered
            await ws.close(CLOSE_NORMAL, "bridge stopping")
            self._ws = None
            return StopReason.STOPPED
        self._last_pong = asyncio.get_running_loop().time()
        log.info("connected as @%s (%s)", self.agent.get("handle"), self.agent.get("id"))
        if self._on_connected is not None:
            self._on_connected(self.agent)

        workers = [
            asyncio.create_task(self._heartbeat(ws)),
            asyncio.create_task(self._resender(ws)),
        ]
        try:
            await self._flush_pending()
            async for message in ws:
                await self._receive(message)
        except ConnectionClosed:
            pass
        finally:
            self._ws = None
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            for pending in self.pending.values():
                pending.sent_at = None
        return self._classify(ws.close_code)

    # ------------------------------------------------------------------ sending

    async def send(
        self, type_: str, payload: dict[str, Any], *, meeting_id: str | None = None
    ) -> str:
        """Send a frame the server acks; it is resent until acked. Returns its id."""
        if type_ not in ACKED_BRIDGE_TYPES:
            raise ValueError(f"{type_} is not an acked frame")
        frame = make_frame(type_, payload, meeting_id=meeting_id)
        self.pending[frame["id"]] = Pending(frame)
        while len(self.pending) > MAX_PENDING:
            dropped, _ = self.pending.popitem(last=False)
            log.warning("dropping unsent frame %s: too many waiting", dropped)
        if self._ws is not None:
            await self._transmit(self.pending[frame["id"]])
        return frame["id"]

    async def _transmit(self, pending: Pending) -> None:
        ws = self._ws
        if ws is None:
            return
        await self._limiter.acquire()
        pending.sent_at = asyncio.get_running_loop().time()
        with contextlib.suppress(ConnectionClosed):
            await ws.send(json.dumps(pending.frame))

    async def _send_unacked(
        self, ws: ClientConnection, type_: str, payload: dict[str, Any]
    ) -> None:
        await self._limiter.acquire()
        with contextlib.suppress(ConnectionClosed):
            await ws.send(json.dumps(make_frame(type_, payload)))

    async def _flush_pending(self) -> None:
        """After (re)connecting, send everything still unacked, with the same ids."""
        for pending in list(self.pending.values()):
            if pending.sent_at is None:
                await self._transmit(pending)

    async def _heartbeat(self, ws: ClientConnection) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(self.timing.heartbeat_s)
            if loop.time() - self._last_pong > self.timing.pong_timeout_s:
                self._failed(f"no pong for {self.timing.pong_timeout_s:.0f}s; reconnecting")
                self._self_closed = True
                await ws.close(4000, "heartbeat timeout")
                return
            await self._send_unacked(ws, "ping", {})

    async def _resender(self, ws: ClientConnection) -> None:
        loop = asyncio.get_running_loop()
        tick = min(1.0, self.timing.ack_timeout_s / 4)
        while True:
            await asyncio.sleep(tick)
            now = loop.time()
            for frame_id, pending in list(self.pending.items()):
                if pending.sent_at is None or now - pending.sent_at < self.timing.ack_timeout_s:
                    continue
                if pending.resends >= self.timing.max_resends:
                    self.pending.pop(frame_id, None)
                    log.warning(
                        "giving up on %s frame %s: no ack after %d resends",
                        pending.frame["type"],
                        frame_id,
                        pending.resends,
                    )
                    continue
                pending.resends += 1
                await self._transmit(pending)

    # ------------------------------------------------------------------ receiving

    async def _receive(self, raw: str | bytes) -> None:
        try:
            frame = parse_server_frame(raw)
        except ProtocolError as error:
            log.warning("ignoring frame: %s", error)
            return
        ws = self._ws
        if frame.id in self._seen:
            # Already handled; ack again in case our first ack was lost.
            if frame.type in ACKED_SERVER_TYPES and ws is not None:
                await self._send_unacked(ws, "ack", {"id": frame.id})
            return
        self._seen.add(frame.id)
        await self._dispatch(frame)
        if frame.type in ACKED_SERVER_TYPES and ws is not None:
            await self._send_unacked(ws, "ack", {"id": frame.id})

    async def _dispatch(self, frame: Any) -> None:
        payload = frame.payload
        match frame.type:
            case "pong":
                self._last_pong = asyncio.get_running_loop().time()
            case "ack":
                self.pending.pop(payload.id, None)
            case "error":
                self.errors.append((payload.code, payload.ref_id))
                if payload.ref_id:
                    # Final: the server acks after the error, so stop resending now.
                    self.pending.pop(payload.ref_id, None)
                log.warning("server error %s (frame %s)", payload.code, payload.ref_id)
            case "auth_ok":
                self.agent, self.permissions = payload.agent, payload.permissions
            case "invitation":
                self._on_invitation(payload)
            case "meeting_joined":
                self.meetings[payload.meeting_id] = MeetingState(
                    payload.meeting_id, payload.code, payload.seat_id, payload.knowledge_profile
                )
                log.info("joined meeting %s", payload.code)
            case "message_posted":
                self._on_message(payload.message)
            case "event":
                log.debug("event %s in %s", payload.event.type, payload.event.meeting_id)
            case "turn_request":
                self._on_turn_request(payload)
            case "project_request":
                self._on_project_request(payload)
            case "summary":
                self._on_summary(payload)

    def _on_invitation(self, payload: Any) -> None:
        self.invitations[payload.invitation_id] = payload
        meeting = payload.meeting
        if payload.auto_accepted:
            log.info(
                "invited to %s (%s); accepted by the owner's mandate", meeting.code, meeting.title
            )
        elif self.config.auto_accept_pending:
            log.info("invited to %s (%s); accepting", meeting.code, meeting.title)
            self._spawn(self._accept_invitation(payload.invitation_id))
        else:
            log.info(
                "invited to %s (%s); waiting for the owner in Yantraxa",
                meeting.code,
                meeting.title,
            )

    async def _accept_invitation(self, invitation_id: str) -> None:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=15)
        url = f"{self.server_url}/v1/agent/invitations/{invitation_id}/accept"
        try:
            response = await self._http.post(
                url, headers={"Authorization": self.credentials.bearer()}
            )
        except httpx.HTTPError as error:
            log.warning("accepting invitation %s failed: %s", invitation_id, error)
            return
        if response.is_success:
            log.info("accepted invitation %s", invitation_id)
        else:
            log.warning(
                "accepting invitation %s failed: HTTP %d", invitation_id, response.status_code
            )

    def _on_message(self, message: MessageRow) -> None:
        state = self.meetings.get(message.meeting_id)
        if state is None or message.seat_id == state.seat_id:
            return  # our own messages come back too
        state.recent.append(message)

    def _on_turn_request(self, request: TurnRequestPayload) -> None:
        if request.turn_id in self._turns:
            return  # a resend of a turn already in progress
        self._turns.add(request.turn_id)
        state = self.meetings.get(request.meeting_id)
        if state is not None and "knowledge" in request.model_fields_set:
            state.turn_knowledge, state.has_turn_knowledge = request.knowledge, True
        self._spawn(self._run_turn(request))

    async def _run_turn(self, request: TurnRequestPayload) -> None:
        try:
            actions = await asyncio.wait_for(
                self.turn_handler.take_turn(request, TurnIO(self, request)),
                self.config.timeout_s,
            )
        except TimeoutError:
            log.warning(
                "turn %s timed out after %.0fs; passing", request.turn_id, self.config.timeout_s
            )
            actions = PASS
        except Exception:
            log.exception("turn %s failed; passing", request.turn_id)
            actions = PASS
        try:
            await self.send(
                "turn_response",
                {"meeting_id": request.meeting_id, "turn_id": request.turn_id, "actions": actions},
                meeting_id=request.meeting_id,
            )
        except ProtocolError as error:
            log.warning("turn %s produced a bad response (%s); passing", request.turn_id, error)
            await self.send(
                "turn_response",
                {"meeting_id": request.meeting_id, "turn_id": request.turn_id, "actions": PASS},
                meeting_id=request.meeting_id,
            )
        finally:
            self._turns.discard(request.turn_id)

    def _on_project_request(self, request: ProjectRequestPayload) -> None:
        if request.turn_id in self._project_turns:
            return  # a resend of a request already in progress
        self._project_turns.add(request.turn_id)
        self._spawn(self._run_project(request))

    def _project_budget(self, request: ProjectRequestPayload) -> float:
        """Seconds the handler may take: the configured timeout, and the server's
        `timeout_ms` (and `deadline_at`, if this clock agrees it's sooner) clamped to
        5 to 55 s, so a clock that is off can't skip the turn or overrun the server's timer."""
        limits = request.limits
        server: list[float] = []
        if limits.timeout_ms is not None:
            server.append(limits.timeout_ms / 1000 - PROJECT_DEADLINE_MARGIN_S)
        if limits.deadline_at:
            try:
                at = datetime.fromisoformat(limits.deadline_at.replace("Z", "+00:00"))
                server.append((at - datetime.now(UTC)).total_seconds() - PROJECT_DEADLINE_MARGIN_S)
            except ValueError:
                pass
        if not server:
            return self.config.timeout_s
        clamped = min(max(min(server), PROJECT_TIMEOUT_MIN_S), PROJECT_TIMEOUT_MAX_S)
        return min(self.config.timeout_s, clamped)

    async def _run_project(self, request: ProjectRequestPayload) -> None:
        handler = getattr(self.turn_handler, "take_project_turn", None)
        budget = self._project_budget(request)
        try:
            if handler is None:
                answer = ProjectAnswer(reason="unsupported")
            else:
                answer = await asyncio.wait_for(handler(request, self.agent), budget)
        except TimeoutError:
            log.warning("project turn %s timed out after %.0fs; passing", request.turn_id, budget)
            answer = ProjectAnswer(reason="timeout")
        except Exception:
            log.exception("project turn %s failed; passing", request.turn_id)
            answer = ProjectAnswer(reason="error")
        try:
            if answer.text and answer.text.strip():
                await self.send(
                    "project_response",
                    {"turn_id": request.turn_id, "text": answer.text},
                )
            else:
                await self.send(
                    "project_pass", {"turn_id": request.turn_id, "reason": answer.reason}
                )
        except ProtocolError as error:
            log.warning(
                "project turn %s produced a bad answer (%s); passing", request.turn_id, error
            )
            await self.send("project_pass", {"turn_id": request.turn_id, "reason": "error"})
        finally:
            self._project_turns.discard(request.turn_id)

    def _on_summary(self, payload: Any) -> None:
        """Writes `summaries/<code>.md`; a later version (say, once the owner has reviewed
        the agent) overwrites it.

        The server only sends summaries the knowledge policy and mandate allow, so a
        meeting this bridge doesn't know (it restarted since joining) is kept too. For a
        known one the bridge also checks the latest knowledge it was given.
        """
        state = self.meetings.get(payload.meeting_id)
        if state is not None and not state.returns_summaries:
            log.info(
                "summary for %s not kept: the profile doesn't return summaries", payload.meeting_id
            )
            return
        extra = payload.model_extra or {}
        text = extra.get("markdown") or extra.get("text")
        if not isinstance(text, str):
            text = json.dumps(extra, indent=2)
        if review := _review_section(extra.get("representation_review")):
            text = text.rstrip("\n") + review
        code = state.code if state is not None else extra.get("code")
        name = code if isinstance(code, str) and _SUMMARY_NAME.match(code) else payload.meeting_id
        folder = self.home / "summaries"
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = folder / f"{name}.md"
        path.write_text(text)
        log.info("summary for %s written to %s", name, path)

    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
