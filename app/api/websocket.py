"""The WebSocket interface.

Carries the events of §8 to clients in the wire format of §9, so a client can
show what the assistant is doing without polling. Clients may also send
messages, which is what makes a remote client a participant rather than only an
observer.

One subscription to the event bus fans out to every connected socket. A client
that stalls is disconnected rather than allowed to hold up the assistant: this
is a live status feed, and stale state is worse than no state.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from fastapi import WebSocket, WebSocketDisconnect

from app.api.audioStream import (
    AudioStreamError,
    IncomingAudio,
    chunkAudio,
    replyHeader,
    streamedReplyHeader,
)
from app.assistant.session import Session
from app.audio.audioBuffer import AudioBuffer
from app.audio.output import AudioOutput
from app.events import Event, EventBus

if TYPE_CHECKING:
    from app.api.service import AssistantService

logger = logging.getLogger(__name__)

# A client this far behind is not going to catch up.
MAXIMUM_QUEUED_EVENTS = 64


class Connection:
    """One connected client, with its own conversation."""

    __slots__ = ("audio", "identifier", "queue", "session", "socket", "wantsAudio")

    def __init__(self, socket: WebSocket, identifier: str, session: Session) -> None:
        self.socket = socket
        self.identifier = identifier
        self.session = session
        self.audio = IncomingAudio()
        # Set when the client sends audio, since a client that speaks expects
        # to be spoken back to rather than have the reply played in the room
        # the server happens to be in.
        self.wantsAudio = False
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=MAXIMUM_QUEUED_EVENTS
        )

    def offer(self, payload: dict[str, Any]) -> bool:
        """Queue an event. Returns False if the client is too far behind."""
        try:
            self.queue.put_nowait(payload)
        except asyncio.QueueFull:
            return False
        return True


class ConnectionManager:
    """Fans assistant events out to every connected client."""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._connections: dict[str, Connection] = {}
        self._subscription: Any = None
        self._counter = 0

    @property
    def connectionCount(self) -> int:
        return len(self._connections)

    def start(self) -> None:
        """Subscribe to the bus once, however many clients connect."""
        if self._subscription is None:
            self._subscription = self._bus.subscribeAll(self._onEvent)

    def stop(self) -> None:
        if self._subscription is not None:
            self._subscription.unsubscribe()
            self._subscription = None

    def _onEvent(self, event: Event) -> None:
        """Called by the bus. Never blocks, so a slow client cannot stall a turn."""
        if not self._connections:
            return

        payload = event.toDict()
        for connection in list(self._connections.values()):
            if not connection.offer(payload):
                logger.warning("Client %s is too far behind; dropping it", connection.identifier)
                self._connections.pop(connection.identifier, None)

    def add(self, socket: WebSocket, session: Session) -> Connection:
        self._counter += 1
        connection = Connection(socket, f"client-{self._counter}", session)
        self._connections[connection.identifier] = connection
        logger.info("Client %s connected (%d total)", connection.identifier, len(self._connections))
        return connection

    def remove(self, connection: Connection) -> None:
        if self._connections.pop(connection.identifier, None) is not None:
            logger.info(
                "Client %s disconnected (%d remaining)",
                connection.identifier,
                len(self._connections),
            )


async def handleConnection(
    socket: WebSocket,
    manager: ConnectionManager,
    service: AssistantService,
) -> None:
    """Serve one client until it disconnects.

    Sending and receiving run as separate tasks, so an event can reach the
    client while it is in the middle of saying something.
    """
    await socket.accept()
    # Each client gets its own conversation, so two people talking to the
    # assistant from different rooms do not share context.
    connection = manager.add(socket, service.createSession())

    await _send(
        socket,
        {
            "type": "assistant.connected",
            "sessionId": connection.session.sessionId,
            "state": service.runtime.state.value,
        },
    )

    sender = asyncio.create_task(_sendEvents(connection))
    receiver = asyncio.create_task(_receiveCommands(connection, service))

    try:
        _, pending = await asyncio.wait(
            {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    finally:
        manager.remove(connection)
        service.removeSession(connection.session.sessionId)


async def _sendEvents(connection: Connection) -> None:
    """Forward queued events to the client until the socket fails."""
    while True:
        payload = await connection.queue.get()
        if not await _send(connection.socket, payload):
            return


async def _receiveCommands(connection: Connection, service: AssistantService) -> None:
    """Act on what a client sends.

    Text frames carry JSON control messages; binary frames carry raw PCM. The
    two channels are separate so audio does not pay for base64.
    """
    while True:
        try:
            frame = await connection.socket.receive()
        except (WebSocketDisconnect, RuntimeError):
            return

        if frame.get("type") == "websocket.disconnect":
            return

        payload = frame.get("bytes")
        if payload is not None:
            try:
                connection.audio.add(payload)
            except AudioStreamError as error:
                await _send(connection.socket, _error(str(error)))
            continue

        raw = frame.get("text")
        if raw is None:
            continue

        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            await _send(connection.socket, _error("That was not valid JSON."))
            continue

        if not isinstance(message, dict):
            await _send(connection.socket, _error("Expected a JSON object."))
            continue

        await _handleCommand(connection, service, message)


async def _handleCommand(
    connection: Connection, service: AssistantService, message: dict[str, Any]
) -> None:
    kind = str(message.get("type") or "")

    if kind == "assistant.message":
        text = str(message.get("text") or "").strip()
        if not text:
            await _send(connection.socket, _error("A message needs some text."))
            return

        result = await service.handleText(
            text,
            speak=bool(message.get("speak", False)),
            session=connection.session,
        )
        if message.get("includeAudio") and result.audio is not None:
            await _sendAudio(connection, result.audio, result.assistantText)

    elif kind == "audio.start":
        connection.audio.begin(message)
        connection.wantsAudio = bool(message.get("includeAudio", True))

    elif kind == "audio.end":
        await _handleUtterance(connection, service)

    elif kind == "audio.cancel":
        connection.audio.reset()

    elif kind == "assistant.interrupt":
        await service.interrupt()

    elif kind == "assistant.status":
        await _send(
            connection.socket,
            {
                "type": "assistant.status",
                **service.status().toDict(),
                "sessionId": connection.session.sessionId,
            },
        )

    elif kind == "assistant.ping":
        await _send(connection.socket, {"type": "assistant.pong"})

    else:
        await _send(connection.socket, _error(f"Unknown message type {kind!r}."))


async def _handleUtterance(connection: Connection, service: AssistantService) -> None:
    """Run a turn from audio the client streamed, and send the reply back."""
    try:
        utterance = connection.audio.finish()
    except AudioStreamError as error:
        await _send(connection.socket, _error(str(error)))
        return

    if not utterance.data:
        await _send(connection.socket, _error("No audio arrived."))
        return

    # A remote client speaks and expects to be answered on its own speaker,
    # not in whichever room the server is sitting in.
    startedAt = time.perf_counter()

    if not connection.wantsAudio:
        await service.handleAudio(utterance, speak=True, session=connection.session)
        return

    # Synthesis streams, so an output that forwards to the socket sends each
    # piece as it is generated. The client hears the reply beginning while the
    # rest of it is still being made.
    output = SocketAudioOutput(connection.socket)
    result = await service.handleAudio(
        utterance, session=connection.session, output=output
    )
    serverSeconds = time.perf_counter() - startedAt

    if not output.started:
        await _send(connection.socket, {"type": "audio.replyEnd", "empty": True})
        return

    await _send(
        connection.socket,
        {
            "type": "audio.replyEnd",
            "text": result.assistantText,
            "serverSeconds": round(serverSeconds, 3),
            "firstAudioSeconds": round(result.timing.firstAudioSeconds, 3),
        },
    )


class SocketAudioOutput(AudioOutput):
    """Sends audio to a client instead of playing it.

    Slotting into the existing output abstraction is what makes streaming
    reach remote clients without any new plumbing: the runtime already calls
    ``play`` once per piece of synthesised audio.
    """

    def __init__(self, socket: WebSocket) -> None:
        self._socket = socket
        self.started = False
        self.failed = False

    async def play(self, audio: AudioBuffer) -> None:
        if self.failed or not audio.data:
            return

        if not self.started:
            self.started = True
            if not await _send(self._socket, streamedReplyHeader(audio)):
                self.failed = True
                return

        try:
            await self._socket.send_bytes(audio.data)
        except (WebSocketDisconnect, RuntimeError) as error:
            logger.debug("Could not send audio to a client: %s", error)
            self.failed = True

    def describe(self) -> str:
        return "websocket client"


async def _sendAudio(
    connection: Connection,
    audio: AudioBuffer,
    text: str,
    *,
    serverSeconds: float = 0.0,
) -> None:
    """Send reply audio as a header, binary frames, then a terminator."""
    header = replyHeader(audio, text=text, serverSeconds=serverSeconds)
    if not await _send(connection.socket, header):
        return

    for chunk in chunkAudio(audio):
        try:
            await connection.socket.send_bytes(chunk)
        except (WebSocketDisconnect, RuntimeError) as error:
            logger.debug("Could not send audio to a client: %s", error)
            return

    await _send(connection.socket, {"type": "audio.replyEnd"})


async def _send(socket: WebSocket, payload: dict[str, Any]) -> bool:
    """Send one message, reporting whether the socket is still usable."""
    try:
        await socket.send_json(payload)
    except (WebSocketDisconnect, RuntimeError) as error:
        logger.debug("Could not send to a client: %s", error)
        return False
    return True


def _error(detail: str) -> dict[str, Any]:
    return {"type": "assistant.error", "message": detail}


__all__ = ["Connection", "ConnectionManager", "handleConnection"]
