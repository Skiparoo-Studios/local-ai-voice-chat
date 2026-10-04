"""A thin client: microphone in, speaker out, nothing else.

This is the device the brief describes in §19 --- a machine that contributes a
microphone and a speaker while the expensive inference happens elsewhere. It
loads no speech models and needs no GPU.

It does run the wake word and voice activity detection locally, which is what
keeps the network quiet: audio is only transmitted while someone is actually
speaking, rather than streaming a room's silence continuously.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.api.audioStream import DEFAULT_SAMPLE_RATE
from app.audio.audioBuffer import AudioBuffer
from app.audio.input import AudioInput, createAudioInput
from app.audio.output import AudioOutput, createAudioOutput
from app.audio.vad import SegmenterSettings, SpeechSegmenter, VadProvider, createVadProvider
from app.audio.wakeWord import WakeWordProvider, createWakeWordProvider

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


class ClientError(Exception):
    """Raised when the client cannot reach or talk to the assistant."""


@dataclass(slots=True)
class ClientStatistics:
    """What happened over a session, and how long the server took."""

    utterancesSent: int = 0
    repliesReceived: int = 0
    bytesSent: int = 0
    bytesReceived: int = 0
    roundTripSeconds: list[float] = field(default_factory=list)

    @property
    def averageRoundTrip(self) -> float:
        if not self.roundTripSeconds:
            return 0.0
        return sum(self.roundTripSeconds) / len(self.roundTripSeconds)

    def describe(self) -> str:
        if not self.roundTripSeconds:
            return f"{self.utterancesSent} utterance(s), no replies"
        return (
            f"{self.utterancesSent} utterance(s), "
            f"{self.averageRoundTrip:.2f}s average round trip, "
            f"{self.bytesSent / 1024:.0f} KB up, {self.bytesReceived / 1024:.0f} KB down"
        )


class RemoteClient:
    """Streams captured speech to an assistant and plays what comes back."""

    def __init__(
        self,
        serverUrl: str,
        *,
        token: str | None = None,
        audioInput: AudioInput,
        audioOutput: AudioOutput,
        wakeWord: WakeWordProvider,
        vad: VadProvider,
        segmenter: SegmenterSettings | None = None,
        echoGuardSeconds: float = 0.4,
    ) -> None:
        self._url = _websocketUrl(serverUrl, token)
        self._displayUrl = serverUrl
        self._input = audioInput
        self._output = audioOutput
        self._wakeWord = wakeWord
        self._vad = vad
        self._segmenterSettings = segmenter or SegmenterSettings()
        self._echoGuardSeconds = echoGuardSeconds

        self._socket: Any = None
        self._stopping = asyncio.Event()
        self.statistics = ClientStatistics()

    @classmethod
    def fromSettings(cls, settings: Settings) -> RemoteClient:
        return cls(
            settings.client.serverUrl,
            token=(
                settings.client.token.get_secret_value() if settings.client.token else None
            ),
            audioInput=createAudioInput(settings),
            audioOutput=createAudioOutput(settings),
            wakeWord=createWakeWordProvider(settings),
            vad=createVadProvider(settings),
            segmenter=SegmenterSettings(
                startFrames=settings.vad.startFrames,
                silenceFrames=settings.vad.silenceFrames,
                prerollFrames=settings.vad.prerollFrames,
                maximumSeconds=settings.vad.maximumSeconds,
                minimumSeconds=settings.vad.minimumSeconds,
            ),
            echoGuardSeconds=settings.audio.echoGuardSeconds,
        )

    @property
    def frameMilliseconds(self) -> int:
        return min(self._vad.frameMilliseconds, self._wakeWord.frameMilliseconds)

    async def start(self) -> None:
        await self._wakeWord.load()
        await self._vad.load()

    async def stop(self) -> None:
        self._stopping.set()

    async def run(self) -> ClientStatistics:
        """Connect, then listen until stopped."""
        websockets = _importWebsockets()
        self.statistics = ClientStatistics()

        try:
            async with websockets.connect(self._url, max_size=None) as socket:
                self._socket = socket
                hello = json.loads(await socket.recv())
                logger.info(
                    "Connected to %s as session %s",
                    self._displayUrl,
                    hello.get("sessionId", "?"),
                )
                await self._listen(socket)
        except Exception as error:
            raise _describeFailure(error, self._displayUrl) from error
        finally:
            self._socket = None

        return self.statistics

    # --- Listening -------------------------------------------------------

    async def _listen(self, socket: Any) -> None:
        segmenter = SpeechSegmenter(self._segmenterSettings)
        awake = self._wakeWord.alwaysAwake
        pending: list[AudioBuffer] = []

        async for frame in self._input.frames(self.frameMilliseconds):
            if self._stopping.is_set():
                return

            if not awake:
                pending.append(frame)
                block = _joinFrames(pending, self._wakeWord.frameMilliseconds)
                if block is None:
                    continue
                pending.clear()

                if self._wakeWord.detect(block) is None:
                    continue

                logger.info("Wake word heard")
                awake = True
                segmenter = SpeechSegmenter(self._segmenterSettings)
                self._vad.reset()
                continue

            result = segmenter.feed(frame, self._vad.isSpeech(frame))
            if not result.isComplete:
                continue

            if result.utterance is not None:
                await self._sendUtterance(socket, result.utterance)
                # The reply is played inside that call, so this loop was not
                # reading frames for the whole of it and the microphone queue
                # now holds the assistant's own voice. Feeding that to the
                # segmenter is how the client ends up answering itself.
                await self._settleAfterSpeaking()

            segmenter = SpeechSegmenter(self._segmenterSettings)
            self._vad.reset()
            awake = self._wakeWord.alwaysAwake
            if not awake:
                self._wakeWord.reset()

    async def _settleAfterSpeaking(self) -> None:
        """Discard what was captured while the reply played, then pause.

        The pause covers what draining cannot: PortAudio's write() returns with
        roughly 180 ms still in the device buffer, and the room reverberates it
        for a moment after that.
        """
        dropped = self._input.drain()
        if dropped:
            logger.debug("Discarded %d frame(s) captured while speaking", dropped)

        if self._echoGuardSeconds > 0:
            await asyncio.sleep(self._echoGuardSeconds)
            self._input.drain()

    async def _sendUtterance(self, socket: Any, utterance: AudioBuffer) -> None:
        """Stream one utterance and play whatever comes back."""
        logger.info("Sending %.2fs of speech", utterance.durationSeconds)
        started = time.perf_counter()

        await socket.send(
            json.dumps(
                {
                    "type": "audio.start",
                    "sampleRate": utterance.sampleRate,
                    "channels": utterance.channels,
                    "includeAudio": True,
                }
            )
        )
        await socket.send(utterance.data)
        await socket.send(json.dumps({"type": "audio.end"}))

        self.statistics.utterancesSent += 1
        self.statistics.bytesSent += len(utterance.data)

        await self._awaitReply(socket, started)

    async def _awaitReply(self, socket: Any, startedAt: float) -> None:
        """Collect the reply, then play it.

        Buffering the whole reply before playing costs a little latency and
        avoids stuttering on a busy network. Playing each chunk as it arrives
        is the obvious improvement once this is measured over something worse
        than a LAN.
        """
        header: dict[str, Any] | None = None
        chunks: list[bytes] = []

        while True:
            try:
                message = await asyncio.wait_for(socket.recv(), timeout=120)
            except TimeoutError:
                logger.warning("The assistant did not reply within two minutes")
                return

            if isinstance(message, bytes):
                chunks.append(message)
                continue

            payload = json.loads(message)
            kind = payload.get("type")

            if kind == "audio.reply":
                header = payload
                logger.info("Reply: %s", payload.get("text", ""))
            elif kind == "assistant.transcription":
                logger.info("Heard: %s", payload.get("text", ""))
            elif kind == "assistant.error":
                logger.error("Assistant error: %s", payload.get("message", ""))
                return
            elif kind == "audio.replyEnd":
                break

        elapsed = time.perf_counter() - startedAt
        self.statistics.roundTripSeconds.append(elapsed)

        if header is None or not chunks:
            logger.info("No audio came back (%.2fs)", elapsed)
            return

        audio = AudioBuffer(
            data=b"".join(chunks),
            sampleRate=int(header.get("sampleRate") or DEFAULT_SAMPLE_RATE),
            channels=int(header.get("channels") or 1),
        )
        self.statistics.repliesReceived += 1
        self.statistics.bytesReceived += len(audio.data)

        logger.info(
            "Reply took %.2fs for %.2fs of audio", elapsed, audio.durationSeconds
        )
        await self._output.play(audio)

    async def sendText(self, text: str) -> dict[str, Any]:
        """Send a typed message, for checking a connection without speaking."""
        websockets = _importWebsockets()

        try:
            async with websockets.connect(self._url, max_size=None) as socket:
                await socket.recv()
                await socket.send(
                    json.dumps({"type": "assistant.message", "text": text})
                )

                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    message = await asyncio.wait_for(socket.recv(), timeout=120)
                    if isinstance(message, bytes):
                        continue
                    payload = json.loads(message)
                    if payload.get("type") in {"assistant.response", "assistant.error"}:
                        return payload
        except Exception as error:
            raise _describeFailure(error, self._displayUrl) from error

        return {}

    def describe(self) -> str:
        return (
            f"server:    {self._displayUrl}\n"
            f"  wake word: {self._wakeWord.describe()}\n"
            f"  vad:       {self._vad.describe()}\n"
            f"  input:     {self._input.describe()}"
        )

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._input.close()
        with contextlib.suppress(Exception):
            self._output.close()


# --- Module helpers ------------------------------------------------------


def _joinFrames(frames: list[AudioBuffer], blockMilliseconds: int) -> AudioBuffer | None:
    """Combine small frames into the larger block the wake word needs."""
    if not frames:
        return None

    total = sum(frame.durationSeconds for frame in frames) * 1000
    if total < blockMilliseconds:
        return None

    return AudioBuffer(
        data=b"".join(frame.data for frame in frames),
        sampleRate=frames[0].sampleRate,
        channels=frames[0].channels,
    )


def _websocketUrl(serverUrl: str, token: str | None) -> str:
    """Turn a server address into the WebSocket endpoint."""
    url = serverUrl.strip().rstrip("/")

    if url.startswith("http://"):
        url = "ws://" + url[len("http://") :]
    elif url.startswith("https://"):
        url = "wss://" + url[len("https://") :]
    elif not url.startswith(("ws://", "wss://")):
        url = "ws://" + url

    if not url.endswith("/ws/assistant"):
        url = f"{url}/ws/assistant"

    # Browsers cannot set handshake headers, so the server accepts the token
    # in the query string; a client using it too keeps one code path.
    return f"{url}?token={token}" if token else url


def _importWebsockets() -> Any:
    try:
        import websockets
    except ImportError as error:
        raise ClientError(
            "The remote client requires the 'websockets' package.\n"
            'Install with: pip install -e ".[api]"'
        ) from error
    return websockets


def _describeFailure(error: Exception, url: str) -> Exception:
    """Translate connection failures into messages that say what to do."""
    if isinstance(error, ClientError):
        return error

    text = str(error).lower()

    if "401" in text or "403" in text or "policy" in text or "1008" in text:
        return ClientError(
            f"The assistant at {url} rejected the credentials.\n"
            "Set client.token to the same value as the server's api.authToken, "
            "through VOICE_CLIENT__TOKEN."
        )

    if "refused" in text or "connect" in text or "timed out" in text or "getaddrinfo" in text:
        return ClientError(
            f"Could not reach the assistant at {url}.\n"
            "Check it is running with --mode serve, that api.host is not "
            "127.0.0.1 if you are connecting from another machine, and that "
            "nothing is blocking the port."
        )

    return ClientError(f"The connection to {url} failed: {error}")


__all__ = ["ClientError", "ClientStatistics", "RemoteClient"]
