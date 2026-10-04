"""The assistant as a long-lived service.

Owns the loaded models and the runtime, so that they are loaded once at startup
and reused for every request. §18 of the brief is explicit about this: a service
that reloaded Whisper or XTTS per request would be unusable.

There is one assistant, not one per client. That is the model the brief
describes in §26 --- several microphones and clients, one local service --- and
it matches the hardware, since there is one GPU and one speaker. Turns are
therefore serialised behind a lock rather than run concurrently: two replies
spoken over each other would be worse than one waiting.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.assistant.handlers import createResponseHandler
from app.assistant.runtime import AssistantRuntime, TurnResult, createRuntime
from app.assistant.session import Session, SessionRegistry
from app.audio.audioBuffer import AudioBuffer
from app.audio.input import MemoryAudioInput, resampleBuffer
from app.audio.output import AudioOutput, createAudioOutput
from app.events import AssistantState, EventBus
from app.speech.speechToText import SpeechToTextProvider, Transcript, createSpeechToTextProvider
from app.speech.textToSpeech import TextToSpeechProvider, createTextToSpeechProvider
from app.speech.voices import VoiceLibrary

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


class ServiceBusyError(Exception):
    """Raised when a turn is already running and the caller would not wait."""


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    """What the assistant is doing, for /status."""

    state: AssistantState
    sessionId: str
    turnCount: int
    uptimeSeconds: float
    busy: bool
    providers: dict[str, str]
    sessionCount: int = 0

    def toDict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "sessionId": self.sessionId,
            "turnCount": self.turnCount,
            "uptimeSeconds": round(self.uptimeSeconds, 1),
            "busy": self.busy,
            "sessionCount": self.sessionCount,
            "providers": self.providers,
        }


class AssistantService:
    """A loaded assistant, driven by text or audio.

    The interface the API talks to. Nothing here knows about HTTP, which is
    what allows the same object to back a command-line client.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        bus: EventBus | None = None,
        audioOutput: AudioOutput | None = None,
    ) -> None:
        self._settings = settings
        self.bus = bus or EventBus()
        self._voices = VoiceLibrary(settings.paths.voices)

        self._speechToText = createSpeechToTextProvider(settings)
        self._textToSpeech = createTextToSpeechProvider(settings, self._voices)
        self._output = audioOutput if audioOutput is not None else createAudioOutput(settings)

        self._runtime = createRuntime(
            settings,
            speechToText=self._speechToText,
            textToSpeech=self._textToSpeech,
            handler=createResponseHandler(
                settings, audioOutput=self._output, textToSpeech=self._textToSpeech
            ),
            # A service is driven by its clients, not by a local microphone.
            # Wake-word mode supplies its own capture.
            audioInput=MemoryAudioInput(),
            audioOutput=self._output,
            bus=self.bus,
        )

        self._turnLock = asyncio.Lock()
        self._startedAt = time.monotonic()
        self._sessions = SessionRegistry()

    # --- Lifecycle -------------------------------------------------------

    async def start(self) -> None:
        await self._runtime.start()
        self._startedAt = time.monotonic()
        logger.info("Assistant service ready")

    async def stop(self) -> None:
        await self.bus.drain()
        await self._runtime.stop()
        self._output.close()
        logger.info("Assistant service stopped")

    # --- Properties ------------------------------------------------------

    @property
    def runtime(self) -> AssistantRuntime:
        return self._runtime

    @property
    def session(self) -> Session:
        return self._runtime.session

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def busy(self) -> bool:
        return self._turnLock.locked()

    @property
    def speechToText(self) -> SpeechToTextProvider:
        return self._speechToText

    @property
    def textToSpeech(self) -> TextToSpeechProvider:
        return self._textToSpeech

    def status(self) -> ServiceStatus:
        return ServiceStatus(
            state=self._runtime.state,
            sessionId=self.session.sessionId,
            turnCount=self.session.turnCount,
            uptimeSeconds=time.monotonic() - self._startedAt,
            busy=self.busy,
            sessionCount=len(self._sessions),
            providers={
                "speechToText": self._speechToText.describe(),
                "textToSpeech": self._textToSpeech.describe(),
                "handler": self._runtime.handler.describe(),
            },
        )

    # --- Turns -----------------------------------------------------------

    async def handleText(
        self, text: str, *, speak: bool = True, session: Session | None = None
    ) -> TurnResult:
        """Reply to text. Serialised, so replies never overlap."""
        async with self._turnLock:
            if speak:
                return await self._runtime.handleText(text, session=session)
            with self._runtime.silenced():
                return await self._runtime.handleText(text, session=session)

    async def handleAudio(
        self,
        audio: AudioBuffer,
        *,
        speak: bool = True,
        session: Session | None = None,
        output: AudioOutput | None = None,
    ) -> TurnResult:
        """Transcribe an utterance and reply to it.

        ``output`` redirects this turn's audio. Because synthesis is streamed,
        an output that forwards to a client streams to it.
        """
        prepared = resampleBuffer(audio, self._speechToText.requiredSampleRate)
        async with self._turnLock:
            if output is not None:
                with self._runtime.usingOutput(output):
                    return await self._runtime.handleAudio(prepared, session=session)
            if speak:
                return await self._runtime.handleAudio(prepared, session=session)
            with self._runtime.silenced():
                return await self._runtime.handleAudio(prepared, session=session)

    # --- Sessions --------------------------------------------------------

    def createSession(self) -> Session:
        """A conversation of its own, for one client."""
        return self._sessions.create(
            maxTurns=self._settings.assistant.maxHistoryTurns,
            systemPrompt=self._settings.assistant.systemPrompt or None,
        )

    def removeSession(self, sessionId: str) -> None:
        self._sessions.remove(sessionId)

    @property
    def sessionCount(self) -> int:
        return len(self._sessions)

    async def transcribe(self, audio: AudioBuffer) -> Transcript:
        """Transcribe without replying."""
        prepared = resampleBuffer(audio, self._speechToText.requiredSampleRate)
        return await self._speechToText.transcribe(
            prepared, language=self._settings.speechToText.language or None
        )

    async def synthesise(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
    ) -> AudioBuffer:
        """Generate speech without playing it."""
        return await self._textToSpeech.synthesise(
            text,
            voice=voice or self._settings.textToSpeech.voice,
            language=language or self._settings.textToSpeech.language,
        )

    async def interrupt(self) -> None:
        await self._runtime.interrupt()

    def clearConversation(self) -> None:
        self._runtime.conversation.clear()


__all__ = ["AssistantService", "ServiceBusyError", "ServiceStatus"]
