"""The assistant runtime.

Orchestrates one turn: capture, transcribe, decide, synthesise, play. It owns
the state machine clients display and publishes the events described in §8 of
the brief, so that nothing downstream has to poll.

The runtime knows about interfaces only. Which speech models, which handler and
which audio devices are in use is decided by configuration before it is built.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.assistant.conversation import Conversation
from app.assistant.handlers import Response, ResponseHandler
from app.assistant.sentences import SentenceStreamer
from app.assistant.session import Session
from app.audio.audioBuffer import AudioBuffer
from app.audio.input import AudioInput
from app.audio.output import AudioOutput, MemoryAudioOutput
from app.events import (
    AssistantError,
    AssistantState,
    AssistantStateChanged,
    EventBus,
    ResponseGenerated,
    SpeechEnded,
    SpeechGenerationCompleted,
    SpeechGenerationStarted,
    SpeechStarted,
    TranscriptionCompleted,
    TranscriptionStarted,
)
from app.speech.speechToText import SpeechToTextProvider, Transcript
from app.speech.textToSpeech import TextToSpeechProvider

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

WARM_UP_TEXT = "Ready."
ACKNOWLEDGEMENT_TEXT = "Yes?"


@dataclass(slots=True)
class TurnTiming:
    """Where the time went in one turn.

    Recorded per stage rather than as a single total, because the whole point
    of measuring is to know which stage to attack.
    """

    captureSeconds: float = 0.0
    transcriptionSeconds: float = 0.0
    responseSeconds: float = 0.0
    # Wall time of the speaking phase. Once audio is streamed into an open
    # device, generation is paced by playback, so this covers both rather than
    # synthesis alone. firstAudioSeconds is the figure a listener experiences.
    synthesisSeconds: float = 0.0
    playbackSeconds: float = 0.0
    audioSeconds: float = 0.0
    # Time from the start of the reply phase until the first audio is ready.
    # Only meaningful when streaming, where generation and synthesis overlap
    # and the sum of their durations is no longer what the user waits for.
    firstAudioSeconds: float = 0.0
    streamed: bool = False

    @property
    def thinkingSeconds(self) -> float:
        """What the user actually waits: from speaking to hearing a reply.

        Once synthesis emits audio before it has finished, the time to the
        first sound and the time to generate everything are different numbers,
        and only the first is what anyone experiences.
        """
        if self.streamed:
            return self.transcriptionSeconds + self.firstAudioSeconds
        if self.firstAudioSeconds:
            return (
                self.transcriptionSeconds + self.responseSeconds + self.firstAudioSeconds
            )
        return self.transcriptionSeconds + self.responseSeconds + self.synthesisSeconds

    @property
    def totalSeconds(self) -> float:
        return self.thinkingSeconds + self.playbackSeconds

    def describe(self) -> str:
        if self.streamed:
            return (
                f"stt {self.transcriptionSeconds:.2f}s  "
                f"first sentence {self.firstAudioSeconds:.2f}s  "
                f"(llm {self.responseSeconds:.2f}s + tts {self.synthesisSeconds:.2f}s total)  "
                f"= {self.thinkingSeconds:.2f}s to first audio"
            )
        if self.firstAudioSeconds:
            # The second figure covers speaking as well as synthesising: once
            # audio is written into an open stream, generation is paced by
            # playback rather than racing ahead of it.
            return (
                f"stt {self.transcriptionSeconds:.2f}s  "
                f"think {self.responseSeconds:.2f}s  "
                f"tts {self.firstAudioSeconds:.2f}s to first sound "
                f"({self.synthesisSeconds:.2f}s spent speaking)  "
                f"= {self.thinkingSeconds:.2f}s to first audio"
            )
        return (
            f"stt {self.transcriptionSeconds:.2f}s  "
            f"think {self.responseSeconds:.2f}s  "
            f"tts {self.synthesisSeconds:.2f}s  "
            f"= {self.thinkingSeconds:.2f}s to first audio"
        )


@dataclass(slots=True)
class TurnResult:
    """Everything that happened in one turn."""

    transcript: Transcript | None = None
    response: Response | None = None
    audio: AudioBuffer | None = None
    timing: TurnTiming = field(default_factory=TurnTiming)
    handled: bool = False

    @property
    def userText(self) -> str:
        return self.transcript.text if self.transcript else ""

    @property
    def assistantText(self) -> str:
        return self.response.text if self.response else ""

    @property
    def shouldStop(self) -> bool:
        return bool(self.response and self.response.shouldStop)


class AssistantRuntime:
    """Runs conversation turns."""

    def __init__(
        self,
        *,
        speechToText: SpeechToTextProvider,
        textToSpeech: TextToSpeechProvider,
        handler: ResponseHandler,
        audioInput: AudioInput,
        audioOutput: AudioOutput,
        bus: EventBus,
        session: Session,
        language: str | None = "en",
        voice: str | None = None,
        maximumRecordingSeconds: float = 30.0,
        warmUpOnStart: bool = True,
        streaming: bool = True,
    ) -> None:
        self._stt = speechToText
        self._tts = textToSpeech
        self._handler = handler
        self._input = audioInput
        self._output = audioOutput
        self._bus = bus
        self._session = session
        self._language = language
        self._voice = voice
        self._maximumRecordingSeconds = maximumRecordingSeconds
        self._warmUpOnStart = warmUpOnStart
        self._streaming = streaming
        self._started = False

        # State belongs to the assistant, not to a session. Conversations can
        # be per-client, but there is one GPU and one speaker, so there is only
        # one thing the assistant can be doing at a time.
        self._state = AssistantState.idle

    # --- Properties ------------------------------------------------------

    @property
    def session(self) -> Session:
        """The default session, used when a caller names none."""
        return self._session

    @property
    def handler(self) -> ResponseHandler:
        return self._handler

    @property
    def conversation(self) -> Conversation:
        return self._session.conversation

    @property
    def state(self) -> AssistantState:
        return self._state

    @property
    def isStarted(self) -> bool:
        return self._started

    # --- Lifecycle -------------------------------------------------------

    async def start(self) -> None:
        """Load both models, then optionally warm them."""
        if self._started:
            return

        started = time.perf_counter()
        # Loading is IO- and GPU-bound in different ways, so overlap them.
        await asyncio.gather(self._stt.load(), self._tts.load(), self._handler.load())
        logger.info("Models loaded in %.1fs", time.perf_counter() - started)

        if self._warmUpOnStart:
            await self.warmUp()

        self._started = True
        await self._setState(AssistantState.idle)

    async def warmUp(self) -> None:
        """Pay the first-run cost before the user is waiting on it.

        The first synthesis after loading costs roughly three times the warm
        rate on CUDA, and the first transcription about twice. Doing both here
        means the first real turn is not the slowest one.
        """
        started = time.perf_counter()
        try:
            await asyncio.gather(
                self._tts.synthesise(WARM_UP_TEXT, voice=self._voice, language=self._language),
                self._stt.transcribe(
                    AudioBuffer.silence(0.5, self._stt.requiredSampleRate),
                    language=self._language,
                ),
                # A language model served by another process loads on first
                # request, not when that process starts, so the only way to
                # pay for it early is to make one.
                self._handler.warmUp(),
            )
        except Exception as error:  # noqa: BLE001 - warm-up must never block startup
            logger.warning("Warm-up failed, continuing anyway: %s", error)
            return
        logger.info("Warmed up in %.1fs", time.perf_counter() - started)

    async def stop(self) -> None:
        if not self._started:
            return
        await self._setState(AssistantState.idle)
        await asyncio.gather(self._stt.unload(), self._tts.unload(), self._handler.unload())
        self._started = False

    # --- Turns -----------------------------------------------------------

    async def runTurn(
        self,
        *,
        stopWhen: asyncio.Event | None = None,
        session: Session | None = None,
    ) -> TurnResult:
        """Capture from the microphone, then run the turn."""
        timing = TurnTiming()
        active = session or self._session

        await self._setState(AssistantState.listening, active)
        await self._bus.publish(SpeechStarted(sessionId=active.sessionId))

        started = time.perf_counter()
        audio = await self._input.record(
            stopWhen=stopWhen, maximumSeconds=self._maximumRecordingSeconds
        )
        timing.captureSeconds = time.perf_counter() - started

        await self._bus.publish(
            SpeechEnded(sessionId=active.sessionId, durationSeconds=audio.durationSeconds)
        )

        return await self.handleAudio(audio, timing=timing, session=active)

    async def handleAudio(
        self,
        audio: AudioBuffer,
        *,
        timing: TurnTiming | None = None,
        session: Session | None = None,
    ) -> TurnResult:
        """Transcribe captured audio, then run the turn."""
        timing = timing or TurnTiming()
        result = TurnResult(timing=timing)
        active = session or self._session

        if not audio.data:
            await self._setState(AssistantState.idle, active)
            return result

        await self._setState(AssistantState.thinking, active)
        await self._bus.publish(TranscriptionStarted(sessionId=active.sessionId))

        started = time.perf_counter()
        transcript = await self._stt.transcribe(audio, language=self._language)
        timing.transcriptionSeconds = time.perf_counter() - started
        result.transcript = transcript

        await self._bus.publish(
            TranscriptionCompleted(
                sessionId=active.sessionId,
                text=transcript.text,
                language=transcript.language,
                durationSeconds=timing.transcriptionSeconds,
            )
        )

        if transcript.isEmpty:
            logger.debug("Nothing was said; not producing a reply")
            await self._setState(AssistantState.idle, active)
            return result

        return await self.handleText(
            transcript.text, timing=timing, result=result, session=active
        )

    async def handleText(
        self,
        text: str,
        *,
        timing: TurnTiming | None = None,
        result: TurnResult | None = None,
        session: Session | None = None,
    ) -> TurnResult:
        """Reply to text and speak it, skipping capture and transcription.

        ``session`` supplies the conversation the turn belongs to. Remote
        clients each have their own, so two people talking to the assistant
        from different rooms do not share context.
        """
        timing = timing or TurnTiming()
        result = result or TurnResult(timing=timing)
        active = session or self._session

        if not text.strip():
            await self._setState(AssistantState.idle, active)
            return result

        if self.state is not AssistantState.thinking:
            await self._setState(AssistantState.thinking, active)

        streaming = self._streaming and self._handler.supportsStreaming

        conversation = active.conversation

        try:
            if streaming:
                # Streaming speaks as it generates, so the audio has already
                # been played by the time this returns.
                response = await self._respondStreaming(text, timing, result, active)
            else:
                response = await self._generateComplete(text, timing, conversation)
        except Exception as error:
            await self._publishError(
                f"Handler failed: {error}", component="handler", session=active
            )
            await self._setState(AssistantState.idle, active)
            raise

        # The handler sees history without the current utterance, so it is
        # recorded afterwards and a turn is never counted twice.
        conversation.addUser(text)
        result.response = response

        if response.isEmpty:
            await self._setState(AssistantState.idle, active)
            return result

        conversation.addAssistant(response.text)
        active.turnCount += 1
        result.handled = True

        # Announced before the reply is spoken, so a client can show the text
        # while the audio plays rather than after it has finished. When
        # streaming, the full text is only known once generation has ended, so
        # this is still the earliest moment it can be sent.
        await self._bus.publish(
            ResponseGenerated(
                sessionId=active.sessionId,
                text=response.text,
                usedLlm=response.usedLlm,
                durationSeconds=timing.responseSeconds,
            )
        )

        if not streaming:
            await self._speak(response.text, timing, result, active)

        await self._setState(AssistantState.idle, active)
        return result

    async def speakUnprompted(
        self, response: Response, *, session: Session | None = None
    ) -> TurnResult:
        """Say something nobody asked for, and record it as a turn.

        Everything ``handleText`` does except decide what to say and record an
        utterance, because there was not one. Used when a handler wants to fill
        a silence rather than answer something.
        """
        timing = TurnTiming()
        result = TurnResult(timing=timing)
        active = session or self._session

        if response.isEmpty:
            return result

        result.response = response
        active.conversation.addAssistant(response.text)
        active.turnCount += 1
        result.handled = True

        await self._bus.publish(
            ResponseGenerated(
                sessionId=active.sessionId,
                text=response.text,
                usedLlm=response.usedLlm,
                durationSeconds=0.0,
            )
        )

        await self._speak(response.text, timing, result, active)
        await self._setState(AssistantState.idle, active)
        return result

    async def _generateComplete(
        self, text: str, timing: TurnTiming, conversation: Conversation
    ) -> Response:
        """Generate the whole reply, without speaking it."""
        started = time.perf_counter()
        response = await self._handler.respond(text, conversation)
        timing.responseSeconds = time.perf_counter() - started
        return response

    async def _respondStreaming(
        self, text: str, timing: TurnTiming, result: TurnResult, session: Session
    ) -> Response:
        """Speak each sentence as it is generated.

        Synthesis of one sentence overlaps generation of the next, and playback
        overlaps synthesis, so what the user waits for is the first sentence
        rather than the whole reply.
        """
        timing.streamed = True
        started = time.perf_counter()

        queue: asyncio.Queue[AudioBuffer | None] = asyncio.Queue()
        player = asyncio.create_task(self._playQueued(queue, timing))

        streamer = SentenceStreamer()
        spoken: list[str] = []
        pieces: list[AudioBuffer] = []

        async def speakPiece(piece: str) -> None:
            synthesisStarted = time.perf_counter()
            spoken.append(piece)

            # The provider may emit audio while it is still generating, so a
            # single sentence starts playing long before it is finished. Where
            # it cannot, the default implementation yields one whole piece and
            # this behaves exactly as before.
            async for audio in self._tts.synthesiseStream(
                piece, voice=self._voice, language=self._language
            ):
                if not timing.firstAudioSeconds:
                    timing.firstAudioSeconds = time.perf_counter() - started
                    await self._setState(AssistantState.speaking, session)

                pieces.append(audio)
                timing.audioSeconds += audio.durationSeconds
                await queue.put(audio)

            timing.synthesisSeconds += time.perf_counter() - synthesisStarted

        await self._output.beginUtterance()
        try:
            async for fragment in self._handler.respondStream(text, session.conversation):
                for piece in streamer.feed(fragment):
                    await speakPiece(piece)

            remainder = streamer.flush()
            if remainder:
                await speakPiece(remainder)
        finally:
            # The player must be released even if generation fails part way,
            # or the task leaks and the turn never completes.
            await queue.put(None)
            await player
            await self._output.endUtterance()

        timing.responseSeconds = time.perf_counter() - started
        result.audio = _joinAudio(pieces)

        return Response(
            text=" ".join(spoken).strip(),
            usedLlm=self._handler.usesLlm,
            handledBy=self._handler.name,
        )

    async def _playQueued(
        self, queue: asyncio.Queue[AudioBuffer | None], timing: TurnTiming
    ) -> None:
        """Play buffers in order until told there are no more."""
        while True:
            audio = await queue.get()
            if audio is None:
                return
            started = time.perf_counter()
            try:
                await self._output.play(audio)
            except Exception as error:  # noqa: BLE001 - one failure must not strand the turn
                logger.error("Playback failed: %s", error)
                return
            finally:
                timing.playbackSeconds += time.perf_counter() - started

    async def _speak(
        self, text: str, timing: TurnTiming, result: TurnResult, session: Session
    ) -> None:
        """Synthesise a complete reply and play it.

        Even here the audio is streamed where the provider can, because most
        of what the assistant says is a single sentence and waiting for the
        whole of one costs several times what waiting for its beginning does.
        """
        await self._setState(AssistantState.speaking, session)
        await self._bus.publish(
            SpeechGenerationStarted(
                sessionId=session.sessionId, text=text, voice=self._voice or ""
            )
        )

        started = time.perf_counter()
        queue: asyncio.Queue[AudioBuffer | None] = asyncio.Queue()
        player = asyncio.create_task(self._playQueued(queue, timing))
        pieces: list[AudioBuffer] = []

        await self._output.beginUtterance()
        try:
            async for audio in self._tts.synthesiseStream(
                text, voice=self._voice, language=self._language
            ):
                if not timing.firstAudioSeconds:
                    timing.firstAudioSeconds = time.perf_counter() - started
                pieces.append(audio)
                timing.audioSeconds += audio.durationSeconds
                await queue.put(audio)
        finally:
            await queue.put(None)
            await player
            await self._output.endUtterance()

        timing.synthesisSeconds = time.perf_counter() - started
        result.audio = _joinAudio(pieces)

        await self._bus.publish(
            SpeechGenerationCompleted(
                sessionId=session.sessionId,
                voice=self._voice or "",
                audioSeconds=timing.audioSeconds,
                durationSeconds=timing.synthesisSeconds,
            )
        )

    @contextlib.contextmanager
    def usingOutput(self, output: AudioOutput) -> Iterator[None]:
        """Send this turn's audio somewhere other than the local speaker.

        Because synthesis is streamed, ``play`` is called for each piece as it
        is produced. An output that forwards to a remote client therefore
        streams to it, with no further plumbing.
        """
        original = self._output
        self._output = output
        try:
            yield
        finally:
            self._output = original

    @contextlib.contextmanager
    def silenced(self) -> Iterator[None]:
        """Run turns without playing them on the local speaker.

        A remote client asking for a reply usually wants the audio returned to
        it, not spoken aloud in an empty room.
        """
        with self.usingOutput(MemoryAudioOutput()):
            yield

    async def interrupt(self) -> None:
        """Stop speaking immediately."""
        await self._output.stop()
        await self._setState(AssistantState.idle)

    async def acknowledge(self, text: str = ACKNOWLEDGEMENT_TEXT) -> None:
        """Say a short word to confirm the assistant is listening.

        Optional, because on a fast machine the reply arrives soon enough that
        an acknowledgement only adds noise, while on a slow one it is the
        difference between waiting and wondering whether it heard.
        """
        try:
            audio = await self._tts.synthesise(
                text, voice=self._voice, language=self._language
            )
            await self._output.play(audio)
        except Exception as error:  # noqa: BLE001 - never block the real turn
            logger.debug("Could not play the acknowledgement: %s", error)

    # --- State -----------------------------------------------------------

    async def _setState(self, state: AssistantState, session: Session | None = None) -> None:
        """Change the assistant's state, and say which session caused it.

        The state is the assistant's, not the session's, because there is one
        speaker. The session identifier travels with the event so a client can
        tell whether the assistant is busy on its behalf or someone else's.
        """
        if state is self._state:
            return

        previous = self._state
        self._state = state

        active = session or self._session
        active.state = state

        await self._bus.publish(
            AssistantStateChanged(
                sessionId=active.sessionId, state=state, previousState=previous
            )
        )

    async def _publishError(
        self, message: str, *, component: str, session: Session | None = None
    ) -> None:
        logger.error("%s", message)
        await self._bus.publish(
            AssistantError(
                sessionId=(session or self._session).sessionId,
                message=message,
                component=component,
            )
        )

    def describe(self) -> str:
        return (
            f"stt: {self._stt.describe()}\n"
            f"  tts: {self._tts.describe()}\n"
            f"  handler: {self._handler.describe()}"
        )


def _joinAudio(pieces: list[AudioBuffer]) -> AudioBuffer | None:
    """Concatenate streamed pieces, so a turn still has one audio result."""
    if not pieces:
        return None
    joined = pieces[0]
    for piece in pieces[1:]:
        joined = joined.concatenate(piece)
    return joined


def createRuntime(
    settings: Settings,
    *,
    speechToText: SpeechToTextProvider,
    textToSpeech: TextToSpeechProvider,
    handler: ResponseHandler,
    audioInput: AudioInput,
    audioOutput: AudioOutput,
    bus: EventBus,
    session: Session | None = None,
) -> AssistantRuntime:
    """Assemble a runtime from configuration and already-built components."""
    return AssistantRuntime(
        speechToText=speechToText,
        textToSpeech=textToSpeech,
        handler=handler,
        audioInput=audioInput,
        audioOutput=audioOutput,
        bus=bus,
        session=session
        or Session.create(
            maxTurns=settings.assistant.maxHistoryTurns,
            systemPrompt=settings.assistant.systemPrompt or None,
        ),
        language=settings.speechToText.language or None,
        voice=settings.textToSpeech.voice,
        maximumRecordingSeconds=settings.speechToText.maximumRecordingSeconds,
        warmUpOnStart=settings.assistant.warmUpOnStart,
        streaming=settings.assistant.streaming,
    )
