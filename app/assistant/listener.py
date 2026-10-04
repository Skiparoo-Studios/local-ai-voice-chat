"""Hands-free listening.

One loop consumes microphone frames continuously and decides what to do with
each, according to where the assistant is:

    waiting    watching for the wake word, discarding everything else
    capturing  accumulating an utterance until the speaker stops
    busy       a turn is running; frames are watched only for barge-in

The turn runs as a separate task rather than inside the loop. That is what
makes barge-in possible: frames keep being read while the assistant is
speaking, so it can be interrupted.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum

from app.assistant.handlers import Response
from app.assistant.runtime import AssistantRuntime, TurnResult
from app.audio.audioBuffer import AudioBuffer
from app.audio.input import AudioInput
from app.audio.vad import SegmenterSettings, SegmentState, SpeechSegmenter, VadProvider
from app.audio.wakeWord import WakeWordProvider
from app.events import (
    AssistantError,
    AssistantState,
    AudioDetected,
    EventBus,
    WakeWordDetected,
)

logger = logging.getLogger(__name__)


class ListenerState(StrEnum):
    """What the listener is doing with incoming frames."""

    waiting = "waiting"
    capturing = "capturing"
    busy = "busy"


@dataclass(slots=True)
class ListenerStatistics:
    """Counts worth reporting after a session."""

    framesSeen: int = 0
    wakeWordsHeard: int = 0
    utterancesCaptured: int = 0
    utterancesDiscarded: int = 0
    bargeIns: int = 0
    turnsRun: int = 0
    # Frames thrown away so the assistant does not hear its own reply.
    framesGuarded: int = 0
    # Things said into a silence rather than in answer to anything.
    promptsSpoken: int = 0
    capturedSeconds: float = 0.0
    startedAt: float = field(default_factory=time.monotonic)

    @property
    def elapsedSeconds(self) -> float:
        return time.monotonic() - self.startedAt

    def describe(self) -> str:
        return (
            f"{self.wakeWordsHeard} wake word(s), "
            f"{self.turnsRun} turn(s), "
            f"{self.utterancesDiscarded} discarded, "
            f"{self.framesGuarded} frame(s) guarded, "
            f"{self.bargeIns} interruption(s) over {self.elapsedSeconds:.0f}s"
        )


class ContinuousListener:
    """Runs the assistant hands-free."""

    def __init__(
        self,
        runtime: AssistantRuntime,
        audioInput: AudioInput,
        wakeWord: WakeWordProvider,
        vad: VadProvider,
        bus: EventBus,
        *,
        segmenter: SegmenterSettings | None = None,
        allowBargeIn: bool = True,
        acknowledgeWake: bool = False,
        echoGuardSeconds: float = 0.4,
    ) -> None:
        self._runtime = runtime
        self._input = audioInput
        self._wakeWord = wakeWord
        self._vad = vad
        self._bus = bus
        self._segmenterSettings = segmenter or SegmenterSettings()
        self._allowBargeIn = allowBargeIn
        self._acknowledgeWake = acknowledgeWake
        self._echoGuardSeconds = echoGuardSeconds

        self._state = ListenerState.waiting
        self._segmenter = SpeechSegmenter(self._segmenterSettings)
        self._turn: asyncio.Task[TurnResult] | None = None
        self._stopping = asyncio.Event()
        # Frames arriving before this are discarded. See _guardAgainstEcho.
        self._ignoreUntil = 0.0
        # When anything last happened, for handlers that fill a silence.
        self._lastActivity = time.monotonic()
        self.statistics = ListenerStatistics()

    @property
    def state(self) -> ListenerState:
        return self._state

    @property
    def frameMilliseconds(self) -> int:
        """The frame size both detectors can work with.

        The wake-word model needs larger frames than the voice activity model,
        so the smaller size drives the stream and wake-word detection consumes
        several frames at a time.
        """
        return min(self._vad.frameMilliseconds, self._wakeWord.frameMilliseconds)

    async def start(self) -> None:
        await self._wakeWord.load()
        await self._vad.load()

    async def stop(self) -> None:
        self._stopping.set()
        if self._turn is not None and not self._turn.done():
            self._turn.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._turn

    async def run(self) -> ListenerStatistics:
        """Listen until stopped."""
        self.statistics = ListenerStatistics()
        self._noteActivity()

        if self._wakeWord.alwaysAwake:
            logger.info("No wake word configured; listening continuously")
            self._beginCapture()

        wakeBuffer = _FrameAggregator(
            self._wakeWord.frameMilliseconds, self.frameMilliseconds
        )

        async for frame in self._input.frames(self.frameMilliseconds):
            if self._stopping.is_set():
                break

            self.statistics.framesSeen += 1
            self._reapFinishedTurn()

            if self._withinEchoGuard():
                self.statistics.framesGuarded += 1
                continue

            if self._state is ListenerState.waiting:
                await self._whileWaiting(frame, wakeBuffer)
            elif self._state is ListenerState.capturing:
                await self._whileCapturing(frame)
                await self._promptIfSilent()
            else:
                await self._whileBusy(frame, wakeBuffer)

        await self._awaitTurn()
        return self.statistics

    # --- Filling a silence ------------------------------------------------

    async def _promptIfSilent(self) -> None:
        """Let the handler speak when nobody has said anything for a while.

        Only while capturing: during a turn the assistant is already talking,
        and while waiting for a wake word it has not been addressed at all.
        Handlers opt in --- almost all return None and this never fires.
        """
        after = self._handlerPromptSeconds()
        if after is None:
            return

        # Someone is part way through speaking. Interrupting them to ask
        # whether they are still there would be absurd.
        if self._segmenter.state is not SegmentState.waiting:
            return

        if time.monotonic() - self._lastActivity < after:
            return

        # Stamped before the prompt runs, so a handler that declines to say
        # anything does not send this straight round again on the next frame.
        self._lastActivity = time.monotonic()

        response = await self._runtime.handler.promptWhenSilent()
        if response is None or response.isEmpty:
            logger.debug("Handler had nothing to say to the silence")
            return

        self.statistics.promptsSpoken += 1
        logger.info("Filling a silence: %s", response.text)
        self._state = ListenerState.busy
        self._turn = asyncio.create_task(self._speakPrompt(response))

    async def _speakPrompt(self, response: Response) -> TurnResult:
        try:
            return await self._runtime.speakUnprompted(response)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.exception("Prompt failed")
            await self._bus.publish(
                AssistantError(message=str(error), component="listener")
            )
            return TurnResult()

    def _handlerPromptSeconds(self) -> float | None:
        """How long this handler waits before speaking, if it does at all."""
        try:
            return self._runtime.handler.promptAfterSeconds
        except AttributeError:  # pragma: no cover - a handler predating the hook
            return None

    def _noteActivity(self) -> None:
        """Restart the silence clock."""
        self._lastActivity = time.monotonic()

    # --- Not listening to itself -----------------------------------------

    def _guardAgainstEcho(self) -> None:
        """Stop the assistant hearing the end of its own reply.

        Two things leak into the microphone when a turn finishes. Anything
        captured while the assistant was speaking is still sitting in the
        capture queue, so it is thrown away. And PortAudio's write() returns
        with roughly 180 ms still in the device buffer, which a room then
        reverberates, so the microphone is ignored for a moment longer.

        Not applied after a barge-in: there the person is already mid-sentence
        and playback was aborted rather than allowed to drain, so a guard would
        swallow the beginning of what they came to say.
        """
        dropped = self._input.drain()
        if dropped:
            logger.debug("Discarded %d frame(s) captured while speaking", dropped)
            self.statistics.framesGuarded += dropped

        if self._echoGuardSeconds > 0:
            self._ignoreUntil = time.monotonic() + self._echoGuardSeconds

    def _withinEchoGuard(self) -> bool:
        """Whether this frame arrived too soon after the assistant spoke."""
        if not self._ignoreUntil:
            return False
        if time.monotonic() < self._ignoreUntil:
            return True

        self._ignoreUntil = 0.0
        # The guarded frames never reached either model, and a segmenter part
        # way through an utterance would otherwise resume as though nothing had
        # been missed.
        self._vad.reset()
        self._segmenter.reset()
        return False

    # --- States ----------------------------------------------------------

    async def _whileWaiting(self, frame: AudioBuffer, wakeBuffer: _FrameAggregator) -> None:
        for block in wakeBuffer.feed(frame):
            detection = self._wakeWord.detect(block)
            if detection is None:
                continue

            self.statistics.wakeWordsHeard += 1
            await self._bus.publish(
                WakeWordDetected(word=detection.word, confidence=detection.confidence)
            )
            wakeBuffer.reset()
            self._beginCapture()

            if self._acknowledgeWake:
                await self._runtime.acknowledge()
            return

    async def _whileCapturing(self, frame: AudioBuffer) -> None:
        result = self._segmenter.feed(frame, self._vad.isSpeech(frame))

        if not result.isComplete:
            return

        if result.utterance is None:
            self.statistics.utterancesDiscarded += 1
            logger.debug("Discarded an utterance: %s", result.reason)
            self._returnToWaiting()
            return

        self.statistics.utterancesCaptured += 1
        self.statistics.capturedSeconds += result.utterance.durationSeconds
        logger.info(
            "Captured %.2fs of speech (%s)", result.utterance.durationSeconds, result.reason
        )
        await self._bus.publish(
            AudioDetected(durationSeconds=result.utterance.durationSeconds)
        )

        self._state = ListenerState.busy
        self._turn = asyncio.create_task(self._runTurn(result.utterance))

    async def _whileBusy(self, frame: AudioBuffer, wakeBuffer: _FrameAggregator) -> None:
        """A turn is running. Watch only for an interruption."""
        if not self._allowBargeIn:
            return
        if self._runtime.state is not AssistantState.speaking:
            return

        for block in wakeBuffer.feed(frame):
            if self._wakeWord.detect(block) is None:
                continue

            logger.info("Interrupted by the wake word")
            self.statistics.bargeIns += 1
            wakeBuffer.reset()
            await self._interrupt()
            return

    # --- Turns -----------------------------------------------------------

    async def _runTurn(self, utterance: AudioBuffer) -> TurnResult:
        try:
            return await self._runtime.handleAudio(utterance)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.exception("Turn failed")
            await self._bus.publish(
                AssistantError(message=str(error), component="listener")
            )
            return TurnResult()

    def _reapFinishedTurn(self) -> None:
        if self._state is not ListenerState.busy or self._turn is None:
            return
        if not self._turn.done():
            return

        self.statistics.turnsRun += 1
        self._turn = None
        # Before returning to capture, not after: with no wake word to wait for
        # the listener goes straight back to listening, and the reply it has
        # just spoken is still arriving at the microphone.
        self._guardAgainstEcho()
        self._returnToWaiting()

    async def _awaitTurn(self) -> None:
        """Let a turn still in flight finish, and settle the state.

        The frame stream can end while a turn is running, so without this the
        listener would be left reporting itself busy after it had stopped.
        """
        if self._turn is None:
            return

        if not self._turn.done():
            with contextlib.suppress(asyncio.CancelledError):
                await self._turn

        self._turn = None
        if self._state is ListenerState.busy:
            self.statistics.turnsRun += 1
            self._guardAgainstEcho()
            self._returnToWaiting()

    async def _interrupt(self) -> None:
        if self._turn is not None and not self._turn.done():
            self._turn.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._turn
        self._turn = None
        await self._runtime.interrupt()
        # No echo guard here. Playback was aborted rather than allowed to
        # drain, and the person is already speaking, so holding the microphone
        # shut would swallow the start of what they interrupted to say.
        # An interruption is a fresh request, not a return to idle waiting.
        self._beginCapture()

    # --- Transitions -----------------------------------------------------

    def _beginCapture(self) -> None:
        self._segmenter = SpeechSegmenter(self._segmenterSettings)
        self._vad.reset()
        self._state = ListenerState.capturing
        # Every route here follows something happening: a wake word, the end of
        # a turn, an interruption. The silence starts now, not when the child
        # last spoke, so the count is from when the assistant stopped talking.
        self._noteActivity()

    def _returnToWaiting(self) -> None:
        if self._wakeWord.alwaysAwake:
            self._beginCapture()
            return
        self._wakeWord.reset()
        self._state = ListenerState.waiting

    def describe(self) -> str:
        return (
            f"wake word: {self._wakeWord.describe()}\n"
            f"  vad:       {self._vad.describe()}\n"
            f"  frames:    {self.frameMilliseconds} ms"
        )


class _FrameAggregator:
    """Collects small frames into the larger blocks a model requires.

    The wake-word model wants 80 ms while voice activity detection wants 32 ms.
    Rather than run two capture streams, the stream delivers the smaller frame
    and this rebuilds the larger one.
    """

    __slots__ = ("_blockMilliseconds", "_frameMilliseconds", "_needed", "_pending")

    def __init__(self, blockMilliseconds: int, frameMilliseconds: int) -> None:
        self._blockMilliseconds = blockMilliseconds
        self._frameMilliseconds = frameMilliseconds
        self._needed = max(1, round(blockMilliseconds / max(1, frameMilliseconds)))
        self._pending: list[AudioBuffer] = []

    def reset(self) -> None:
        self._pending.clear()

    def feed(self, frame: AudioBuffer) -> list[AudioBuffer]:
        """Add a frame, returning any complete blocks."""
        if self._needed <= 1:
            return [frame]

        self._pending.append(frame)
        if len(self._pending) < self._needed:
            return []

        block = AudioBuffer(
            data=b"".join(item.data for item in self._pending),
            sampleRate=frame.sampleRate,
            channels=frame.channels,
        )
        self._pending.clear()
        return [block]


__all__ = ["ContinuousListener", "ListenerState", "ListenerStatistics"]
