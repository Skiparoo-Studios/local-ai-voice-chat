"""The hands-free listening loop."""

from __future__ import annotations

import asyncio
import time

import pytest

from app.assistant.handlers import EchoHandler
from app.assistant.listener import ContinuousListener, ListenerState
from app.assistant.runtime import AssistantRuntime
from app.assistant.session import Session
from app.audio.audioBuffer import AudioBuffer
from app.audio.input import AudioInput
from app.audio.output import MemoryAudioOutput
from app.audio.vad import EnergyVadProvider, SegmenterSettings, SegmentState
from app.audio.wakeWord import (
    AlwaysAwakeProvider,
    Detection,
    ManualWakeWordProvider,
    UnknownWakeWordProviderError,
    WakeWordProvider,
    createWakeWordProvider,
    resolveWakeWordProviderClass,
)
from app.config import Settings
from app.events import EventBus, WakeWordDetected
from app.speech.models.scriptedProvider import ScriptedProvider
from app.speech.models.toneProvider import ToneProvider

RATE = 16000
FRAME_MS = 32


def speechFrame() -> AudioBuffer:
    return AudioBuffer.tone(220.0, FRAME_MS / 1000, RATE, amplitude=0.5, fadeSeconds=0.0)


def silenceFrame() -> AudioBuffer:
    return AudioBuffer.silence(FRAME_MS / 1000, RATE)


class ScriptedFrameInput(AudioInput):
    """Replays a pattern of speech and silence, then stops."""

    def __init__(self, pattern: str) -> None:
        self._pattern = pattern

    @property
    def sampleRate(self) -> int:
        return RATE

    async def frames(self, frameMilliseconds: int = FRAME_MS):
        for symbol in self._pattern:
            yield speechFrame() if symbol == "s" else silenceFrame()
            # Real time, not just a yield: a turn runs synthesis and
            # transcription in worker threads, which a bare sleep(0) does not
            # give them the chance to finish.
            await asyncio.sleep(0.001)


class TriggeringWakeWord(WakeWordProvider):
    """Fires once, after a set number of frames."""

    name = "triggering"

    def __init__(self, afterFrames: int = 1) -> None:
        self._afterFrames = afterFrames
        self._seen = 0
        self.fired = 0

    @classmethod
    def fromSettings(cls, settings: Settings) -> TriggeringWakeWord:
        return cls()

    @property
    def frameMilliseconds(self) -> int:
        return FRAME_MS

    def detect(self, frame: AudioBuffer) -> Detection | None:
        self._seen += 1
        if self._seen == self._afterFrames:
            self.fired += 1
            return Detection(word="test", confidence=1.0)
        return None


class CountingFrameInput(ScriptedFrameInput):
    """Reports how often the listener threw queued audio away."""

    def __init__(self, pattern: str) -> None:
        super().__init__(pattern)
        self.drainCalls = 0
        self.queued = 0

    def drain(self) -> int:
        self.drainCalls += 1
        dropped, self.queued = self.queued, 0
        return dropped


class NeverFiringWakeWord(WakeWordProvider):
    """Keeps the listener in the waiting state."""

    name = "never"

    @classmethod
    def fromSettings(cls, settings: Settings) -> NeverFiringWakeWord:
        return cls()

    @property
    def frameMilliseconds(self) -> int:
        return FRAME_MS

    def detect(self, frame: AudioBuffer) -> Detection | None:
        return None


class PromptingHandler(EchoHandler):
    """A handler that fills silences, as toddler mode does."""

    def __init__(self, *, afterSeconds: float = 0.0, reply: str | None = "Still there?") -> None:
        self._afterSeconds = afterSeconds
        self._reply = reply
        self.promptCalls = 0

    @property
    def promptAfterSeconds(self) -> float | None:
        return self._afterSeconds

    async def promptWhenSilent(self):
        from app.assistant.handlers import Response

        self.promptCalls += 1
        return Response(text=self._reply) if self._reply else None


def buildListener(
    pattern: str,
    *,
    wakeWord: WakeWordProvider | None = None,
    bus: EventBus | None = None,
    allowBargeIn: bool = True,
    echoGuardSeconds: float = 0.0,
    audioInput: AudioInput | None = None,
) -> tuple[ContinuousListener, MemoryAudioOutput, AssistantRuntime]:
    output = MemoryAudioOutput()
    runtime = AssistantRuntime(
        speechToText=ScriptedProvider(phrases=("turn the kitchen light off",)),
        textToSpeech=ToneProvider(),
        handler=EchoHandler(),
        audioInput=ScriptedFrameInput(""),
        audioOutput=output,
        bus=bus or EventBus(),
        session=Session.create(),
        language="en",
        voice="default",
        warmUpOnStart=False,
    )
    listener = ContinuousListener(
        runtime,
        audioInput or ScriptedFrameInput(pattern),
        wakeWord or AlwaysAwakeProvider(),
        EnergyVadProvider(threshold=0.02),
        bus or EventBus(),
        segmenter=SegmenterSettings(
            startFrames=2, silenceFrames=3, prerollFrames=2, minimumSeconds=0.0
        ),
        allowBargeIn=allowBargeIn,
        # Off unless a test asks for it: the existing cases are about capture,
        # and a guard would suppress the second utterance in most of them.
        echoGuardSeconds=echoGuardSeconds,
    )
    return listener, output, runtime


class TestAlwaysAwake:
    async def testUtteranceIsCapturedAndAnswered(self):
        listener, output, _ = buildListener("..ssssssssss....." + "." * 20)
        await listener.start()

        statistics = await listener.run()

        assert statistics.utterancesCaptured == 1
        assert statistics.turnsRun == 1
        assert len(output.played) == 1

    async def testSilenceAloneProducesNothing(self):
        listener, output, _ = buildListener("." * 60)
        await listener.start()

        statistics = await listener.run()

        assert statistics.utterancesCaptured == 0
        assert output.played == []

    async def testTwoUtterancesProduceTwoTurns(self):
        """Speech during a turn is ignored, so the gap must outlast the reply."""
        pattern = "..ssssssss....." + "." * 60 + "ssssssss....." + "." * 60
        listener, output, _ = buildListener(pattern)
        await listener.start()

        statistics = await listener.run()

        assert statistics.utterancesCaptured == 2
        assert len(output.played) == 2

    async def testSpeechDuringATurnDoesNotStartACapture(self):
        """Deliberate: the assistant does not transcribe over its own reply.

        Driven through the state machine rather than by timing, because the
        wall clock makes this racy: on Windows a 1 ms sleep is really 15 ms.
        """
        listener, _, _ = buildListener("")
        await listener.start()
        listener._state = ListenerState.busy

        from app.assistant.listener import _FrameAggregator

        aggregator = _FrameAggregator(FRAME_MS, FRAME_MS)
        for _ in range(30):
            await listener._whileBusy(speechFrame(), aggregator)

        assert listener.state is ListenerState.busy
        assert listener.statistics.utterancesCaptured == 0

    async def testItStartsCapturingImmediately(self):
        listener, _, _ = buildListener("")
        await listener.start()
        await listener.run()

        assert listener.state is ListenerState.capturing


class TestNotHearingItself:
    """The assistant was picking up its own reply and answering it.

    Capture never stops, so when a turn ends two things are still arriving at
    the microphone: audio recorded while the assistant was speaking and still
    queued, and the tail of the reply itself --- PortAudio's write() returns
    with about 180 ms left in the device buffer, which the room reverberates.
    With no wake word to wait for, the listener goes straight back to capturing
    into both.
    """

    async def testQueuedAudioIsThrownAwayWhenATurnEnds(self):
        audioInput = CountingFrameInput("..ssssssssss....." + "." * 20)
        listener, _, _ = buildListener("", audioInput=audioInput)
        await listener.start()

        statistics = await listener.run()

        assert statistics.turnsRun == 1
        assert audioInput.drainCalls == 1

    async def testDiscardedFramesAreCounted(self):
        audioInput = CountingFrameInput("..ssssssssss....." + "." * 20)
        audioInput.queued = 7
        listener, _, _ = buildListener("", audioInput=audioInput)
        await listener.start()

        statistics = await listener.run()

        assert statistics.framesGuarded >= 7

    async def testSpeechIsIgnoredForAMomentAfterTheReply(self):
        """The guard is far longer than the test takes, so anything after the
        first turn must be suppressed."""
        pattern = "..ssssssss....." + "." * 30 + "ssssssssss" + "." * 30
        listener, output, _ = buildListener(pattern, echoGuardSeconds=30.0)
        await listener.start()

        statistics = await listener.run()

        assert statistics.utterancesCaptured == 1
        assert len(output.played) == 1
        assert statistics.framesGuarded > 0

    async def testTheGuardIsHeldForTheConfiguredTime(self):
        listener, _, _ = buildListener("", echoGuardSeconds=30.0)

        listener._guardAgainstEcho()

        assert listener._withinEchoGuard()

    async def testTheGuardLiftsWhenItExpires(self):
        listener, _, _ = buildListener("", echoGuardSeconds=30.0)
        listener._guardAgainstEcho()

        listener._ignoreUntil = time.monotonic() - 0.001

        assert not listener._withinEchoGuard()

    async def testTheSegmenterIsResetWhenTheGuardLifts(self):
        """Guarded frames never reached it, so a part-built utterance must not
        carry on as though nothing had been missed."""
        listener, _, _ = buildListener("", echoGuardSeconds=30.0)
        await listener.start()
        listener._segmenter.feed(speechFrame(), True)
        listener._segmenter.feed(speechFrame(), True)

        listener._guardAgainstEcho()
        listener._ignoreUntil = time.monotonic() - 0.001
        listener._withinEchoGuard()

        assert listener._segmenter.state is SegmentState.waiting

    async def testTheGuardCanBeTurnedOff(self):
        """Headphones make it unnecessary, and it costs responsiveness."""
        listener, _, _ = buildListener("", echoGuardSeconds=0.0)

        listener._guardAgainstEcho()

        assert not listener._withinEchoGuard()

    async def testQueuedAudioIsStillDiscardedWithTheGuardOff(self):
        """Draining and waiting are separate defences; only one is optional."""
        audioInput = CountingFrameInput("")
        listener, _, _ = buildListener("", echoGuardSeconds=0.0, audioInput=audioInput)

        listener._guardAgainstEcho()

        assert audioInput.drainCalls == 1

    async def testAnInterruptionIsNotGuarded(self):
        """The person is already mid-sentence and playback was aborted rather
        than allowed to drain, so a guard would swallow what they came to say."""
        listener, _, _ = buildListener("", echoGuardSeconds=30.0)
        await listener.start()

        await listener._interrupt()

        assert not listener._withinEchoGuard()
        assert listener.state is ListenerState.capturing


class TestFillingASilence:
    """Handlers may say something when nobody has spoken for a while. Almost
    none do; toddler mode is the exception."""

    async def testNothingIsSaidByDefault(self):
        listener, output, _ = buildListener("." * 40)
        await listener.start()

        statistics = await listener.run()

        assert statistics.promptsSpoken == 0
        assert output.played == []

    async def testAPromptingHandlerIsSpokenAfterTheDelay(self):
        listener, output, runtime = buildListener("." * 40)
        runtime._handler = PromptingHandler(afterSeconds=0.0)
        await listener.start()

        statistics = await listener.run()

        assert statistics.promptsSpoken >= 1
        assert output.played

    async def testNothingIsSaidBeforeTheDelayElapses(self):
        listener, output, runtime = buildListener("." * 40)
        runtime._handler = PromptingHandler(afterSeconds=60.0)
        await listener.start()

        statistics = await listener.run()

        assert statistics.promptsSpoken == 0
        assert output.played == []

    async def testAChildMidSentenceIsNotInterrupted(self):
        """Asking whether someone is still there while they are answering is
        exactly the wrong moment."""
        listener, _, runtime = buildListener("")
        runtime._handler = PromptingHandler(afterSeconds=0.0)
        await listener.start()
        listener._segmenter.feed(speechFrame(), True)
        listener._segmenter.feed(speechFrame(), True)
        assert listener._segmenter.state is not SegmentState.waiting

        await listener._promptIfSilent()

        assert listener.statistics.promptsSpoken == 0

    async def testAHandlerWithNothingToSayIsNotRetriedEveryFrame(self):
        """It declined once; asking again 32 ms later would be pointless."""
        listener, _, runtime = buildListener("")
        handler = PromptingHandler(afterSeconds=1.0, reply=None)
        runtime._handler = handler
        await listener.start()
        listener._lastActivity = time.monotonic() - 10

        await listener._promptIfSilent()
        await listener._promptIfSilent()

        assert handler.promptCalls == 1

    async def testTheDelayIsWaitedAgainAfterADeclinedPrompt(self):
        listener, _, runtime = buildListener("")
        handler = PromptingHandler(afterSeconds=1.0, reply=None)
        runtime._handler = handler
        await listener.start()

        listener._lastActivity = time.monotonic() - 10
        await listener._promptIfSilent()
        listener._lastActivity = time.monotonic() - 10
        await listener._promptIfSilent()

        assert handler.promptCalls == 2

    async def testTheClockRestartsWhenTheAssistantStopsSpeaking(self):
        """Counted from the end of the reply, not from when the child spoke,
        or a long reply would be followed instantly by a prompt."""
        listener, _, _ = buildListener("")
        before = listener._lastActivity

        listener._beginCapture()

        assert listener._lastActivity >= before

    async def testPromptingIsSkippedWhileWaitingForAWakeWord(self):
        """It has not been addressed, so it has nothing to follow up on."""
        listener, output, runtime = buildListener(
            "." * 40, wakeWord=NeverFiringWakeWord()
        )
        runtime._handler = PromptingHandler(afterSeconds=0.0)
        await listener.start()

        statistics = await listener.run()

        assert listener.state is ListenerState.waiting
        assert statistics.promptsSpoken == 0
        assert output.played == []


class TestDraining:
    def testTheDefaultInputHasNothingToDrain(self):
        assert ScriptedFrameInput("").drain() == 0

    async def testTheMicrophoneDrainsItsQueue(self):
        """SoundDeviceInput holds the queue only while frames() is running."""
        from app.audio.input import SoundDeviceInput

        microphone = SoundDeviceInput.__new__(SoundDeviceInput)
        microphone._queue = asyncio.Queue()
        for _ in range(5):
            microphone._queue.put_nowait(b"\x00\x00")

        assert microphone.drain() == 5
        assert microphone.drain() == 0

    def testDrainingBeforeCaptureStartsIsHarmless(self):
        from app.audio.input import SoundDeviceInput

        microphone = SoundDeviceInput.__new__(SoundDeviceInput)
        microphone._queue = None

        assert microphone.drain() == 0


class TestWakeWord:
    async def testNothingIsCapturedBeforeTheWakeWord(self):
        listener, output, _ = buildListener(
            "ssssssssss" + "." * 20, wakeWord=TriggeringWakeWord(afterFrames=999)
        )
        await listener.start()

        statistics = await listener.run()

        assert statistics.wakeWordsHeard == 0
        assert statistics.utterancesCaptured == 0
        assert output.played == []

    async def testCaptureBeginsAfterTheWakeWord(self):
        listener, output, _ = buildListener(
            "..ssssssssss....." + "." * 20, wakeWord=TriggeringWakeWord(afterFrames=1)
        )
        await listener.start()

        statistics = await listener.run()

        assert statistics.wakeWordsHeard == 1
        assert statistics.utterancesCaptured == 1
        assert len(output.played) == 1

    async def testWakeWordIsPublishedAsAnEvent(self):
        bus = EventBus()
        heard: list[WakeWordDetected] = []
        bus.subscribe(WakeWordDetected, heard.append)

        listener, _, _ = buildListener(
            "..ssssssssss....." + "." * 20,
            wakeWord=TriggeringWakeWord(afterFrames=1),
            bus=bus,
        )
        await listener.start()
        await listener.run()

        assert len(heard) == 1
        assert heard[0].word == "test"

    async def testListenerReturnsToWaitingAfterATurn(self):
        listener, _, _ = buildListener(
            "..ssssssssss....." + "." * 25, wakeWord=TriggeringWakeWord(afterFrames=1)
        )
        await listener.start()
        await listener.run()

        assert listener.state is ListenerState.waiting


class _AlwaysDetects:
    """Stands in for the wake-word model, reporting certainty every frame."""

    def predict(self, samples):
        return {"test": 1.0}


class TestWakeWordProviders:
    def testAlwaysAwakeReportsItself(self):
        assert AlwaysAwakeProvider().alwaysAwake
        assert not ManualWakeWordProvider().alwaysAwake

    def testManualProviderFiresOnceWhenTriggered(self):
        provider = ManualWakeWordProvider()

        assert provider.detect(silenceFrame()) is None

        provider.trigger()
        assert provider.detect(silenceFrame()) is not None
        assert provider.detect(silenceFrame()) is None

    def testRefractoryPeriodIsMeasuredInAudioTime(self):
        """Wall-clock timing swallows every detection when replaying a file.

        A recording is processed far faster than real time, so a wall-clock
        refractory period suppresses everything after the first hit.
        """
        from app.audio.wakeWord import OpenWakeWordProvider

        provider = OpenWakeWordProvider(threshold=0.5, refractorySeconds=2.0)
        provider._model = _AlwaysDetects()
        provider._numpy = pytest.importorskip("numpy")

        # 80 ms frames: 25 of them is two seconds of audio, processed instantly.
        frame = AudioBuffer.silence(0.08, RATE)
        detections = [provider.detect(frame) for _ in range(60)]
        heard = [detection for detection in detections if detection is not None]

        assert len(heard) >= 2

    def testRegistryResolvesProviders(self):
        assert resolveWakeWordProviderClass("always-awake") is AlwaysAwakeProvider

    def testUnknownProviderListsAvailableOnes(self):
        with pytest.raises(UnknownWakeWordProviderError, match="Available:"):
            resolveWakeWordProviderClass("nope")

    def testConfigurationSelectsTheProvider(self):
        provider = createWakeWordProvider(Settings(wakeWord={"provider": "always-awake"}))

        assert isinstance(provider, AlwaysAwakeProvider)


class TestLifecycle:
    async def testStopEndsTheLoop(self):
        listener, _, _ = buildListener("." * 1000)
        await listener.start()

        async def stopSoon() -> None:
            await asyncio.sleep(0.01)
            await listener.stop()

        asyncio.create_task(stopSoon())  # noqa: RUF006
        statistics = await asyncio.wait_for(listener.run(), timeout=5.0)

        assert statistics.framesSeen > 0

    async def testStatisticsAreDescribed(self):
        listener, _, _ = buildListener("..ssssssssss....." + "." * 20)
        await listener.start()

        statistics = await listener.run()

        assert "turn" in statistics.describe()

    async def testFrameSizeIsTheSmallerOfTheTwoModels(self):
        listener, _, _ = buildListener("")

        assert listener.frameMilliseconds == FRAME_MS

    async def testFailingTurnDoesNotEndTheSession(self):
        """One bad turn must not stop the assistant listening."""
        listener, _, runtime = buildListener("..ssssssssss....." + "." * 25)

        class BrokenHandler(EchoHandler):
            async def respond(self, text, conversation):
                raise RuntimeError("handler died")

        runtime._handler = BrokenHandler()
        await listener.start()

        statistics = await listener.run()

        assert statistics.utterancesCaptured == 1
        assert statistics.turnsRun == 1
