"""The assistant runtime: turn orchestration, state machine and events."""

from __future__ import annotations

import pytest

from app.assistant.conversation import Conversation
from app.assistant.handlers import EchoHandler, Response, ResponseHandler, RuleHandler
from app.assistant.runtime import AssistantRuntime, TurnTiming
from app.assistant.session import Session
from app.audio import AudioBuffer, MemoryAudioInput, MemoryAudioOutput
from app.events import (
    AssistantState,
    AssistantStateChanged,
    Event,
    EventBus,
    ResponseGenerated,
    TranscriptionCompleted,
)
from app.speech.models.scriptedProvider import ScriptedProvider
from app.speech.models.toneProvider import ToneProvider


class RecordingHandler(ResponseHandler):
    """Captures what it was asked, and replies with a fixed string."""

    name = "recording"

    def __init__(self, reply: str = "Acknowledged.", shouldStop: bool = False) -> None:
        self.reply = reply
        self.shouldStop = shouldStop
        self.seen: list[str] = []
        self.historyLengths: list[int] = []

    async def respond(self, text: str, conversation: Conversation) -> Response:
        self.seen.append(text)
        self.historyLengths.append(len(conversation))
        return Response(text=self.reply, handledBy=self.name, shouldStop=self.shouldStop)


class FailingHandler(ResponseHandler):
    name = "failing"

    async def respond(self, text: str, conversation: Conversation) -> Response:
        raise RuntimeError("handler exploded")


def buildRuntime(
    *,
    handler: ResponseHandler | None = None,
    phrases: tuple[str, ...] = ("turn the kitchen light off",),
    bus: EventBus | None = None,
    warmUp: bool = False,
    recorded: AudioBuffer | None = None,
) -> tuple[AssistantRuntime, MemoryAudioOutput]:
    output = MemoryAudioOutput()
    source = MemoryAudioInput([recorded] if recorded is not None else None)
    runtime = AssistantRuntime(
        speechToText=ScriptedProvider(phrases=phrases),
        textToSpeech=ToneProvider(),
        handler=handler or EchoHandler(),
        audioInput=source,
        audioOutput=output,
        bus=bus or EventBus(),
        session=Session.create(),
        language="en",
        voice="default",
        warmUpOnStart=warmUp,
    )
    return runtime, output


class TestLifecycle:
    async def testStartLoadsBothProviders(self):
        runtime, _ = buildRuntime()

        await runtime.start()

        assert runtime.isStarted
        assert runtime.state is AssistantState.idle

    async def testStartingTwiceIsHarmless(self):
        runtime, _ = buildRuntime()

        await runtime.start()
        await runtime.start()

        assert runtime.isStarted

    async def testStopUnloadsProviders(self):
        runtime, _ = buildRuntime()
        await runtime.start()

        await runtime.stop()

        assert not runtime.isStarted

    async def testWarmUpDoesNotPlayAudio(self):
        """Warm-up must pay the first-run cost without making a noise."""
        runtime, output = buildRuntime(warmUp=True)

        await runtime.start()

        assert output.played == []

    async def testWarmUpFailureDoesNotBlockStartup(self):
        runtime, _ = buildRuntime(warmUp=True)

        class BrokenTts(ToneProvider):
            async def synthesise(self, text, *, voice=None, language=None):
                raise RuntimeError("no model")

        runtime._tts = BrokenTts()
        await runtime.start()

        assert runtime.isStarted


class TestTextTurn:
    async def testRoundTripProducesSpokenAudio(self):
        runtime, output = buildRuntime()
        await runtime.start()

        result = await runtime.handleText("hello there")

        assert result.handled
        assert result.assistantText == "You said: hello there"
        assert len(output.played) == 1
        assert output.played[0].durationSeconds > 0

    async def testConversationRecordsBothSides(self):
        runtime, _ = buildRuntime()
        await runtime.start()

        await runtime.handleText("hello")

        assert [turn.text for turn in runtime.conversation.turns] == [
            "hello",
            "You said: hello",
        ]

    async def testHandlerSeesHistoryWithoutTheCurrentUtterance(self):
        """Otherwise the current turn would appear twice to a language model."""
        handler = RecordingHandler()
        runtime, _ = buildRuntime(handler=handler)
        await runtime.start()

        await runtime.handleText("first")
        await runtime.handleText("second")

        assert handler.seen == ["first", "second"]
        assert handler.historyLengths == [0, 2]

    async def testEmptyTextIsIgnored(self):
        runtime, output = buildRuntime()
        await runtime.start()

        result = await runtime.handleText("   ")

        assert not result.handled
        assert output.played == []

    async def testEmptyResponseIsNotSpoken(self):
        runtime, output = buildRuntime(handler=RecordingHandler(reply=""))
        await runtime.start()

        result = await runtime.handleText("hello")

        assert not result.handled
        assert output.played == []

    async def testTurnCountIncrements(self):
        runtime, _ = buildRuntime()
        await runtime.start()

        await runtime.handleText("one")
        await runtime.handleText("two")

        assert runtime.session.turnCount == 2

    async def testStopRequestIsSurfaced(self):
        runtime, _ = buildRuntime(handler=RecordingHandler(reply="Goodbye.", shouldStop=True))
        await runtime.start()

        assert (await runtime.handleText("goodbye")).shouldStop


class TestAudioTurn:
    async def testTranscribedAudioDrivesTheTurn(self):
        runtime, output = buildRuntime(phrases=("turn the kitchen light off",))
        await runtime.start()

        result = await runtime.handleAudio(AudioBuffer.tone(440.0, 1.0, 16000))

        assert result.userText == "turn the kitchen light off"
        assert result.assistantText == "You said: turn the kitchen light off"
        assert len(output.played) == 1

    async def testEmptyAudioProducesNoTurn(self):
        runtime, output = buildRuntime()
        await runtime.start()

        result = await runtime.handleAudio(AudioBuffer(data=b"", sampleRate=16000))

        assert not result.handled
        assert output.played == []

    async def testEmptyTranscriptProducesNoReply(self):
        """Push-to-talk regularly captures silence; that is not a question."""
        runtime, output = buildRuntime(phrases=("",))
        await runtime.start()

        result = await runtime.handleAudio(AudioBuffer.tone(440.0, 1.0, 16000))

        assert not result.handled
        assert output.played == []

    async def testRunTurnCapturesThenReplies(self):
        recorded = AudioBuffer.tone(440.0, 1.0, 16000)
        runtime, output = buildRuntime(recorded=recorded)
        await runtime.start()

        result = await runtime.runTurn()

        assert result.userText == "turn the kitchen light off"
        assert len(output.played) == 1


class TestStateMachine:
    """Exit criterion: idle -> listening -> thinking -> speaking -> idle."""

    async def testSpokenTurnFollowsTheExpectedSequence(self):
        bus = EventBus()
        states: list[AssistantState] = []
        bus.subscribe(AssistantStateChanged, lambda event: states.append(event.state))

        runtime, _ = buildRuntime(bus=bus, recorded=AudioBuffer.tone(440.0, 1.0, 16000))
        await runtime.start()
        states.clear()

        await runtime.runTurn()

        assert states == [
            AssistantState.listening,
            AssistantState.thinking,
            AssistantState.speaking,
            AssistantState.idle,
        ]

    async def testTypedTurnSkipsListening(self):
        bus = EventBus()
        states: list[AssistantState] = []
        bus.subscribe(AssistantStateChanged, lambda event: states.append(event.state))

        runtime, _ = buildRuntime(bus=bus)
        await runtime.start()
        states.clear()

        await runtime.handleText("hello")

        assert states == [
            AssistantState.thinking,
            AssistantState.speaking,
            AssistantState.idle,
        ]

    async def testStateChangeCarriesThePreviousState(self):
        bus = EventBus()
        changes: list[AssistantStateChanged] = []
        bus.subscribe(AssistantStateChanged, changes.append)

        runtime, _ = buildRuntime(bus=bus)
        await runtime.start()
        changes.clear()

        await runtime.handleText("hello")

        assert changes[0].previousState is AssistantState.idle
        assert changes[1].previousState is AssistantState.thinking

    async def testRepeatedStateIsNotRepublished(self):
        bus = EventBus()
        changes: list[Event] = []
        bus.subscribe(AssistantStateChanged, changes.append)

        runtime, _ = buildRuntime(bus=bus)
        await runtime.start()
        changes.clear()

        await runtime.handleText("")

        assert changes == []

    async def testRuntimeReturnsToIdleAfterATurn(self):
        runtime, _ = buildRuntime()
        await runtime.start()

        await runtime.handleText("hello")

        assert runtime.state is AssistantState.idle


class TestEvents:
    async def testTurnPublishesTranscriptionAndResponse(self):
        bus = EventBus()
        transcriptions: list[TranscriptionCompleted] = []
        responses: list[ResponseGenerated] = []
        bus.subscribe(TranscriptionCompleted, transcriptions.append)
        bus.subscribe(ResponseGenerated, responses.append)

        runtime, _ = buildRuntime(bus=bus)
        await runtime.start()

        await runtime.handleAudio(AudioBuffer.tone(440.0, 1.0, 16000))

        assert transcriptions[0].text == "turn the kitchen light off"
        assert responses[0].text == "You said: turn the kitchen light off"

    async def testEventsCarryTheSessionIdentifier(self):
        bus = EventBus()
        responses: list[ResponseGenerated] = []
        bus.subscribe(ResponseGenerated, responses.append)

        runtime, _ = buildRuntime(bus=bus)
        await runtime.start()

        await runtime.handleText("hello")

        assert responses[0].sessionId == runtime.session.sessionId

    async def testHandlerFailureReturnsToIdleAndRaises(self):
        runtime, _ = buildRuntime(handler=FailingHandler())
        await runtime.start()

        with pytest.raises(RuntimeError, match="exploded"):
            await runtime.handleText("hello")

        assert runtime.state is AssistantState.idle


class TestTiming:
    async def testTurnRecordsPerStageTimings(self):
        runtime, _ = buildRuntime()
        await runtime.start()

        result = await runtime.handleAudio(AudioBuffer.tone(440.0, 1.0, 16000))

        assert result.timing.transcriptionSeconds >= 0
        assert result.timing.synthesisSeconds > 0
        assert result.timing.audioSeconds > 0

    def testThinkingExcludesPlayback(self):
        """Time to first audio is what a user feels, not total turn length."""
        timing = TurnTiming(
            transcriptionSeconds=0.2,
            responseSeconds=0.1,
            synthesisSeconds=0.5,
            playbackSeconds=3.0,
        )

        assert timing.thinkingSeconds == pytest.approx(0.8)
        assert timing.totalSeconds == pytest.approx(3.8)

    def testDescriptionNamesEachStage(self):
        description = TurnTiming(
            transcriptionSeconds=0.2, responseSeconds=0.1, synthesisSeconds=0.5
        ).describe()

        assert "stt" in description
        assert "think" in description
        assert "tts" in description


class TestRuleHandler:
    async def testGreetingIsRecognised(self):
        response = await RuleHandler().respond("hello there", Conversation())

        assert "how can i help" in response.text.lower()
        assert not response.usedLlm

    async def testGoodbyeRequestsStop(self):
        response = await RuleHandler().respond("goodbye", Conversation())

        assert response.shouldStop

    async def testStopIsAcknowledged(self):
        response = await RuleHandler().respond("stop", Conversation())

        assert response.text == "Stopped."
        assert not response.shouldStop

    async def testWordBoundariesPreventFalseMatches(self):
        """'stopwatch' must not trigger the stop rule."""
        response = await RuleHandler().respond("set a stopwatch", Conversation())

        assert response.text.startswith("You said:")

    async def testAssistantNameIsReported(self):
        response = await RuleHandler(assistantName="Ada").respond(
            "what is your name", Conversation()
        )

        assert "Ada" in response.text

    async def testUnmatchedInputIsEchoed(self):
        response = await RuleHandler().respond("the kitchen is dark", Conversation())

        assert response.text == "You said: the kitchen is dark"

    async def testEmptyInputProducesNoReply(self):
        assert (await RuleHandler().respond("  ", Conversation())).isEmpty

    async def testNothingReachesALanguageModel(self):
        """Layer 1 of §12: deterministic replies need no model."""
        response = await RuleHandler().respond("hello", Conversation())

        assert not response.usedLlm

    def testMatchReportsWhetherARuleFired(self):
        """The router needs to tell a real answer from the echo fallback."""
        handler = RuleHandler()

        assert handler.match("what is the time") is not None
        assert handler.match("the kitchen is dark") is None
        assert handler.match("   ") is None

    def testMatchAnswersTheClock(self):
        response = RuleHandler().match("what time is it")

        assert response is not None
        assert ":" in response.text
