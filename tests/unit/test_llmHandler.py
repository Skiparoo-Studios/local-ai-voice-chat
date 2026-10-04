"""The language model handler, and streaming through the runtime."""

from __future__ import annotations

import asyncio

import pytest

from app.assistant.conversation import Conversation
from app.assistant.handlers import RuleHandler, UnknownHandlerError, createResponseHandler
from app.assistant.llmHandler import LlmHandler, sanitiseForSpeech, sanitiseFragment
from app.assistant.runtime import AssistantRuntime
from app.assistant.session import Session
from app.audio import AudioBuffer, MemoryAudioInput, MemoryAudioOutput
from app.config import Settings
from app.events import AssistantState, AssistantStateChanged, EventBus, ResponseGenerated
from app.intelligence.llmProvider import createLlmProvider
from app.intelligence.providers.scriptedLlmProvider import ScriptedLlmProvider
from app.speech.models.scriptedProvider import ScriptedProvider
from app.speech.models.toneProvider import ToneProvider


def buildRuntime(
    *,
    replies: tuple[str, ...] = ("The kitchen light is now off. Anything else you need?",),
    streaming: bool = True,
    fragmentDelaySeconds: float = 0.0,
    bus: EventBus | None = None,
) -> tuple[AssistantRuntime, MemoryAudioOutput]:
    output = MemoryAudioOutput()
    handler = LlmHandler(
        ScriptedLlmProvider(replies=replies, fragmentDelaySeconds=fragmentDelaySeconds),
        streaming=streaming,
    )
    runtime = AssistantRuntime(
        speechToText=ScriptedProvider(phrases=("turn the kitchen light off",)),
        textToSpeech=ToneProvider(),
        handler=handler,
        audioInput=MemoryAudioInput([AudioBuffer.tone(440.0, 1.0, 16000)]),
        audioOutput=output,
        bus=bus or EventBus(),
        session=Session.create(),
        language="en",
        voice="default",
        warmUpOnStart=False,
        streaming=streaming,
    )
    return runtime, output


def _configured(handler):
    """The handler configuration chose, looking through every wrapper.

    createResponseHandler layers a sentry gate, toddler-mode switch and book
    handler in front of whatever was configured. Each exposes the one it
    wraps as `base`, so peeling until there is no base left reaches the
    handler these tests are actually about.
    """
    while hasattr(handler, "base"):
        handler = handler.base
    return handler


class TestHandlerSelection:
    """Exit criterion: choosing the handler is configuration, not code."""

    def testRulesIsTheDefault(self):
        assert isinstance(_configured(createResponseHandler(Settings())), RuleHandler)

    def testLlmIsSelectedByConfiguration(self):
        settings = Settings(
            assistant={"handler": "llm"},
            llm={"provider": "scripted", "model": "any"},
        )

        handler = createResponseHandler(settings)

        assert isinstance(_configured(handler), LlmHandler)
        assert handler.usesLlm

    def testUnknownHandlerIsRejected(self):
        settings = Settings()
        settings.assistant.handler = "nonsense"

        with pytest.raises(UnknownHandlerError, match="Available:"):
            createResponseHandler(settings)

    def testMisconfigurationIsReportedNotTracebacked(self):
        """A missing model is something to fix, not a bug to debug.

        Every one of these carries a message saying what to do, so main prints
        it on its own; anything absent from the list keeps its traceback.
        """
        from app.main import RUNTIME_ERRORS

        settings = Settings(assistant={"handler": "llm"}, llm={"provider": "ollama"})

        with pytest.raises(RUNTIME_ERRORS):
            createResponseHandler(settings)


class TestMessageBuilding:
    def testSystemPromptLeadsTheMessages(self):
        handler = LlmHandler(ScriptedLlmProvider(), assistantName="Ada")

        messages = handler.buildMessages("hello", Conversation())

        assert messages[0]["role"] == "system"
        assert "Ada" in messages[0]["content"]

    def testCurrentUtteranceIsAppendedLast(self):
        """The runtime records it only afterwards, so the handler must add it."""
        conversation = Conversation()
        conversation.addUser("earlier question")
        conversation.addAssistant("earlier answer")

        messages = LlmHandler(ScriptedLlmProvider()).buildMessages("new question", conversation)

        assert messages[-1] == {"role": "user", "content": "new question"}
        assert [message["role"] for message in messages] == [
            "system",
            "user",
            "assistant",
            "user",
        ]

    def testCustomSystemPromptReplacesTheDefault(self):
        handler = LlmHandler(ScriptedLlmProvider(), systemPrompt="Be terse.")

        assert handler.buildMessages("x", Conversation())[0]["content"] == "Be terse."

    def testDefaultPromptForbidsMarkdown(self):
        """Spoken replies must not contain formatting, so the prompt says so."""
        prompt = LlmHandler(ScriptedLlmProvider()).buildMessages("x", Conversation())[0]

        assert "markdown" in prompt["content"].lower()


class TestSanitising:
    """§17: model output is untrusted. It is also spoken, so it must be plain."""

    def testBoldAndItalicMarkersAreRemoved(self):
        assert sanitiseForSpeech("The **kitchen** light is *off*") == (
            "The kitchen light is off"
        )

    def testHeadingsAreRemoved(self):
        assert sanitiseForSpeech("## Result\nThe light is off") == "Result\nThe light is off"

    def testBulletsAreRemoved(self):
        cleaned = sanitiseForSpeech("- first item\n- second item")

        assert "-" not in cleaned
        assert "first item" in cleaned

    def testCodeBlocksAreRemoved(self):
        cleaned = sanitiseForSpeech("Try this:\n```python\nprint(1)\n```\nDone")

        assert "print" not in cleaned
        assert "Done" in cleaned

    def testInlineCodeKeepsItsContent(self):
        assert sanitiseForSpeech("Run `ollama serve` first") == "Run ollama serve first"

    def testLinksKeepTheirText(self):
        assert sanitiseForSpeech("See [the docs](http://example.com)") == "See the docs"

    def testPlainTextIsUnchanged(self):
        text = "The kitchen light is now off."

        assert sanitiseForSpeech(text) == text

    def testFragmentSanitisingDropsStrayMarkers(self):
        assert sanitiseFragment("**bold") == "bold"
        assert sanitiseFragment(" plain ") == " plain "


class TestStreamingRuntime:
    async def testStreamedTurnSpeaksSeveralPieces(self):
        """Each sentence becomes its own utterance, played in order."""
        runtime, output = buildRuntime(
            replies=("The kitchen light is now off. Would you like the lamp on instead?",)
        )
        await runtime.start()

        result = await runtime.handleText("turn the kitchen light off")

        assert result.handled
        assert len(output.played) == 2

    async def testStreamedReplyTextIsComplete(self):
        reply = "The kitchen light is now off. Would you like the lamp on instead?"
        runtime, _ = buildRuntime(replies=(reply,))
        await runtime.start()

        result = await runtime.handleText("hello")

        assert result.assistantText.split() == reply.split()

    async def testStreamedTurnIsMarkedAsStreamed(self):
        runtime, _ = buildRuntime()
        await runtime.start()

        result = await runtime.handleText("hello")

        assert result.timing.streamed
        assert result.timing.firstAudioSeconds > 0

    async def testNonStreamedTurnIsNotMarked(self):
        runtime, _ = buildRuntime(streaming=False)
        await runtime.start()

        result = await runtime.handleText("hello")

        assert not result.timing.streamed

    async def testFirstAudioIsMeasuredOnBothPaths(self):
        """Synthesis streams even when the reply does not, because most of
        what the assistant says is a single sentence."""
        streamed, _ = buildRuntime(streaming=True)
        complete, _ = buildRuntime(streaming=False)
        await streamed.start()
        await complete.start()

        first = await streamed.handleText("hello")
        second = await complete.handleText("hello")

        assert first.timing.firstAudioSeconds > 0
        assert second.timing.firstAudioSeconds > 0

    async def testStreamingReachesFirstAudioBeforeGenerationEnds(self):
        """The whole point: first audio must precede the end of generation."""
        runtime, _ = buildRuntime(
            replies=(
                "The kitchen light is now off. "
                "The hallway light is still on. "
                "Would you like that one off as well?",
            ),
            fragmentDelaySeconds=0.01,
        )
        await runtime.start()

        result = await runtime.handleText("hello")

        assert result.timing.firstAudioSeconds < result.timing.responseSeconds

    async def testStreamedAndCompleteRepliesAgreeOnText(self):
        reply = "The kitchen light is now off. Would you like the lamp on instead?"

        streamed, _ = buildRuntime(replies=(reply,), streaming=True)
        complete, _ = buildRuntime(replies=(reply,), streaming=False)
        await streamed.start()
        await complete.start()

        first = await streamed.handleText("hello")
        second = await complete.handleText("hello")

        assert first.assistantText.split() == second.assistantText.split()

    async def testStreamedTurnRecordsConversation(self):
        runtime, _ = buildRuntime()
        await runtime.start()

        await runtime.handleText("turn the kitchen light off")

        assert [turn.role.value for turn in runtime.conversation.turns] == [
            "user",
            "assistant",
        ]

    async def testStreamedTurnFollowsTheStateSequence(self):
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

    async def testStreamedTurnReportsUsingTheModel(self):
        bus = EventBus()
        responses: list[ResponseGenerated] = []
        bus.subscribe(ResponseGenerated, responses.append)

        runtime, _ = buildRuntime(bus=bus)
        await runtime.start()

        await runtime.handleText("hello")

        assert responses[0].usedLlm

    async def testStreamedTurnProducesJoinedAudio(self):
        runtime, output = buildRuntime(
            replies=("The light is now off. The heating is still on.",)
        )
        await runtime.start()

        result = await runtime.handleText("hello")

        assert result.audio is not None
        playedSeconds = sum(piece.durationSeconds for piece in output.played)
        assert result.audio.durationSeconds == pytest.approx(playedSeconds, abs=0.01)


class TestWarmUp:
    """§18: models must not be loaded per request. A service loads on first use."""

    async def testWarmUpMakesARequest(self):
        from app.intelligence.llmProvider import LlmResponse

        calls: list[list[dict]] = []

        class CountingProvider(ScriptedLlmProvider):
            async def generate(self, messages, tools=None):
                calls.append(messages)
                return LlmResponse(text="ready")

        handler = LlmHandler(CountingProvider(), streaming=False)
        await handler.warmUp()

        assert len(calls) == 1

    async def testWarmUpFailureDoesNotBlockStartup(self):
        class BrokenProvider(ScriptedLlmProvider):
            async def generate(self, messages, tools=None):
                raise RuntimeError("service down")

        await LlmHandler(BrokenProvider(), streaming=False).warmUp()

    async def testRuntimeWarmsTheHandler(self):
        """The runtime warmed speech but not the handler, which is where the
        first-turn cost actually landed."""
        warmed: list[bool] = []

        class RecordingHandler(RuleHandler):
            async def warmUp(self) -> None:
                warmed.append(True)

        runtime, _ = buildRuntime()
        runtime._handler = RecordingHandler()
        runtime._warmUpOnStart = True
        await runtime.start()

        assert warmed == [True]

    async def testHandlersNeedNotImplementWarmUp(self):
        await RuleHandler().warmUp()


class TestModelResidency:
    def testKeepAliveDefaultsToLongerThanOllamas(self):
        """Ollama unloads after five minutes; an assistant used every ten
        would then reload on every turn."""
        assert Settings().llm.keepAlive == "30m"

    def testKeepAliveReachesTheProvider(self):
        settings = Settings(
            llm={"provider": "ollama", "model": "test", "keepAlive": "-1"}
        )
        provider = createLlmProvider(settings)

        assert provider._keepAlive == "-1"


class TestStreamingFailures:
    async def testGenerationFailureDoesNotStrandThePlayer(self):
        """A mid-stream failure must not leave the turn hanging on a task."""

        class BrokenProvider(ScriptedLlmProvider):
            async def generateStream(self, messages):
                yield "The light is now off. "
                raise RuntimeError("model died")

        runtime, _ = buildRuntime()
        runtime._handler = LlmHandler(BrokenProvider(), streaming=True)
        await runtime.start()

        with pytest.raises(RuntimeError, match="model died"):
            await asyncio.wait_for(runtime.handleText("hello"), timeout=5.0)

        assert runtime.state is AssistantState.idle

    async def testRuleHandlerStillWorksWithStreamingEnabled(self):
        """Handlers that do not stream fall back to the complete path."""
        runtime, output = buildRuntime()
        runtime._handler = RuleHandler()
        await runtime.start()

        result = await runtime.handleText("hello there")

        assert "How can I help" in result.assistantText
        assert len(output.played) == 1
        assert not result.timing.streamed

    async def testDefaultStreamYieldsOneFragment(self):
        """Any handler can be driven through the streaming path."""
        fragments = [
            fragment async for fragment in RuleHandler().respondStream("hello", Conversation())
        ]

        assert len(fragments) == 1
