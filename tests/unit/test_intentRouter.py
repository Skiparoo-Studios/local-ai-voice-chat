"""Layered intent routing: deterministic first, model second, chat last."""

from __future__ import annotations

import logging

import pytest

from app.assistant.conversation import Conversation
from app.assistant.intentRouter import IntentRouter, buildIntentRouter
from app.assistant.llmHandler import LlmHandler
from app.assistant.toolRegistry import ToolRegistry
from app.automation.simulatedProvider import SimulatedAutomationProvider
from app.config import Settings
from app.intelligence.llmProvider import ToolCall
from app.intelligence.providers.scriptedLlmProvider import ScriptedLlmProvider
from app.tools.homeAutomation import ActionPolicy, HomeAutomationTool, buildAutomationTools


def buildRouter(
    *,
    allowedActions: tuple[str, ...] = (
        "light.turnOn",
        "light.turnOff",
        "light.toggle",
        "switch.turnOn",
        "switch.turnOff",
    ),
    toolCalls: tuple[ToolCall, ...] = (),
    replies: tuple[str, ...] = ("It's fairly mild outside today.",),
    withLlm: bool = True,
    allowStateChanges: bool = True,
) -> tuple[IntentRouter, SimulatedAutomationProvider]:
    provider = SimulatedAutomationProvider()
    registry = ToolRegistry(allowStateChanges=allowStateChanges)

    automationTool = None
    for tool in buildAutomationTools(provider, allowedActions):
        registry.register(tool)
        if isinstance(tool, HomeAutomationTool):
            automationTool = tool

    llmHandler = None
    if withLlm:
        llmHandler = LlmHandler(
            ScriptedLlmProvider(replies=replies, toolCalls=toolCalls), streaming=False
        )

    router = IntentRouter(
        registry, llmHandler=llmHandler, automationTool=automationTool
    )
    return router, provider


class TestLayerOne:
    """Deterministic commands must never reach a model."""

    @pytest.mark.parametrize(
        "utterance,action,deviceId",
        [
            ("turn off the bedroom light", "light.turnOff", "light.bedroom_ceiling"),
            ("turn the bedroom light off", "light.turnOff", "light.bedroom_ceiling"),
            ("turn on the hallway light", "light.turnOn", "light.hallway"),
            ("switch off the lounge lamp", "light.turnOff", "light.lounge_lamp"),
            ("toggle the hallway light", "light.toggle", "light.hallway"),
            ("please turn off the coffee machine", "switch.turnOff", "switch.coffee_machine"),
        ],
    )
    async def testCommandsResolveWithoutAModel(self, utterance, action, deviceId):
        router, provider = buildRouter()

        response = await router.respond(utterance, Conversation())

        assert router.lastLayer == 1
        assert not response.usedLlm
        assert provider.calls[0].action == action
        assert provider.calls[0].parameters["target"] == deviceId

    async def testSwitchDomainIsInferredFromTheDevice(self):
        """Nobody says 'switch domain'; the device list settles it."""
        router, provider = buildRouter()

        await router.respond("turn off the coffee machine", Conversation())

        assert provider.calls[0].action == "switch.turnOff"

    async def testConfirmationIsSpoken(self):
        router, _ = buildRouter()

        response = await router.respond("turn off the bedroom light", Conversation())

        assert "turned" in response.text.lower()

    async def testPluralAddressesEveryMatch(self):
        router, provider = buildRouter()

        await router.respond("turn off all the kitchen lights", Conversation())

        assert len(provider.calls) == 2

    async def testAmbiguousCommandAsks(self):
        router, provider = buildRouter()

        response = await router.respond("turn off the kitchen light", Conversation())

        assert "Did you mean" in response.text
        assert provider.calls == []

    async def testUnrecognisedTargetFallsThroughToTheModel(self):
        """'Turn off the news' is not a device command."""
        router, provider = buildRouter()

        response = await router.respond("turn off the news", Conversation())

        assert router.lastLayer != 1
        assert provider.calls == []
        assert response.usedLlm

    async def testConversationDoesNotTriggerCommands(self):
        router, provider = buildRouter()

        await router.respond("it is quite dark in the kitchen", Conversation())

        assert provider.calls == []

    async def testTheClockIsAnsweredWithoutTheModel(self):
        """A language model does not know the time. Asking one is slower and wrong."""
        router, provider = buildRouter()

        response = await router.respond("what is the time", Conversation())

        assert router.lastLayer == 1
        assert not response.usedLlm
        assert ":" in response.text
        assert provider.calls == []

    @pytest.mark.parametrize(
        "utterance",
        ["what is your name", "hello there", "thank you", "what can you do"],
    )
    async def testDeterministicAnswersSurviveConfiguringAModel(self, utterance):
        """These were lost the moment an llm was configured: the rules handler
        was only ever consulted when there was no model to consult instead."""
        router, _ = buildRouter()

        response = await router.respond(utterance, Conversation())

        assert router.lastLayer == 1
        assert not response.usedLlm

    async def testUnmatchedPhrasingStillReachesTheModel(self):
        """The rules handler echoes as a last resort; the router must not."""
        router, _ = buildRouter(replies=("It's fairly mild outside today.",))

        response = await router.respond("what is the weather like", Conversation())

        assert response.usedLlm
        assert not response.text.startswith("You said:")

    async def testLayerOneWorksWithNoModelConfigured(self):
        """Deterministic commands must work with no model service running."""
        router, provider = buildRouter(withLlm=False)

        response = await router.respond("turn off the bedroom light", Conversation())

        assert router.lastLayer == 1
        assert provider.calls[0].action == "light.turnOff"
        assert "turned" in response.text.lower()


class TestLayerTwo:
    async def testModelProposedToolIsExecuted(self):
        router, provider = buildRouter(
            toolCalls=(
                ToolCall(
                    name="homeAutomation",
                    arguments={"action": "light.turnOn", "target": "lounge lamp"},
                ),
            )
        )

        response = await router.respond("it's a bit gloomy in the lounge", Conversation())

        assert router.lastLayer == 2
        assert response.usedLlm
        assert provider.calls[0].action == "light.turnOn"

    async def testSilenceIsNeverTheAnswer(self):
        """A model returning neither action nor text must not leave the user guessing."""
        router, provider = buildRouter(replies=("",))

        response = await router.respond("what is the weather like", Conversation())

        assert response.text
        assert "not sure" in response.text.lower()
        assert provider.calls == []

    async def testModelReplyIsUsedWhenNoToolApplies(self):
        router, provider = buildRouter(replies=("It's fairly mild outside today.",))

        response = await router.respond("what is the weather like", Conversation())

        assert response.text == "It's fairly mild outside today."
        assert response.usedLlm
        assert provider.calls == []

    async def testToolSchemasAreOfferedToTheModel(self):
        router, _ = buildRouter()

        schemas = router._registry.schemas()

        assert {schema["function"]["name"] for schema in schemas} == {
            "homeAutomation",
            "listDevices",
        }

    async def testModelIsToldNotToClaimUncalledActions(self):
        """A small model will otherwise say it acted when it did not."""
        from app.intelligence.llmProvider import LlmProvider

        captured: list[list[dict]] = []

        class CapturingProvider(ScriptedLlmProvider):
            async def generate(self, messages, tools=None):
                captured.append(messages)
                return await super().generate(messages, tools)

        router, _ = buildRouter()
        router._llm = LlmHandler(CapturingProvider(replies=("ok",)), streaming=False)
        assert isinstance(router._llm.provider, LlmProvider)

        await router.respond("what is the weather like", Conversation())

        systemContent = " ".join(
            message["content"] for message in captured[0] if message["role"] == "system"
        )
        assert "unless you called a tool" in systemContent


class TestModelOutputIsUntrusted:
    """§17: an action outside the allow-list is refused and logged."""

    async def testForbiddenActionFromTheModelIsRefused(self, caplog):
        router, provider = buildRouter(
            allowedActions=("light.turnOff",),
            toolCalls=(
                ToolCall(
                    name="homeAutomation",
                    arguments={"action": "lock.unlock", "target": "front door"},
                ),
            ),
        )

        with caplog.at_level(logging.WARNING):
            response = await router.respond("let me in", Conversation())

        assert provider.calls == []
        assert "not something I'm allowed to do" in response.text
        assert "Refused" in caplog.text

    async def testUnknownToolFromTheModelIsRefused(self):
        router, provider = buildRouter(
            toolCalls=(ToolCall(name="runShellCommand", arguments={"cmd": "rm -rf /"}),)
        )

        response = await router.respond("clean up", Conversation())

        assert provider.calls == []
        assert response.text
        assert "No tool named" in response.text

    async def testInvalidArgumentsFromTheModelAreRefused(self):
        router, provider = buildRouter(
            toolCalls=(ToolCall(name="homeAutomation", arguments={"nonsense": True}),)
        )

        response = await router.respond("do the thing", Conversation())

        assert provider.calls == []
        assert "didn't understand" in response.text

    async def testStateChangesCanBeDisabledEntirely(self):
        router, provider = buildRouter(allowStateChanges=False)

        response = await router.respond("turn off the bedroom light", Conversation())

        assert provider.calls == []
        assert "disabled" in response.text

    async def testReadOnlyToolsStillWorkWhenStateChangesAreDisabled(self):
        router, _ = buildRouter(
            allowStateChanges=False,
            toolCalls=(ToolCall(name="listDevices", arguments={}),),
        )

        response = await router.respond("what can you control", Conversation())

        assert "I can control" in response.text


class TestRouterConfiguration:
    def testBuiltFromSettings(self):
        provider = SimulatedAutomationProvider()
        registry = ToolRegistry()
        for tool in buildAutomationTools(provider):
            registry.register(tool)

        router = buildIntentRouter(Settings(), registry)

        assert isinstance(router, IntentRouter)

    def testRouterDoesNotClaimToStream(self):
        """Tool selection and streaming do not combine, so it says so."""
        router, _ = buildRouter()

        assert not router.supportsStreaming

    def testRouterReportsUsingAModel(self):
        withModel, _ = buildRouter(withLlm=True)
        without, _ = buildRouter(withLlm=False)

        assert withModel.usesLlm
        assert not without.usesLlm

    async def testEmptyInputProducesNoReply(self):
        router, _ = buildRouter()

        assert (await router.respond("  ", Conversation())).isEmpty

    async def testStreamFallsBackToOneFragment(self):
        router, _ = buildRouter()

        fragments = [
            fragment
            async for fragment in router.respondStream(
                "turn off the bedroom light", Conversation()
            )
        ]

        assert len(fragments) == 1


class TestAllowListDefaults:
    def testDefaultsPermitLightsAndSwitchesOnly(self):
        policy = ActionPolicy(tuple(Settings().automation.allowedActions))

        assert policy.permits("light.turnOff")
        assert policy.permits("switch.turnOn")
        assert not policy.permits("lock.unlock")
        assert not policy.permits("cover.openCover")
