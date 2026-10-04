"""The tool system, its security boundary, and the automation tool."""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel, Field

from app.assistant.toolRegistry import ToolRegistry
from app.automation.automationProvider import Device
from app.automation.homeAssistant import translateAction
from app.automation.simulatedProvider import SimulatedAutomationProvider
from app.tools.base import (
    Tool,
    ToolError,
    ToolNotPermittedError,
    ToolResult,
    ToolValidationError,
)
from app.tools.homeAutomation import (
    ActionPolicy,
    HomeAutomationInput,
    HomeAutomationTool,
    ListDevicesTool,
    buildAutomationTools,
)


class SampleInput(BaseModel):
    value: int = Field(description="A number")


class SampleTool(Tool):
    name = "sample"
    description = "A tool for tests"
    inputModel = SampleInput
    readOnly = True

    def __init__(self) -> None:
        self.calls: list[int] = []

    async def execute(self, parameters: BaseModel) -> ToolResult:
        self.calls.append(parameters.value)
        return ToolResult.ok(f"value was {parameters.value}")


class MutatingTool(SampleTool):
    name = "mutating"
    readOnly = False


class ExplodingTool(SampleTool):
    name = "exploding"

    async def execute(self, parameters: BaseModel) -> ToolResult:
        raise RuntimeError("something broke")


class TestRegistry:
    async def testRegisteredToolCanBeInvoked(self):
        registry = ToolRegistry()
        tool = SampleTool()
        registry.register(tool)

        result = await registry.invoke("sample", {"value": 7})

        assert result.succeeded
        assert tool.calls == [7]

    def testDuplicateRegistrationIsRejected(self):
        registry = ToolRegistry()
        registry.register(SampleTool())

        with pytest.raises(ToolError, match="already registered"):
            registry.register(SampleTool())

    async def testUnknownToolIsRefused(self):
        with pytest.raises(ToolNotPermittedError, match="No tool named"):
            await ToolRegistry().invoke("nope", {})

    async def testToolFailureIsWrapped(self):
        registry = ToolRegistry()
        registry.register(ExplodingTool())

        with pytest.raises(ToolError, match="failed"):
            await registry.invoke("exploding", {"value": 1})

    def testLengthAndMembership(self):
        registry = ToolRegistry()
        registry.register(SampleTool())

        assert len(registry) == 1
        assert registry.has("sample")
        assert not registry.has("other")


class TestValidation:
    """§17: every tool invocation is validated. Model output arrives here."""

    async def testMissingArgumentIsRejected(self):
        registry = ToolRegistry()
        registry.register(SampleTool())

        with pytest.raises(ToolValidationError, match="not valid"):
            await registry.invoke("sample", {})

    async def testWrongTypeIsRejected(self):
        registry = ToolRegistry()
        registry.register(SampleTool())

        with pytest.raises(ToolValidationError):
            await registry.invoke("sample", {"value": "not a number"})

    async def testValidationErrorNamesTheField(self):
        registry = ToolRegistry()
        registry.register(SampleTool())

        with pytest.raises(ToolValidationError, match="value"):
            await registry.invoke("sample", {})

    async def testToolIsNotRunWhenValidationFails(self):
        registry = ToolRegistry()
        tool = SampleTool()
        registry.register(tool)

        with pytest.raises(ToolValidationError):
            await registry.invoke("sample", {"value": "x"})

        assert tool.calls == []


class TestStateChangePolicy:
    """§17: read-only tools are separated from those that change the world."""

    async def testMutatingToolIsRefusedWhenDisabled(self):
        registry = ToolRegistry(allowStateChanges=False)
        registry.register(MutatingTool())

        with pytest.raises(ToolNotPermittedError, match="physical environment"):
            await registry.invoke("mutating", {"value": 1})

    async def testReadOnlyToolStillWorksWhenDisabled(self):
        registry = ToolRegistry(allowStateChanges=False)
        registry.register(SampleTool())

        assert (await registry.invoke("sample", {"value": 1})).succeeded

    def testToolsDeclareWhetherTheyMutate(self):
        assert SampleTool().readOnly
        assert not SampleTool().mutatesPhysicalState
        assert MutatingTool().mutatesPhysicalState


class TestSchemas:
    def testSchemaMatchesFunctionCallingShape(self):
        schema = SampleTool().toSchema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "sample"
        assert "value" in schema["function"]["parameters"]["properties"]

    def testRegistryExportsAllSchemas(self):
        registry = ToolRegistry()
        registry.register(SampleTool())
        registry.register(MutatingTool())

        assert len(registry.schemas()) == 2

    def testReadOnlySchemasCanBeRequestedAlone(self):
        """Restricting what a model can see is the cheapest control there is."""
        registry = ToolRegistry()
        registry.register(SampleTool())
        registry.register(MutatingTool())

        schemas = registry.schemas(readOnlyOnly=True)

        assert [schema["function"]["name"] for schema in schemas] == ["sample"]


class TestActionPolicy:
    """§17: an allow-list of permitted automation actions."""

    def testPermittedActionPasses(self):
        ActionPolicy(("light.turnOff",)).check("light.turnOff")

    def testUnlistedActionIsRefused(self):
        with pytest.raises(ToolNotPermittedError, match="not on the list"):
            ActionPolicy(("light.turnOff",)).check("light.turnOn")

    def testWildcardsArePermitted(self):
        policy = ActionPolicy(("light.*",))

        assert policy.permits("light.turnOn")
        assert policy.permits("light.setBrightness")
        assert not policy.permits("lock.unlock")

    def testSensitiveActionsGetASpecificExplanation(self):
        with pytest.raises(ToolNotPermittedError, match="deliberately"):
            ActionPolicy().check("lock.unlock")

    def testDefaultsExcludeLocksAndCovers(self):
        policy = ActionPolicy()

        assert not policy.permits("lock.unlock")
        assert not policy.permits("cover.openCover")
        assert policy.permits("light.turnOff")


class TestHomeAutomationTool:
    async def testTurningALightOffReachesTheProvider(self):
        provider = SimulatedAutomationProvider()
        tool = HomeAutomationTool(provider)

        result = await tool.execute(
            HomeAutomationInput(action="light.turnOff", target="bedroom light")
        )

        assert result.succeeded
        assert provider.calls[0].action == "light.turnOff"
        assert provider.calls[0].parameters["target"] == "light.bedroom_ceiling"

    async def testDeviceStateChanges(self):
        provider = SimulatedAutomationProvider()
        tool = HomeAutomationTool(provider)

        await tool.execute(HomeAutomationInput(action="light.turnOn", target="bedroom light"))

        assert provider.stateOf("light.bedroom_ceiling") == "on"

    async def testConfirmationIsSpeakable(self):
        tool = HomeAutomationTool(SimulatedAutomationProvider())

        result = await tool.execute(
            HomeAutomationInput(action="light.turnOff", target="bedroom light")
        )

        assert result.message == "I've turned the bedroom ceiling light off."

    async def testForbiddenActionNeverReachesTheProvider(self):
        provider = SimulatedAutomationProvider()
        tool = HomeAutomationTool(provider, ActionPolicy(("light.turnOff",)))

        with pytest.raises(ToolNotPermittedError):
            await tool.execute(
                HomeAutomationInput(action="lock.unlock", target="front door")
            )

        assert provider.calls == []

    async def testAmbiguousTargetAsksRatherThanGuessing(self):
        provider = SimulatedAutomationProvider()
        tool = HomeAutomationTool(provider)

        result = await tool.execute(
            HomeAutomationInput(action="light.turnOff", target="kitchen light")
        )

        assert not result.succeeded
        assert "Did you mean" in result.message
        assert provider.calls == []

    async def testUnknownTargetIsReported(self):
        tool = HomeAutomationTool(SimulatedAutomationProvider())

        result = await tool.execute(
            HomeAutomationInput(action="light.turnOff", target="garage door")
        )

        assert not result.succeeded
        assert "couldn't find" in result.message

    async def testActingOnAllMatchingDevices(self):
        provider = SimulatedAutomationProvider()
        tool = HomeAutomationTool(provider)

        result = await tool.execute(
            HomeAutomationInput(action="light.turnOff", target="kitchen lights", all=True)
        )

        assert result.succeeded
        assert len(provider.calls) == 2

    async def testDeviceListIsCachedBetweenCalls(self):
        """Fetching devices per command lands straight in perceived latency."""
        provider = SimulatedAutomationProvider()
        tool = HomeAutomationTool(provider)

        first = await tool.devices()
        second = await tool.devices()

        assert first is second


class TestListDevicesTool:
    async def testListsEverything(self):
        from app.tools.homeAutomation import ListDevicesInput

        result = await ListDevicesTool(SimulatedAutomationProvider()).execute(
            ListDevicesInput()
        )

        assert result.succeeded
        assert "Kitchen" in result.message

    async def testFiltersByArea(self):
        from app.tools.homeAutomation import ListDevicesInput

        result = await ListDevicesTool(SimulatedAutomationProvider()).execute(
            ListDevicesInput(area="Bedroom")
        )

        assert "Bedroom" in result.message
        assert "Study" not in result.message

    def testItIsReadOnly(self):
        assert ListDevicesTool(SimulatedAutomationProvider()).readOnly


class TestBuildAutomationTools:
    def testProducesBothTools(self):
        tools = buildAutomationTools(SimulatedAutomationProvider())

        assert {tool.name for tool in tools} == {"homeAutomation", "listDevices"}

    def testAllowedActionsAreApplied(self):
        tools = buildAutomationTools(SimulatedAutomationProvider(), ("light.turnOff",))
        automation = next(tool for tool in tools if tool.name == "homeAutomation")

        assert automation.policy.allowed == ("light.turnOff",)


class TestHomeAssistantTranslation:
    def testCamelCaseVerbBecomesSnakeCaseService(self):
        assert translateAction("light.turnOff") == ("light", "turn_off")

    def testSingleWordVerbIsUnchanged(self):
        assert translateAction("light.toggle") == ("light", "toggle")

    def testMalformedActionIsRejected(self):
        with pytest.raises(Exception, match=r"domain\.verb"):
            translateAction("turnOff")


class TestSimulatedProvider:
    async def testDevicesHaveAreasAndDomains(self):
        devices = await SimulatedAutomationProvider().listDevices()

        assert any(device.area == "Kitchen" for device in devices)
        assert any(device.domain == "switch" for device in devices)

    async def testToggleFlipsState(self):
        provider = SimulatedAutomationProvider()

        await provider.execute("light.toggle", {"target": "light.hallway"})

        assert provider.stateOf("light.hallway") == "off"

    async def testCustomDeviceListIsUsed(self):
        provider = SimulatedAutomationProvider(
            devices=[Device("light.only", "Only light", "light", "Nowhere", "off")]
        )

        assert len(await provider.listDevices()) == 1


class TestRefusalIsLogged:
    """A refusal nobody can see is indistinguishable from a silent failure."""

    async def testRefusedStateChangeIsLogged(self, caplog):
        registry = ToolRegistry(allowStateChanges=False)
        registry.register(MutatingTool())

        with caplog.at_level(logging.WARNING), pytest.raises(ToolNotPermittedError):
            await registry.invoke("mutating", {"value": 1})

        assert "Refused" in caplog.text

    def testRefusedActionIsLogged(self, caplog):
        with caplog.at_level(logging.WARNING), pytest.raises(ToolNotPermittedError):
            ActionPolicy(("light.turnOff",)).check("lock.unlock")

        assert "allow-list" in caplog.text
