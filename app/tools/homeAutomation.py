"""The home automation tool.

Sits between the assistant and the automation layer: it resolves a spoken
target to a device, checks the action against the allow-list, and hands a
generic action to whichever provider is configured.

The allow-list lives here rather than in the tool registry because it is
expressed in automation vocabulary. Nothing above this module needs to know
that ``light.turnOff`` exists.
"""

from __future__ import annotations

import fnmatch
import logging
from typing import ClassVar

from pydantic import BaseModel, Field

from app.automation.automationProvider import (
    AutomationError,
    AutomationProvider,
    Device,
)
from app.automation.deviceMatcher import matchDevice, matchDevices
from app.tools.base import Tool, ToolNotPermittedError, ToolResult

logger = logging.getLogger(__name__)

# Actions permitted unless configuration says otherwise. Deliberately covers
# lights, switches and fans only: locks and covers are the ones where a
# mistaken call has consequences worth an explicit decision.
DEFAULT_ALLOWED_ACTIONS = (
    "light.turnOn",
    "light.turnOff",
    "light.toggle",
    "switch.turnOn",
    "switch.turnOff",
    "switch.toggle",
    "fan.turnOn",
    "fan.turnOff",
)

# Actions that should never run without deliberate configuration, listed so the
# refusal message can explain itself.
SENSITIVE_ACTIONS = frozenset({"lock.unlock", "cover.openCover", "alarm.disarm"})


class HomeAutomationInput(BaseModel):
    """Arguments for the home automation tool."""

    action: str = Field(
        description=(
            "The action to perform, as domain.verb. For example light.turnOn, "
            "light.turnOff, light.toggle, switch.turnOn, switch.turnOff."
        )
    )
    target: str = Field(
        description=(
            "Which device or room to act on, in plain words. "
            "For example 'kitchen light' or 'lounge lamp'."
        )
    )
    all: bool = Field(
        default=False,
        description="Act on every matching device rather than just the closest match.",
    )


class ActionPolicy:
    """Decides which automation actions may run.

    Patterns support wildcards, so ``light.*`` permits every light action while
    leaving locks and covers out.
    """

    def __init__(self, allowed: tuple[str, ...] = DEFAULT_ALLOWED_ACTIONS) -> None:
        self._allowed = tuple(allowed)

    @property
    def allowed(self) -> tuple[str, ...]:
        return self._allowed

    def permits(self, action: str) -> bool:
        return any(fnmatch.fnmatch(action, pattern) for pattern in self._allowed)

    def check(self, action: str) -> None:
        """Raise unless ``action`` is permitted."""
        if self.permits(action):
            return

        logger.warning("Refused automation action %r: not on the allow-list", action)
        if action in SENSITIVE_ACTIONS:
            raise ToolNotPermittedError(
                f"{action} is not something I'm allowed to do. "
                "It has to be added to automation.allowedActions deliberately."
            )
        raise ToolNotPermittedError(
            f"{action} is not on the list of permitted actions. "
            f"Permitted: {', '.join(self._allowed)}"
        )


class HomeAutomationTool(Tool):
    """Controls devices through the configured automation provider."""

    name: ClassVar[str] = "homeAutomation"
    description: ClassVar[str] = (
        "Control lights, switches and other devices in the home. "
        "Use this whenever the user asks to turn something on or off."
    )
    inputModel: ClassVar[type[BaseModel]] = HomeAutomationInput
    readOnly: ClassVar[bool] = False

    def __init__(
        self,
        provider: AutomationProvider,
        policy: ActionPolicy | None = None,
    ) -> None:
        self._provider = provider
        self._policy = policy or ActionPolicy()
        self._devices: list[Device] | None = None

    @property
    def provider(self) -> AutomationProvider:
        return self._provider

    @property
    def policy(self) -> ActionPolicy:
        return self._policy

    async def devices(self, *, refresh: bool = False) -> list[Device]:
        """The device list, fetched once and cached.

        Device lists change rarely and fetching costs a round trip on every
        command, which lands directly in the latency the user feels.
        """
        if self._devices is None or refresh:
            self._devices = await self._provider.listDevices()
        return self._devices

    async def execute(self, parameters: BaseModel) -> ToolResult:
        assert isinstance(parameters, HomeAutomationInput)

        # Checked before anything is resolved, so a forbidden action cannot
        # even learn which devices exist.
        self._policy.check(parameters.action)

        devices = await self.devices()
        if not devices:
            return ToolResult.failed("I couldn't find any devices to control.")

        if parameters.all:
            return await self._executeOnAll(parameters, devices)
        return await self._executeOnOne(parameters, devices)

    async def _executeOnOne(
        self, parameters: HomeAutomationInput, devices: list[Device]
    ) -> ToolResult:
        match = matchDevice(parameters.target, devices)

        if match.ambiguous:
            names = " or the ".join(device.describedName for device in match.alternatives)
            return ToolResult.failed(
                f"Did you mean the {names}?", ambiguous=True, target=parameters.target
            )

        if not match.found or match.best is None:
            return ToolResult.failed(
                f"I couldn't find anything called {parameters.target}.",
                target=parameters.target,
            )

        return await self._run(parameters.action, match.best)

    async def _executeOnAll(
        self, parameters: HomeAutomationInput, devices: list[Device]
    ) -> ToolResult:
        matched = matchDevices(parameters.target, devices)
        if not matched:
            return ToolResult.failed(
                f"I couldn't find anything called {parameters.target}.",
                target=parameters.target,
            )

        failures: list[str] = []
        for device in matched:
            result = await self._run(parameters.action, device)
            if not result.succeeded:
                failures.append(device.describedName)

        if failures:
            return ToolResult.failed(
                f"I couldn't reach {', '.join(failures)}.", failed=failures
            )
        return ToolResult.ok(
            _confirmation(parameters.action, f"{len(matched)} devices"),
            count=len(matched),
        )

    async def _run(self, action: str, device: Device) -> ToolResult:
        try:
            result = await self._provider.execute(action, {"target": device.deviceId})
        except AutomationError as error:
            logger.warning("Automation failed for %s: %s", device.deviceId, error)
            return ToolResult.failed(f"I couldn't do that: {error}")

        if not result.succeeded:
            return ToolResult.failed(result.message or "That didn't work.")

        # The device list caches state, which this call has just invalidated.
        self._devices = None

        return ToolResult.ok(
            _confirmation(action, f"the {device.describedName.lower()}"),
            deviceId=device.deviceId,
            action=action,
        )

    def describe(self) -> str:
        return f"{self.name} via {self._provider.describe()}"


class ListDevicesInput(BaseModel):
    """Arguments for listing devices."""

    area: str = Field(default="", description="Only devices in this room, if given.")


class ListDevicesTool(Tool):
    """Reports what can be controlled. Read-only, so unrestricted."""

    name: ClassVar[str] = "listDevices"
    description: ClassVar[str] = "List the devices in the home that can be controlled."
    inputModel: ClassVar[type[BaseModel]] = ListDevicesInput
    readOnly: ClassVar[bool] = True

    def __init__(self, provider: AutomationProvider) -> None:
        self._provider = provider

    async def execute(self, parameters: BaseModel) -> ToolResult:
        assert isinstance(parameters, ListDevicesInput)

        devices = await self._provider.listDevices()
        if parameters.area:
            wanted = parameters.area.strip().lower()
            devices = [
                device for device in devices if (device.area or "").lower() == wanted
            ]

        if not devices:
            return ToolResult.failed("I couldn't find any devices.")

        names = ", ".join(device.describedName for device in devices[:10])
        more = f" and {len(devices) - 10} more" if len(devices) > 10 else ""
        return ToolResult.ok(
            f"I can control {names}{more}.",
            deviceIds=[device.deviceId for device in devices],
        )


def _confirmation(action: str, target: str) -> str:
    """A short spoken confirmation for an action."""
    verb = action.partition(".")[2]
    phrases = {
        "turnOn": f"I've turned {target} on.",
        "turnOff": f"I've turned {target} off.",
        "toggle": f"I've toggled {target}.",
        "open": f"I've opened {target}.",
        "close": f"I've closed {target}.",
        "lock": f"I've locked {target}.",
        "unlock": f"I've unlocked {target}.",
    }
    return phrases.get(verb, f"Done: {action} on {target}.")


def buildAutomationTools(
    provider: AutomationProvider,
    allowedActions: tuple[str, ...] = DEFAULT_ALLOWED_ACTIONS,
) -> list[Tool]:
    """The automation tools, ready to register."""
    return [
        HomeAutomationTool(provider, ActionPolicy(allowedActions)),
        ListDevicesTool(provider),
    ]


__all__ = [
    "ActionPolicy",
    "HomeAutomationInput",
    "HomeAutomationTool",
    "ListDevicesTool",
    "buildAutomationTools",
]
