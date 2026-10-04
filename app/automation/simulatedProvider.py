"""A simulated house.

Serves two purposes: tests can exercise the whole tool path without a Home
Assistant instance, and the assistant is usable end to end before anyone
configures one. It records every call, so tests can assert on what was asked
rather than only on what came back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from app.automation.automationProvider import (
    AutomationProvider,
    AutomationResult,
    Device,
    DeviceNotFoundError,
)

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_DEVICES = (
    ("light.kitchen_ceiling", "Ceiling light", "light", "Kitchen", "off"),
    ("light.kitchen_under_cabinet", "Under cabinet light", "light", "Kitchen", "off"),
    ("light.lounge_lamp", "Lamp", "light", "Lounge room", "on"),
    ("light.bedroom_ceiling", "Ceiling light", "light", "Bedroom", "off"),
    ("light.hallway", "Hallway light", "light", "Hallway", "on"),
    ("switch.coffee_machine", "Coffee machine", "switch", "Kitchen", "off"),
    ("switch.desk_fan", "Desk fan", "switch", "Study", "off"),
    ("climate.lounge", "Thermostat", "climate", "Lounge room", "21"),
    ("sensor.outside_temperature", "Outside temperature", "sensor", None, "14"),
)


@dataclass(frozen=True, slots=True)
class RecordedCall:
    """One action, as the tool asked for it."""

    action: str
    parameters: dict[str, Any] = field(default_factory=dict)


class SimulatedAutomationProvider(AutomationProvider):
    """An in-memory house that responds to actions."""

    name: ClassVar[str] = "simulated"

    def __init__(self, devices: list[Device] | None = None) -> None:
        self._devices: dict[str, Device] = {}
        for device in devices if devices is not None else _defaultDevices():
            self._devices[device.deviceId] = device
        self.calls: list[RecordedCall] = []

    @classmethod
    def fromSettings(cls, settings: Settings) -> SimulatedAutomationProvider:
        return cls()

    async def listDevices(self) -> list[Device]:
        return list(self._devices.values())

    async def execute(self, action: str, parameters: dict[str, Any]) -> AutomationResult:
        self.calls.append(RecordedCall(action=action, parameters=dict(parameters)))

        target = str(parameters.get("target") or "")
        device = self._devices.get(target)
        if device is None:
            raise DeviceNotFoundError(f"No simulated device with id {target!r}")

        verb = action.partition(".")[2]
        newState = _stateAfter(verb, device.state)

        if newState is not None:
            self._devices[target] = Device(
                deviceId=device.deviceId,
                name=device.name,
                domain=device.domain,
                area=device.area,
                state=newState,
                attributes=device.attributes,
            )

        logger.info("Simulated %s on %s", action, device.describedName)
        return AutomationResult.ok(
            action=action,
            target=target,
            message=f"{device.describedName} is now {newState or 'updated'}",
            state=newState,
            simulated=True,
        )

    async def isAvailable(self) -> bool:
        return True

    def stateOf(self, deviceId: str) -> str | None:
        device = self._devices.get(deviceId)
        return device.state if device else None

    def describe(self) -> str:
        return f"simulated ({len(self._devices)} devices)"


def _defaultDevices() -> list[Device]:
    return [
        Device(deviceId=deviceId, name=name, domain=domain, area=area, state=state)
        for deviceId, name, domain, area, state in DEFAULT_DEVICES
    ]


def _stateAfter(verb: str, current: str | None) -> str | None:
    """What the device's state becomes after a verb."""
    if verb in {"turnOn", "open", "lock"}:
        return "on"
    if verb in {"turnOff", "close", "unlock"}:
        return "off"
    if verb == "toggle":
        return "off" if current == "on" else "on"
    return current
