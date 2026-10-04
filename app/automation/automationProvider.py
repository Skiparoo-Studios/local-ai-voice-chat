"""Home automation interface and provider registry.

Per §10 of the brief, assistant logic must not contain Home Assistant-specific
behaviour. Actions cross this boundary in a generic form --- ``light.turnOff``
with a target --- and each provider translates that into whatever its platform
requires.
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


class AutomationError(Exception):
    """Raised when an automation request fails."""


class AutomationUnavailableError(AutomationError):
    """Raised when the automation platform cannot be reached."""


class UnknownAutomationProviderError(AutomationError):
    """Raised when configuration names a provider that does not exist."""


class DeviceNotFoundError(AutomationError):
    """Raised when no device matches what was asked for."""


@dataclass(frozen=True, slots=True)
class Device:
    """Something the assistant can act on."""

    deviceId: str
    name: str
    domain: str
    area: str | None = None
    state: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def describedName(self) -> str:
        """How a person would refer to this device.

        The area is only prefixed when the name does not already carry it.
        Home Assistant names are often already qualified, and "the hallway
        hallway light" is how that reads aloud otherwise.
        """
        if not self.area:
            return self.name
        if self.area.lower() in self.name.lower():
            return self.name
        return f"{self.area} {self.name}"

    def describe(self) -> str:
        state = f" ({self.state})" if self.state else ""
        return f"{self.describedName}{state} [{self.deviceId}]"


@dataclass(frozen=True, slots=True)
class AutomationResult:
    """The outcome of one action."""

    succeeded: bool
    action: str
    target: str = ""
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, action: str, target: str, message: str = "", **details: Any) -> AutomationResult:
        return cls(
            succeeded=True, action=action, target=target, message=message, details=details
        )

    @classmethod
    def failed(cls, action: str, target: str, message: str) -> AutomationResult:
        return cls(succeeded=False, action=action, target=target, message=message)


class AutomationProvider(ABC):
    """Executes generic actions against a home automation platform."""

    name: ClassVar[str] = "unnamed"

    @classmethod
    @abstractmethod
    def fromSettings(cls, settings: Settings) -> AutomationProvider:
        """Construct from configuration."""

    @abstractmethod
    async def listDevices(self) -> list[Device]:
        """Every device the platform exposes."""

    @abstractmethod
    async def execute(self, action: str, parameters: dict[str, Any]) -> AutomationResult:
        """Perform ``action``. ``parameters`` must include a resolved target."""

    async def load(self) -> None:  # noqa: B027 - optional hook
        """Open connections and prime any device cache."""

    async def unload(self) -> None:  # noqa: B027 - optional hook
        """Close connections."""

    async def isAvailable(self) -> bool:
        """Whether the platform can be reached right now."""
        return True

    def describe(self) -> str:
        return self.name


# Provider name to "module:class". Imported lazily on selection.
AUTOMATION_PROVIDER_REGISTRY: dict[str, str] = {
    "home-assistant": "app.automation.homeAssistant:HomeAssistantProvider",
    "simulated": "app.automation.simulatedProvider:SimulatedAutomationProvider",
}


def registerAutomationProvider(name: str, target: str) -> None:
    """Register an additional provider as ``"module:class"``."""
    AUTOMATION_PROVIDER_REGISTRY[name] = target


def resolveAutomationProviderClass(name: str) -> type[AutomationProvider]:
    """Import and return the provider class registered under ``name``."""
    try:
        target = AUTOMATION_PROVIDER_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(AUTOMATION_PROVIDER_REGISTRY))
        raise UnknownAutomationProviderError(
            f"Unknown automation provider {name!r}. Available: {known}"
        ) from None

    moduleName, _, className = target.partition(":")
    try:
        module = importlib.import_module(moduleName)
    except ImportError as error:
        raise AutomationUnavailableError(
            f"Provider {name!r} could not be imported: {error}"
        ) from error

    providerClass = getattr(module, className)
    if not issubclass(providerClass, AutomationProvider):
        raise UnknownAutomationProviderError(f"{target} is not an AutomationProvider")
    return providerClass


def createAutomationProvider(settings: Settings) -> AutomationProvider:
    """Build the provider named in configuration."""
    providerClass = resolveAutomationProviderClass(settings.automation.provider)
    logger.debug("Creating automation provider %s", settings.automation.provider)
    return providerClass.fromSettings(settings)
