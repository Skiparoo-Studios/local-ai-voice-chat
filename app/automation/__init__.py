"""Home automation providers and device matching."""

from app.automation.automationProvider import (
    AUTOMATION_PROVIDER_REGISTRY,
    AutomationError,
    AutomationProvider,
    AutomationResult,
    AutomationUnavailableError,
    Device,
    DeviceNotFoundError,
    UnknownAutomationProviderError,
    createAutomationProvider,
    registerAutomationProvider,
    resolveAutomationProviderClass,
)
from app.automation.deviceMatcher import MatchResult, matchDevice, matchDevices

__all__ = [
    "AUTOMATION_PROVIDER_REGISTRY",
    "AutomationError",
    "AutomationProvider",
    "AutomationResult",
    "AutomationUnavailableError",
    "Device",
    "DeviceNotFoundError",
    "MatchResult",
    "UnknownAutomationProviderError",
    "createAutomationProvider",
    "matchDevice",
    "matchDevices",
    "registerAutomationProvider",
    "resolveAutomationProviderClass",
]
