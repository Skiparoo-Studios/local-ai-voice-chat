"""Language model interface and provider registry.

The brief is explicit that no part of the application should depend on a
particular language-model SDK, so providers speak HTTP directly and the rest of
the system sees only this interface.

Streaming is first class rather than an extra. Stage 3 measured synthesis at
84% of round-trip latency, and the only way to improve that is to start
speaking the first sentence while the model is still writing the rest.
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

Message = dict[str, str]


class LlmError(Exception):
    """Raised when a language model request fails."""


class LlmUnavailableError(LlmError):
    """Raised when the model or its service cannot be reached."""


class UnknownLlmProviderError(LlmError):
    """Raised when configuration names a provider that does not exist."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool the model proposes running.

    Proposed, not decided: the arguments are untrusted model output and are
    validated against the tool's schema before anything happens.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LlmResponse:
    """A complete generation."""

    text: str
    model: str = ""
    promptTokens: int = 0
    completionTokens: int = 0
    durationSeconds: float = 0.0
    toolCalls: tuple[ToolCall, ...] = ()

    @property
    def isEmpty(self) -> bool:
        return not self.text.strip()

    @property
    def hasToolCalls(self) -> bool:
        return bool(self.toolCalls)

    @property
    def tokensPerSecond(self) -> float:
        if self.durationSeconds <= 0 or not self.completionTokens:
            return 0.0
        return self.completionTokens / self.durationSeconds

    def describe(self) -> str:
        rate = f", {self.tokensPerSecond:.1f} tok/s" if self.tokensPerSecond else ""
        return f"{self.completionTokens} tokens in {self.durationSeconds:.2f}s{rate}"


class LlmProvider(ABC):
    """Generates text from a conversation.

    Implementations own their connection for the lifetime of the object.
    """

    name: ClassVar[str] = "unnamed"

    @classmethod
    @abstractmethod
    def fromSettings(cls, settings: Settings) -> LlmProvider:
        """Construct from configuration."""

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> LlmResponse:
        """Generate a complete reply.

        When ``tools`` is given the model may propose calling one instead of
        answering. Streaming and tool selection do not combine cleanly across
        services, so tool-using turns take this path.
        """

    @abstractmethod
    def generateStream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Generate a reply as it arrives, yielding text fragments."""

    async def load(self) -> None:  # noqa: B027 - optional hook
        """Open connections or warm the model."""

    async def unload(self) -> None:  # noqa: B027 - optional hook
        """Close connections."""

    async def isAvailable(self) -> bool:
        """Whether the service can be reached right now."""
        return True

    async def listModels(self) -> list[str]:
        """Models the service offers, where it can say."""
        return []

    def describe(self) -> str:
        return self.name


# Provider name to "module:class". Imported lazily on selection.
LLM_PROVIDER_REGISTRY: dict[str, str] = {
    "ollama": "app.intelligence.providers.ollamaProvider:OllamaProvider",
    "openai-compatible": (
        "app.intelligence.providers.openAiCompatibleProvider:OpenAiCompatibleProvider"
    ),
    "scripted": "app.intelligence.providers.scriptedLlmProvider:ScriptedLlmProvider",
}


def registerLlmProvider(name: str, target: str) -> None:
    """Register an additional provider as ``"module:class"``."""
    LLM_PROVIDER_REGISTRY[name] = target


def resolveLlmProviderClass(name: str) -> type[LlmProvider]:
    """Import and return the provider class registered under ``name``."""
    try:
        target = LLM_PROVIDER_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(LLM_PROVIDER_REGISTRY))
        raise UnknownLlmProviderError(
            f"Unknown language model provider {name!r}. Available: {known}"
        ) from None

    moduleName, _, className = target.partition(":")
    try:
        module = importlib.import_module(moduleName)
    except ImportError as error:
        raise LlmUnavailableError(
            f"Provider {name!r} could not be imported: {error}"
        ) from error

    providerClass = getattr(module, className)
    if not issubclass(providerClass, LlmProvider):
        raise UnknownLlmProviderError(f"{target} is not an LlmProvider")
    return providerClass


def createLlmProvider(settings: Settings) -> LlmProvider:
    """Build the provider named in configuration."""
    providerClass = resolveLlmProviderClass(settings.llm.provider)
    logger.debug("Creating language model provider %s", settings.llm.provider)
    return providerClass.fromSettings(settings)
