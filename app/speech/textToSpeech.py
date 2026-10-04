"""Text-to-speech interface and provider registry.

Nothing above this module knows which model produces the audio. Providers are
selected by name from configuration, and their modules are imported only when
selected, so choosing the tone provider does not drag in torch.
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, ClassVar

from app.audio.audioBuffer import AudioBuffer
from app.speech.voices import VoiceLibrary

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


class SpeechSynthesisError(Exception):
    """Raised when speech cannot be generated."""


class ProviderUnavailableError(SpeechSynthesisError):
    """Raised when a provider's dependencies or models are missing."""


class UnknownProviderError(SpeechSynthesisError):
    """Raised when configuration names a provider that does not exist."""


class TextToSpeechProvider(ABC):
    """Turns text into audio.

    Implementations own their model for the lifetime of the object. ``load``
    is separate from construction so that the expensive work is explicit, can
    be awaited, and happens exactly once.
    """

    name: ClassVar[str] = "unnamed"

    @classmethod
    @abstractmethod
    def fromSettings(cls, settings: Settings, voices: VoiceLibrary) -> TextToSpeechProvider:
        """Construct from configuration."""

    @abstractmethod
    async def load(self) -> None:
        """Prepare the model. Calling twice must not reload it."""

    @abstractmethod
    async def synthesise(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
    ) -> AudioBuffer:
        """Generate speech. ``voice`` and ``language`` default to configuration."""

    @property
    def supportsStreaming(self) -> bool:
        """Whether audio is emitted before the whole utterance is generated.

        Sentence-level streaming cannot help a reply that is one sentence, and
        most of what a home assistant says is one sentence.
        """
        return False

    async def synthesiseStream(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
    ) -> AsyncIterator[AudioBuffer]:
        """Generate speech in pieces, as they become available.

        The default yields the whole utterance as one piece, so any provider
        can be driven through the streaming path.
        """
        yield await self.synthesise(text, voice=voice, language=language)

    async def unload(self) -> None:  # noqa: B027
        """Release the model and any device memory it holds."""

    @property
    def isLoaded(self) -> bool:
        return True

    @property
    @abstractmethod
    def sampleRate(self) -> int:
        """Sample rate of generated audio."""

    def availableVoices(self) -> list[str]:
        return []

    def describe(self) -> str:
        return self.name


# Provider name to "module:class". Imported lazily on selection.
PROVIDER_REGISTRY: dict[str, str] = {
    "xtts": "app.speech.models.xttsProvider:XttsProvider",
    "piper": "app.speech.models.piperProvider:PiperProvider",
    "tone": "app.speech.models.toneProvider:ToneProvider",
}


def registerProvider(name: str, target: str) -> None:
    """Register an additional provider as ``"module:class"``."""
    PROVIDER_REGISTRY[name] = target


def resolveProviderClass(name: str) -> type[TextToSpeechProvider]:
    """Import and return the provider class registered under ``name``."""
    try:
        target = PROVIDER_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(PROVIDER_REGISTRY))
        raise UnknownProviderError(
            f"Unknown text-to-speech provider {name!r}. Available: {known}"
        ) from None

    moduleName, _, className = target.partition(":")
    try:
        module = importlib.import_module(moduleName)
    except ImportError as error:
        raise ProviderUnavailableError(
            f"Provider {name!r} could not be imported: {error}"
        ) from error

    providerClass = getattr(module, className)
    if not issubclass(providerClass, TextToSpeechProvider):
        raise UnknownProviderError(
            f"{target} is not a TextToSpeechProvider"
        )
    return providerClass


def createTextToSpeechProvider(
    settings: Settings,
    voices: VoiceLibrary | None = None,
) -> TextToSpeechProvider:
    """Build the provider named in configuration."""
    voices = voices or VoiceLibrary(settings.paths.voices)
    providerClass = resolveProviderClass(settings.textToSpeech.provider)
    logger.debug("Creating text-to-speech provider %s", settings.textToSpeech.provider)
    return providerClass.fromSettings(settings, voices)
