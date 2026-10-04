"""Speech-to-text interface and provider registry.

Mirrors the text-to-speech arrangement: providers are named in configuration
and imported only when selected, so choosing the scripted provider does not
drag in Whisper.
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from app.audio.audioBuffer import AudioBuffer

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Raised when audio cannot be transcribed."""


class SttProviderUnavailableError(TranscriptionError):
    """Raised when a provider's dependencies or models are missing."""


class UnknownSttProviderError(TranscriptionError):
    """Raised when configuration names a provider that does not exist."""


@dataclass(frozen=True, slots=True)
class Transcript:
    """The result of transcribing one utterance.

    Richer than a bare string because the assistant needs the detected language
    for replies, and the timings for the latency budget this project tracks.
    """

    text: str
    language: str | None = None
    languageConfidence: float = 0.0
    audioSeconds: float = 0.0
    transcriptionSeconds: float = 0.0
    segments: tuple[str, ...] = field(default_factory=tuple)

    @property
    def isEmpty(self) -> bool:
        return not self.text.strip()

    @property
    def realTimeFactor(self) -> float:
        """Transcription time per second of audio. Below 1.0 beats real time."""
        if self.audioSeconds <= 0:
            return 0.0
        return self.transcriptionSeconds / self.audioSeconds

    def describe(self) -> str:
        return (
            f"{self.transcriptionSeconds:.2f}s for {self.audioSeconds:.2f}s of audio "
            f"(RTF {self.realTimeFactor:.2f})"
        )


class SpeechToTextProvider(ABC):
    """Turns audio into text.

    Implementations own their model for the lifetime of the object, so that
    ``load`` happens once rather than per request.
    """

    name: ClassVar[str] = "unnamed"

    @classmethod
    @abstractmethod
    def fromSettings(cls, settings: Settings) -> SpeechToTextProvider:
        """Construct from configuration."""

    @abstractmethod
    async def load(self) -> None:
        """Prepare the model. Calling twice must not reload it."""

    @abstractmethod
    async def transcribe(
        self,
        audio: AudioBuffer,
        *,
        language: str | None = None,
    ) -> Transcript:
        """Transcribe one utterance. ``language`` defaults to configuration."""

    async def unload(self) -> None:  # noqa: B027 - optional hook
        """Release the model and any device memory it holds."""

    @property
    def isLoaded(self) -> bool:
        return True

    @property
    def requiredSampleRate(self) -> int:
        """Sample rate this provider expects its input at."""
        return 16000

    def describe(self) -> str:
        return self.name


# Provider name to "module:class". Imported lazily on selection.
STT_PROVIDER_REGISTRY: dict[str, str] = {
    "faster-whisper": "app.speech.models.fasterWhisperProvider:FasterWhisperProvider",
    "scripted": "app.speech.models.scriptedProvider:ScriptedProvider",
}


def registerSttProvider(name: str, target: str) -> None:
    """Register an additional provider as ``"module:class"``."""
    STT_PROVIDER_REGISTRY[name] = target


def resolveSttProviderClass(name: str) -> type[SpeechToTextProvider]:
    """Import and return the provider class registered under ``name``."""
    try:
        target = STT_PROVIDER_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(STT_PROVIDER_REGISTRY))
        raise UnknownSttProviderError(
            f"Unknown speech-to-text provider {name!r}. Available: {known}"
        ) from None

    moduleName, _, className = target.partition(":")
    try:
        module = importlib.import_module(moduleName)
    except ImportError as error:
        raise SttProviderUnavailableError(
            f"Provider {name!r} could not be imported: {error}"
        ) from error

    providerClass = getattr(module, className)
    if not issubclass(providerClass, SpeechToTextProvider):
        raise UnknownSttProviderError(f"{target} is not a SpeechToTextProvider")
    return providerClass


def createSpeechToTextProvider(settings: Settings) -> SpeechToTextProvider:
    """Build the provider named in configuration."""
    providerClass = resolveSttProviderClass(settings.speechToText.provider)
    logger.debug("Creating speech-to-text provider %s", settings.speechToText.provider)
    return providerClass.fromSettings(settings)
