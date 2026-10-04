"""Wake-word detection.

The brief asks that the assistant not be tied to any particular engine, so this
is an interface with openWakeWord behind it. openWakeWord is Apache-2.0, runs
on onnxruntime, and needs no account or key --- Porcupine is the obvious
alternative but requires both.

Detection is deliberately rate-limited. A wake word spans about a second of
audio, so without a refractory period one utterance fires the detector on every
overlapping frame.
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from app.audio.audioBuffer import AudioBuffer

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

WAKE_SAMPLE_RATE = 16000
# openWakeWord expects 1280 samples at 16 kHz.
OPEN_WAKE_WORD_FRAME_MILLISECONDS = 80


class WakeWordError(Exception):
    """Raised when wake-word detection fails."""


class UnknownWakeWordProviderError(WakeWordError):
    """Raised when configuration names a provider that does not exist."""


@dataclass(frozen=True, slots=True)
class Detection:
    """A wake word that has just been heard."""

    word: str
    confidence: float


class WakeWordProvider(ABC):
    """Watches a frame stream for a wake word."""

    name: ClassVar[str] = "unnamed"

    @classmethod
    @abstractmethod
    def fromSettings(cls, settings: Settings) -> WakeWordProvider:
        """Construct from configuration."""

    @property
    def frameMilliseconds(self) -> int:
        """The frame size this provider requires."""
        return OPEN_WAKE_WORD_FRAME_MILLISECONDS

    @property
    def sampleRate(self) -> int:
        return WAKE_SAMPLE_RATE

    @property
    def alwaysAwake(self) -> bool:
        """Whether the assistant should listen without waiting to be woken."""
        return False

    @abstractmethod
    def detect(self, frame: AudioBuffer) -> Detection | None:
        """Whether this frame completes a wake word."""

    def reset(self) -> None:  # noqa: B027
        """Forget any state carried between frames."""

    async def load(self) -> None:  # noqa: B027 - optional hook
        """Prepare the model."""

    async def unload(self) -> None:  # noqa: B027 - optional hook
        """Release the model."""

    def describe(self) -> str:
        return self.name


class AlwaysAwakeProvider(WakeWordProvider):
    """No wake word: every utterance is treated as addressed to the assistant.

    Appropriate for push-to-talk, for a headset, and for tests. Not for a
    device sitting in a room.
    """

    name: ClassVar[str] = "always-awake"

    @classmethod
    def fromSettings(cls, settings: Settings) -> AlwaysAwakeProvider:
        return cls()

    @property
    def alwaysAwake(self) -> bool:
        return True

    def detect(self, frame: AudioBuffer) -> Detection | None:
        return None

    def describe(self) -> str:
        return "always awake (no wake word)"


class ManualWakeWordProvider(WakeWordProvider):
    """Fires when told to. Intended for tests and for a push-to-talk button."""

    name: ClassVar[str] = "manual"

    def __init__(self, word: str = "manual") -> None:
        self._word = word
        self._armed = False

    @classmethod
    def fromSettings(cls, settings: Settings) -> ManualWakeWordProvider:
        return cls(word=settings.wakeWord.word or "manual")

    def trigger(self) -> None:
        """Wake on the next frame."""
        self._armed = True

    def detect(self, frame: AudioBuffer) -> Detection | None:
        if not self._armed:
            return None
        self._armed = False
        return Detection(word=self._word, confidence=1.0)

    def describe(self) -> str:
        return f"manual ({self._word})"


class OpenWakeWordProvider(WakeWordProvider):
    """openWakeWord, with its pretrained models."""

    name: ClassVar[str] = "openwakeword"

    def __init__(
        self,
        word: str = "hey_jarvis",
        threshold: float = 0.5,
        refractorySeconds: float = 2.0,
    ) -> None:
        self._word = word
        self._threshold = threshold
        self._refractorySeconds = refractorySeconds
        self._model: Any = None
        self._numpy: Any = None
        # Measured in audio time, accumulated from the frames themselves,
        # rather than off the wall clock. Live they are the same; replaying a
        # recording they are not, and a wall-clock refractory period would
        # swallow every detection after the first.
        self._audioSeconds = 0.0
        self._lastDetectionAt = float("-inf")

    @classmethod
    def fromSettings(cls, settings: Settings) -> OpenWakeWordProvider:
        section = settings.wakeWord
        return cls(
            word=section.word or "hey_jarvis",
            threshold=section.threshold,
            refractorySeconds=section.refractorySeconds,
        )

    @property
    def frameMilliseconds(self) -> int:
        return OPEN_WAKE_WORD_FRAME_MILLISECONDS

    async def load(self) -> None:
        if self._model is not None:
            return

        try:
            openwakeword = importlib.import_module("openwakeword")
            modelModule = importlib.import_module("openwakeword.model")
            self._numpy = importlib.import_module("numpy")
        except ImportError as error:
            raise WakeWordError(
                "Wake-word detection requires the 'openwakeword' package.\n"
                'Install with: pip install -e ".[wake]"\n'
                "Or set wakeWord.provider to 'always-awake' to listen continuously."
            ) from error

        # The pretrained models download once, on first use.
        try:
            openwakeword.utils.download_models()
        except Exception as error:  # noqa: BLE001 - already-present models are fine
            logger.debug("Model download step reported: %s", error)

        try:
            self._model = modelModule.Model(
                wakeword_models=[self._word], inference_framework="onnx"
            )
        except Exception as error:
            raise _describeLoadFailure(error, self._word) from error

        logger.info(
            "Wake word %r ready (threshold %.2f)", self._word, self._threshold
        )

    async def unload(self) -> None:
        self._model = None

    def detect(self, frame: AudioBuffer) -> Detection | None:
        if self._model is None:
            raise WakeWordError("Wake word model is not loaded; call load() first")

        samples = self._numpy.frombuffer(frame.data, dtype="<i2")
        self._audioSeconds += frame.durationSeconds

        try:
            scores = self._model.predict(samples)
        except Exception as error:
            raise WakeWordError(f"Wake-word detection failed: {error}") from error

        best, confidence = _bestScore(scores)
        if confidence < self._threshold:
            return None

        # One spoken wake word spans many frames; without this it fires on
        # each of them and the assistant wakes repeatedly.
        if self._audioSeconds - self._lastDetectionAt < self._refractorySeconds:
            return None
        self._lastDetectionAt = self._audioSeconds

        logger.info("Heard wake word %r (%.2f)", best, confidence)
        return Detection(word=best, confidence=confidence)

    def reset(self) -> None:
        if self._model is not None and hasattr(self._model, "reset"):
            self._model.reset()
        self._audioSeconds = 0.0
        self._lastDetectionAt = float("-inf")

    def describe(self) -> str:
        return f"openwakeword [{self._word}] (threshold {self._threshold})"


WAKE_WORD_PROVIDER_REGISTRY: dict[str, str] = {
    "openwakeword": "app.audio.wakeWord:OpenWakeWordProvider",
    "always-awake": "app.audio.wakeWord:AlwaysAwakeProvider",
    "manual": "app.audio.wakeWord:ManualWakeWordProvider",
}


def resolveWakeWordProviderClass(name: str) -> type[WakeWordProvider]:
    """Import and return the provider class registered under ``name``."""
    try:
        target = WAKE_WORD_PROVIDER_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(WAKE_WORD_PROVIDER_REGISTRY))
        raise UnknownWakeWordProviderError(
            f"Unknown wake-word provider {name!r}. Available: {known}"
        ) from None

    moduleName, _, className = target.partition(":")
    module = importlib.import_module(moduleName)
    return getattr(module, className)


def createWakeWordProvider(settings: Settings) -> WakeWordProvider:
    """Build the provider named in configuration."""
    return resolveWakeWordProviderClass(settings.wakeWord.provider).fromSettings(settings)


def _bestScore(scores: Any) -> tuple[str, float]:
    """The highest-scoring wake word in a prediction."""
    if not scores:
        return "", 0.0
    best = max(scores.items(), key=lambda item: item[1])
    return str(best[0]), float(best[1])


def _describeLoadFailure(error: Exception, word: str) -> Exception:
    """Translate opaque load failures into messages that say what to do."""
    text = str(error).lower()

    if "not found" in text or "no such file" in text or "does not exist" in text:
        return WakeWordError(
            f"No wake-word model named {word!r}.\n"
            "The pretrained models include: alexa, hey_jarvis, hey_mycroft, "
            "hey_rhasspy, timer, weather.\n"
            "Set wakeWord.word to one of those, or give a path to a .onnx model."
        )

    return WakeWordError(f"Could not load the wake-word model {word!r}: {error}")


__all__ = [
    "AlwaysAwakeProvider",
    "Detection",
    "ManualWakeWordProvider",
    "OpenWakeWordProvider",
    "WakeWordError",
    "WakeWordProvider",
    "createWakeWordProvider",
    "resolveWakeWordProviderClass",
]
