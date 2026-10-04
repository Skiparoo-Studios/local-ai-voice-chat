"""Voice activity detection, and turning frames into utterances.

Two things live here. A :class:`VadProvider` answers "is this frame speech?",
and a :class:`SpeechSegmenter` uses those answers to decide where an utterance
starts and ends.

The segmenter is the part that matters for how the assistant feels. Ending on
silence rather than a timer is what makes it possible to just stop talking; a
pre-roll buffer is what stops the first word being clipped, because speech is
always recognised a fraction of a second after it has begun.
"""

from __future__ import annotations

import importlib
import logging
import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar

from app.audio.audioBuffer import AudioBuffer

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

VAD_SAMPLE_RATE = 16000


class VadError(Exception):
    """Raised when voice activity detection fails."""


class UnknownVadProviderError(VadError):
    """Raised when configuration names a provider that does not exist."""


class VadProvider(ABC):
    """Decides whether a frame contains speech."""

    name: ClassVar[str] = "unnamed"

    @classmethod
    @abstractmethod
    def fromSettings(cls, settings: Settings) -> VadProvider:
        """Construct from configuration."""

    @property
    def frameMilliseconds(self) -> int:
        """The frame size this provider requires."""
        return 32

    @property
    def sampleRate(self) -> int:
        return VAD_SAMPLE_RATE

    @abstractmethod
    def isSpeech(self, frame: AudioBuffer) -> bool:
        """Whether this frame contains speech."""

    def reset(self) -> None:  # noqa: B027
        """Forget any state carried between frames."""

    async def load(self) -> None:  # noqa: B027 - optional hook
        """Prepare the model."""

    async def unload(self) -> None:  # noqa: B027 - optional hook
        """Release the model."""

    def describe(self) -> str:
        return self.name


class EnergyVadProvider(VadProvider):
    """Root-mean-square loudness against a threshold.

    Crude, but dependency-free and entirely predictable, which makes it the
    right thing for tests and for hardware too small for a model. It cannot
    tell speech from a slammed door, so Silero is the default.
    """

    name: ClassVar[str] = "energy"

    def __init__(self, threshold: float = 0.02) -> None:
        self._threshold = threshold

    @classmethod
    def fromSettings(cls, settings: Settings) -> EnergyVadProvider:
        return cls(threshold=settings.vad.energyThreshold)

    def isSpeech(self, frame: AudioBuffer) -> bool:
        return rootMeanSquare(frame) >= self._threshold

    def describe(self) -> str:
        return f"energy (threshold {self._threshold})"


class SileroVadProvider(VadProvider):
    """Silero VAD, which distinguishes speech from other sound.

    The model requires exactly 512 samples at 16 kHz per call, so the frame
    size is fixed rather than configurable.
    """

    name: ClassVar[str] = "silero"
    REQUIRED_SAMPLES: ClassVar[int] = 512

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold
        self._model: Any = None
        self._torch: Any = None

    @classmethod
    def fromSettings(cls, settings: Settings) -> SileroVadProvider:
        return cls(threshold=settings.vad.threshold)

    @property
    def frameMilliseconds(self) -> int:
        # 512 samples at 16 kHz.
        return 32

    async def load(self) -> None:
        if self._model is not None:
            return

        try:
            torch = importlib.import_module("torch")
            silero = importlib.import_module("silero_vad")
        except ImportError as error:
            raise VadError(
                "Silero voice activity detection requires the 'silero-vad' package.\n"
                'Install with: pip install -e ".[wake]"\n'
                "Or set vad.provider to 'energy' for the dependency-free fallback."
            ) from error

        self._torch = torch
        self._model = silero.load_silero_vad(onnx=False)
        logger.info("Silero VAD loaded (threshold %.2f)", self._threshold)

    async def unload(self) -> None:
        self._model = None

    def isSpeech(self, frame: AudioBuffer) -> bool:
        if self._model is None:
            raise VadError("Silero VAD is not loaded; call load() first")

        samples = frame.toFloatSamples()
        if len(samples) != self.REQUIRED_SAMPLES:
            raise VadError(
                f"Silero VAD needs exactly {self.REQUIRED_SAMPLES} samples per frame, "
                f"got {len(samples)}. Feed it 32 ms frames at {VAD_SAMPLE_RATE} Hz."
            )

        tensor = self._torch.tensor(samples, dtype=self._torch.float32)
        with self._torch.no_grad():
            probability = float(self._model(tensor, VAD_SAMPLE_RATE).item())
        return probability >= self._threshold

    def reset(self) -> None:
        if self._model is not None and hasattr(self._model, "reset_states"):
            self._model.reset_states()

    def describe(self) -> str:
        return f"silero (threshold {self._threshold})"


VAD_PROVIDER_REGISTRY: dict[str, str] = {
    "silero": "app.audio.vad:SileroVadProvider",
    "energy": "app.audio.vad:EnergyVadProvider",
}


def resolveVadProviderClass(name: str) -> type[VadProvider]:
    """Import and return the provider class registered under ``name``."""
    try:
        target = VAD_PROVIDER_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(VAD_PROVIDER_REGISTRY))
        raise UnknownVadProviderError(
            f"Unknown voice activity provider {name!r}. Available: {known}"
        ) from None

    moduleName, _, className = target.partition(":")
    module = importlib.import_module(moduleName)
    return getattr(module, className)


def createVadProvider(settings: Settings) -> VadProvider:
    """Build the provider named in configuration."""
    return resolveVadProviderClass(settings.vad.provider).fromSettings(settings)


def rootMeanSquare(frame: AudioBuffer) -> float:
    """Loudness of a frame, between 0 and 1."""
    samples = frame.toFloatSamples()
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


# --- Segmentation --------------------------------------------------------


class SegmentState(StrEnum):
    """Where the segmenter is in an utterance."""

    waiting = "waiting"
    speaking = "speaking"
    complete = "complete"


@dataclass(slots=True)
class SegmentResult:
    """What the segmenter concluded after a frame."""

    state: SegmentState
    utterance: AudioBuffer | None = None
    reason: str = ""

    @property
    def isComplete(self) -> bool:
        return self.state is SegmentState.complete


@dataclass(slots=True)
class SegmenterSettings:
    """How an utterance is delimited."""

    # Consecutive speech frames before capture starts. More than one, so a
    # single noisy frame does not begin an utterance.
    startFrames: int = 2
    # Silence before the utterance is considered finished. About 0.8 seconds,
    # which is long enough to pause mid-sentence without being cut off.
    silenceFrames: int = 25
    # Audio kept from before speech was detected, so the first word survives.
    prerollFrames: int = 10
    maximumSeconds: float = 15.0
    minimumSeconds: float = 0.3


class SpeechSegmenter:
    """Assembles frames into one utterance, ending on silence."""

    def __init__(self, settings: SegmenterSettings | None = None) -> None:
        self._settings = settings or SegmenterSettings()
        self._preroll: deque[AudioBuffer] = deque(maxlen=self._settings.prerollFrames)
        self._collected: list[AudioBuffer] = []
        self._speechRun = 0
        self._silenceRun = 0
        self._state = SegmentState.waiting

    @property
    def state(self) -> SegmentState:
        return self._state

    @property
    def capturedSeconds(self) -> float:
        return sum(frame.durationSeconds for frame in self._collected)

    def reset(self) -> None:
        self._preroll.clear()
        self._collected.clear()
        self._speechRun = 0
        self._silenceRun = 0
        self._state = SegmentState.waiting

    def feed(self, frame: AudioBuffer, isSpeech: bool) -> SegmentResult:
        """Offer one frame, and say whether the utterance is now complete."""
        if self._state is SegmentState.waiting:
            return self._whileWaiting(frame, isSpeech)
        return self._whileSpeaking(frame, isSpeech)

    def _whileWaiting(self, frame: AudioBuffer, isSpeech: bool) -> SegmentResult:
        self._preroll.append(frame)

        if not isSpeech:
            self._speechRun = 0
            return SegmentResult(state=SegmentState.waiting)

        self._speechRun += 1
        if self._speechRun < self._settings.startFrames:
            return SegmentResult(state=SegmentState.waiting)

        # Speech began before it was recognised, so the buffered frames go in.
        self._collected = list(self._preroll)
        self._preroll.clear()
        self._silenceRun = 0
        self._state = SegmentState.speaking
        return SegmentResult(state=SegmentState.speaking)

    def _whileSpeaking(self, frame: AudioBuffer, isSpeech: bool) -> SegmentResult:
        self._collected.append(frame)

        if self.capturedSeconds >= self._settings.maximumSeconds:
            return self._finish("reached the maximum length")

        if isSpeech:
            self._silenceRun = 0
            return SegmentResult(state=SegmentState.speaking)

        self._silenceRun += 1
        if self._silenceRun >= self._settings.silenceFrames:
            return self._finish("silence")

        return SegmentResult(state=SegmentState.speaking)

    def _finish(self, reason: str) -> SegmentResult:
        utterance = self._join()
        self._state = SegmentState.complete

        if utterance.durationSeconds < self._settings.minimumSeconds:
            logger.debug("Discarding %.2fs utterance as too short", utterance.durationSeconds)
            return SegmentResult(
                state=SegmentState.complete, utterance=None, reason="too short"
            )

        return SegmentResult(state=SegmentState.complete, utterance=utterance, reason=reason)

    def _join(self) -> AudioBuffer:
        if not self._collected:
            return AudioBuffer(data=b"", sampleRate=VAD_SAMPLE_RATE)
        first = self._collected[0]
        return AudioBuffer(
            data=b"".join(frame.data for frame in self._collected),
            sampleRate=first.sampleRate,
            channels=first.channels,
        )


__all__ = [
    "EnergyVadProvider",
    "SegmentResult",
    "SegmentState",
    "SegmenterSettings",
    "SileroVadProvider",
    "SpeechSegmenter",
    "VadError",
    "VadProvider",
    "createVadProvider",
    "resolveVadProviderClass",
    "rootMeanSquare",
]
