"""Speech recognition with faster-whisper.

Whisper- and CTranslate2-specific handling is confined here: model download
location, compute-type selection, CUDA library discovery on Windows, and the
built-in voice-activity filter that stops Whisper inventing text during
silence.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from app.audio.audioBuffer import AudioBuffer
from app.speech.device import describeCudaDevice, resolveDevice
from app.speech.speechToText import (
    SpeechToTextProvider,
    SttProviderUnavailableError,
    Transcript,
    TranscriptionError,
)

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000

# CTranslate2 supports these directly; "auto" is resolved per device below.
COMPUTE_TYPE_FOR_DEVICE = {"cuda": "float16", "cpu": "int8"}


class FasterWhisperProvider(SpeechToTextProvider):
    """faster-whisper, loaded once and kept resident."""

    name: ClassVar[str] = "faster-whisper"

    def __init__(
        self,
        *,
        modelSize: str = "small",
        device: str = "auto",
        computeType: str = "auto",
        language: str | None = "en",
        beamSize: int = 5,
        vadFilter: bool = True,
        modelsDirectory: Path | None = None,
    ) -> None:
        self._modelSize = modelSize
        self._devicePreference = device
        self._computeTypePreference = computeType
        self._language = language
        self._beamSize = beamSize
        self._vadFilter = vadFilter
        self._modelsDirectory = modelsDirectory

        self._model: Any = None
        self._device: str | None = None
        self._computeType: str | None = None

    @classmethod
    def fromSettings(cls, settings: Settings) -> FasterWhisperProvider:
        section = settings.speechToText
        return cls(
            modelSize=section.model,
            device=section.device,
            computeType=section.computeType,
            language=section.language or None,
            beamSize=section.beamSize,
            vadFilter=section.vadFilter,
            modelsDirectory=settings.paths.models,
        )

    # --- Lifecycle -------------------------------------------------------

    @property
    def isLoaded(self) -> bool:
        return self._model is not None

    @property
    def requiredSampleRate(self) -> int:
        return WHISPER_SAMPLE_RATE

    async def load(self) -> None:
        if self.isLoaded:
            logger.debug("Whisper already loaded, not reloading")
            return

        started = time.perf_counter()
        await asyncio.to_thread(self._loadBlocking)
        logger.info(
            "Whisper %s loaded on %s (%s) in %.1fs",
            self._modelSize,
            self._device,
            self._computeType,
            time.perf_counter() - started,
        )

    def _loadBlocking(self) -> None:
        self._device = resolveDevice(self._devicePreference)
        self._computeType = self._resolveComputeType(self._device)

        if self._device == "cuda":
            _makeCudaLibrariesDiscoverable()
            description = describeCudaDevice()
            if description:
                logger.info("Using CUDA device: %s", description)

        model = _importWhisperModel()

        downloadRoot = None
        if self._modelsDirectory is not None:
            downloadRoot = str((self._modelsDirectory / "whisper").resolve())
            os.makedirs(downloadRoot, exist_ok=True)

        try:
            self._model = model(
                self._modelSize,
                device=self._device,
                compute_type=self._computeType,
                download_root=downloadRoot,
            )
        except Exception as error:
            raise _describeLoadFailure(error, self._modelSize, self._device) from error

    def _resolveComputeType(self, device: str) -> str:
        if self._computeTypePreference not in ("auto", "default", ""):
            return self._computeTypePreference
        return COMPUTE_TYPE_FOR_DEVICE.get(device, "int8")

    async def unload(self) -> None:
        if not self.isLoaded:
            return
        await asyncio.to_thread(self._unloadBlocking)
        logger.info("Whisper unloaded")

    def _unloadBlocking(self) -> None:
        self._model = None
        gc.collect()

    # --- Transcription ---------------------------------------------------

    async def transcribe(
        self,
        audio: AudioBuffer,
        *,
        language: str | None = None,
    ) -> Transcript:
        if not self.isLoaded:
            raise TranscriptionError("Whisper is not loaded; call load() first")

        if audio.sampleRate != WHISPER_SAMPLE_RATE:
            raise TranscriptionError(
                f"Whisper expects {WHISPER_SAMPLE_RATE} Hz audio, got {audio.sampleRate} Hz. "
                "Resample before transcribing (app.audio.input.resampleBuffer does this)."
            )

        if not audio.data:
            return Transcript(text="", language=language or self._language)

        return await asyncio.to_thread(
            self._transcribeBlocking, audio, language or self._language
        )

    def _transcribeBlocking(self, audio: AudioBuffer, language: str | None) -> Transcript:
        samples = _toFloatArray(audio)

        started = time.perf_counter()
        try:
            segments, info = self._model.transcribe(
                samples,
                language=language,
                beam_size=self._beamSize,
                # Whisper invents plausible text during silence; the filter
                # drops non-speech regions before they reach the decoder.
                vad_filter=self._vadFilter,
            )
            # Segments are produced lazily, so the work happens on iteration.
            texts = tuple(segment.text.strip() for segment in segments)
        except Exception as error:
            raise TranscriptionError(f"Whisper failed to transcribe: {error}") from error

        elapsed = time.perf_counter() - started

        return Transcript(
            text=" ".join(text for text in texts if text),
            language=getattr(info, "language", language),
            languageConfidence=float(getattr(info, "language_probability", 0.0) or 0.0),
            audioSeconds=audio.durationSeconds,
            transcriptionSeconds=elapsed,
            segments=texts,
        )

    def describe(self) -> str:
        device = self._device or f"{self._devicePreference} (unresolved)"
        computeType = self._computeType or self._computeTypePreference
        return f"faster-whisper [{self._modelSize}] on {device} ({computeType})"


# --- Module helpers ------------------------------------------------------


def _toFloatArray(audio: AudioBuffer) -> Any:
    """Convert 16-bit PCM to the float32 array CTranslate2 expects."""
    try:
        import numpy
    except ImportError as error:
        raise SttProviderUnavailableError(
            'faster-whisper requires numpy. Install with: pip install -e ".[stt]"'
        ) from error

    samples = numpy.frombuffer(audio.data, dtype="<i2")
    if audio.channels > 1:
        samples = samples.reshape(-1, audio.channels).mean(axis=1)
    return samples.astype(numpy.float32) / 32768.0


def _importWhisperModel() -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise SttProviderUnavailableError(
            "Speech recognition requires the 'faster-whisper' package.\n"
            'Install with: pip install -e ".[stt]"'
        ) from error
    return WhisperModel


def _makeCudaLibrariesDiscoverable() -> None:
    """Put torch's bundled CUDA libraries on the Windows DLL search path.

    CTranslate2 needs cuBLAS 12 and cuDNN 9. A CUDA build of torch ships both
    in ``torch/lib``, but Windows only searches directories that have been
    registered, and importing torch registers them for torch alone. Without
    this, CUDA transcription fails with an opaque DLL load error even though
    the libraries are present.
    """
    if sys.platform != "win32":
        return

    candidates: list[Path] = []
    try:
        import torch

        candidates.append(Path(torch.__file__).parent / "lib")
    except ImportError:
        logger.debug("torch not installed; not adding its CUDA libraries to the search path")

    sitePackages = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if sitePackages.is_dir():
        candidates.extend(path for path in sitePackages.glob("*/bin") if path.is_dir())

    for directory in candidates:
        if not directory.is_dir():
            continue
        try:
            os.add_dll_directory(str(directory))
            logger.debug("Added %s to the DLL search path", directory)
        except OSError as error:
            logger.debug("Could not add %s to the DLL search path: %s", directory, error)


def _describeLoadFailure(error: Exception, modelSize: str, device: str) -> Exception:
    """Translate opaque load failures into actionable messages."""
    text = str(error).lower()

    if "cudnn" in text or "cublas" in text or "dll load failed" in text:
        return SttProviderUnavailableError(
            "CTranslate2 could not load the CUDA libraries it needs (cuBLAS 12 and "
            "cuDNN 9).\n"
            "Installing a CUDA build of torch provides both:\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cu126\n"
            "Or set speechToText.device to 'cpu'.\n\n"
            f"Original error: {error}"
        )

    if "out of memory" in text:
        return SttProviderUnavailableError(
            f"Not enough GPU memory to load the {modelSize!r} Whisper model. "
            "Use a smaller model, set speechToText.computeType to 'int8', "
            "or set speechToText.device to 'cpu'.\n\n"
            f"Original error: {error}"
        )

    if "efficient" in text or "not supported" in text or "compute type" in text:
        return SttProviderUnavailableError(
            f"The requested compute type is not supported on {device}. "
            "Set speechToText.computeType to 'auto', or try 'int8'.\n\n"
            f"Original error: {error}"
        )

    return SttProviderUnavailableError(
        f"Could not load Whisper model {modelSize!r} on {device}: {error}"
    )


__all__ = ["FasterWhisperProvider"]
