"""A synthetic provider that produces tones rather than speech.

It exists to keep the rest of the system honest: the pipeline, the audio
output path and the REPL can all be exercised without a 2 GB model download,
and tests can run anywhere. Different voice names produce different pitches,
which makes it obvious at a glance whether the voice parameter is actually
reaching the provider.
"""

from __future__ import annotations

import asyncio
import logging
import zlib
from typing import TYPE_CHECKING, ClassVar

from app.audio.audioBuffer import AudioBuffer
from app.speech.textToSpeech import TextToSpeechProvider
from app.speech.voices import VoiceLibrary

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

SECONDS_PER_WORD = 0.22
MINIMUM_SECONDS = 0.25
GAP_SECONDS = 0.05


class ToneProvider(TextToSpeechProvider):
    """Generates a tone per word, at a pitch derived from the voice name."""

    name: ClassVar[str] = "tone"

    def __init__(
        self,
        *,
        voice: str = "default",
        language: str = "en",
        sampleRate: int = 22050,
    ) -> None:
        self._voice = voice
        self._language = language
        self._sampleRate = sampleRate
        self._loaded = False

    @classmethod
    def fromSettings(cls, settings: Settings, voices: VoiceLibrary) -> ToneProvider:
        return cls(
            voice=settings.textToSpeech.voice,
            language=settings.textToSpeech.language,
        )

    async def load(self) -> None:
        self._loaded = True
        logger.info("Tone provider ready (no model to load)")

    async def unload(self) -> None:
        self._loaded = False

    @property
    def isLoaded(self) -> bool:
        return self._loaded

    @property
    def sampleRate(self) -> int:
        return self._sampleRate

    async def synthesise(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
    ) -> AudioBuffer:
        return await asyncio.to_thread(self._render, text, voice or self._voice)

    def _render(self, text: str, voice: str) -> AudioBuffer:
        words = text.split() or [""]
        base = self._pitchFor(voice)
        gap = AudioBuffer.silence(GAP_SECONDS, self._sampleRate)

        buffer = AudioBuffer(data=b"", sampleRate=self._sampleRate)
        for index, word in enumerate(words):
            # Vary within the utterance so speech-like contour is audible.
            frequency = base * (1.0 + 0.06 * ((index % 5) - 2))
            duration = max(MINIMUM_SECONDS, len(word) * 0.06)
            buffer = buffer.concatenate(
                AudioBuffer.tone(frequency, duration, self._sampleRate)
            ).concatenate(gap)
        return buffer

    @staticmethod
    def _pitchFor(voice: str) -> float:
        """Map a voice name to a stable pitch between 180 Hz and 380 Hz."""
        digest = zlib.crc32(voice.encode("utf-8"))
        return 180.0 + (digest % 200)

    def availableVoices(self) -> list[str]:
        return ["default", "high", "low"]

    def describe(self) -> str:
        return f"tone (synthetic, {self._sampleRate} Hz)"
