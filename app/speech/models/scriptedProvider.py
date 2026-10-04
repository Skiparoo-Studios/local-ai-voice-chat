"""A speech-to-text provider that returns pre-arranged text.

The counterpart to the tone provider: it lets the capture path, the assistant
runtime and the tests run without loading Whisper. Stage 3's conversation loop
depends on being testable this way, since a real round trip otherwise needs a
microphone and two models.

Phrases are returned in order and then repeated, so a scripted conversation can
be replayed deterministically.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from app.audio.audioBuffer import AudioBuffer
from app.speech.speechToText import SpeechToTextProvider, Transcript

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_PHRASES = (
    "turn the kitchen light off",
    "what is the weather like",
    "set a timer for ten minutes",
)


class ScriptedProvider(SpeechToTextProvider):
    """Returns configured phrases instead of transcribing."""

    name: ClassVar[str] = "scripted"

    def __init__(
        self,
        phrases: tuple[str, ...] = DEFAULT_PHRASES,
        language: str = "en",
    ) -> None:
        self._phrases = phrases or DEFAULT_PHRASES
        self._language = language
        self._index = 0
        self._loaded = False

    @classmethod
    def fromSettings(cls, settings: Settings) -> ScriptedProvider:
        return cls(language=settings.speechToText.language or "en")

    async def load(self) -> None:
        self._loaded = True
        logger.info("Scripted speech-to-text ready (no model to load)")

    async def unload(self) -> None:
        self._loaded = False

    @property
    def isLoaded(self) -> bool:
        return self._loaded

    async def transcribe(
        self,
        audio: AudioBuffer,
        *,
        language: str | None = None,
    ) -> Transcript:
        phrase = self._phrases[self._index % len(self._phrases)]
        self._index += 1

        return Transcript(
            text=phrase,
            language=language or self._language,
            languageConfidence=1.0,
            audioSeconds=audio.durationSeconds,
            transcriptionSeconds=0.0,
            segments=(phrase,),
        )

    def describe(self) -> str:
        return f"scripted ({len(self._phrases)} phrases)"
