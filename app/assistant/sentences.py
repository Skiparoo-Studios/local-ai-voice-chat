"""Turning a stream of model fragments into speakable pieces.

A language model emits a few characters at a time. Synthesis needs whole
phrases: too short and the speech is clipped and toneless, too long and the
latency saved by streaming is given straight back.

This accumulates fragments and releases text at sentence boundaries, subject to
a minimum length so that "Yes." does not become its own utterance, and a
maximum so that a model producing no punctuation still gets spoken.
"""

from __future__ import annotations

import re

DEFAULT_MINIMUM_CHARACTERS = 20
DEFAULT_MAXIMUM_CHARACTERS = 200

# A sentence ends at . ! ? or a newline, allowing for closing quotes and
# brackets, and must be followed by whitespace or the end of the buffer.
SENTENCE_END = re.compile(r'[.!?]["\')\]]*(?=\s)|\n+')

# Abbreviations that end in a full stop without ending a sentence. Kept short
# deliberately: an over-eager list would suppress real sentence ends.
ABBREVIATIONS = frozenset(
    {"mr", "mrs", "ms", "dr", "prof", "st", "e.g", "i.e", "etc", "vs", "approx"}
)


class SentenceStreamer:
    """Accumulates fragments and releases complete pieces of speech."""

    def __init__(
        self,
        minimumCharacters: int = DEFAULT_MINIMUM_CHARACTERS,
        maximumCharacters: int = DEFAULT_MAXIMUM_CHARACTERS,
    ) -> None:
        if minimumCharacters < 0:
            raise ValueError("minimumCharacters cannot be negative")
        if maximumCharacters < max(1, minimumCharacters):
            raise ValueError("maximumCharacters must be at least minimumCharacters")

        self._minimumCharacters = minimumCharacters
        self._maximumCharacters = maximumCharacters
        self._buffer = ""

    @property
    def pending(self) -> str:
        """Text held back, waiting for a boundary."""
        return self._buffer

    def feed(self, fragment: str) -> list[str]:
        """Add a fragment, returning any pieces that are now ready to speak."""
        if fragment:
            self._buffer += fragment

        ready: list[str] = []
        while True:
            piece = self._takeNext()
            if piece is None:
                break
            ready.append(piece)
        return ready

    def flush(self) -> str | None:
        """Release whatever remains, however short. Called when the stream ends."""
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining or None

    def _takeNext(self) -> str | None:
        candidate = self._takeAtSentenceEnd()
        if candidate is not None:
            return candidate
        return self._takeAtLengthLimit()

    def _takeAtSentenceEnd(self) -> str | None:
        for match in SENTENCE_END.finditer(self._buffer):
            end = match.end()
            piece = self._buffer[:end].strip()

            if len(piece) < self._minimumCharacters:
                # Too short to speak well on its own; wait for more.
                continue
            if _endsWithAbbreviation(piece):
                continue

            self._buffer = self._buffer[end:].lstrip()
            return piece
        return None

    def _takeAtLengthLimit(self) -> str | None:
        """Split an over-long buffer on whitespace.

        A model that produces no punctuation would otherwise never yield
        anything until the stream ended, defeating the point of streaming.
        """
        if len(self._buffer) <= self._maximumCharacters:
            return None

        cut = self._buffer.rfind(" ", self._minimumCharacters, self._maximumCharacters)
        if cut <= 0:
            cut = self._maximumCharacters

        piece = self._buffer[:cut].strip()
        self._buffer = self._buffer[cut:].lstrip()
        return piece or None


def _endsWithAbbreviation(piece: str) -> bool:
    """Whether the final full stop belongs to an abbreviation."""
    if not piece.endswith("."):
        return False
    lastWord = piece[:-1].rsplit(" ", 1)[-1].lower()
    return lastWord in ABBREVIATIONS


def splitIntoSentences(
    text: str,
    minimumCharacters: int = DEFAULT_MINIMUM_CHARACTERS,
    maximumCharacters: int = DEFAULT_MAXIMUM_CHARACTERS,
) -> list[str]:
    """Split complete text the same way the streamer would.

    Used to compare streamed and non-streamed synthesis on equal terms.
    """
    streamer = SentenceStreamer(minimumCharacters, maximumCharacters)
    pieces = streamer.feed(text)
    remainder = streamer.flush()
    if remainder:
        pieces.append(remainder)
    return pieces
