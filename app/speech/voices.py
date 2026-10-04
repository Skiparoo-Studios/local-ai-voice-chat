"""Voice sample discovery.

Voice recordings and anything derived from them stay out of source control and
out of the application package. This module only locates them; how a particular
model turns samples into a speaker representation is that provider's business.

Two layouts are supported under the voices directory:

    voices/michael.wav          a single sample
    voices/michael/*.wav        several samples, which clone better

"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SAMPLE_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
CACHE_DIRECTORY_NAME = ".cache"


class VoiceNotFoundError(Exception):
    """Raised when a requested voice has no samples."""


@dataclass(frozen=True, slots=True)
class Voice:
    """A named voice and the recordings that define it."""

    name: str
    samples: tuple[Path, ...]

    @property
    def sampleCount(self) -> int:
        return len(self.samples)


class VoiceLibrary:
    """Resolves voice names to sample files on disk."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    @property
    def cacheDirectory(self) -> Path:
        """Where providers may cache derived data such as embeddings."""
        return self._root / CACHE_DIRECTORY_NAME

    def listVoices(self) -> list[str]:
        """Names of every voice with at least one sample, sorted."""
        if not self._root.is_dir():
            return []

        names: set[str] = set()
        for entry in self._root.iterdir():
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                if any(self._samplesIn(entry)):
                    names.add(entry.name)
            elif entry.suffix.lower() in SAMPLE_EXTENSIONS:
                names.add(entry.stem)
        return sorted(names)

    def has(self, name: str) -> bool:
        try:
            self.get(name)
        except VoiceNotFoundError:
            return False
        return True

    def get(self, name: str) -> Voice:
        """Resolve ``name``, raising a helpful error if it does not exist."""
        if not name:
            raise VoiceNotFoundError("No voice name was given.")

        directory = self._root / name
        if directory.is_dir():
            samples = tuple(sorted(self._samplesIn(directory)))
            if samples:
                return Voice(name=name, samples=samples)

        for extension in SAMPLE_EXTENSIONS:
            candidate = self._root / f"{name}{extension}"
            if candidate.is_file():
                return Voice(name=name, samples=(candidate,))

        raise VoiceNotFoundError(self._describeMissing(name))

    def _samplesIn(self, directory: Path) -> list[Path]:
        return [
            entry
            for entry in directory.iterdir()
            if entry.is_file() and entry.suffix.lower() in SAMPLE_EXTENSIONS
        ]

    def _describeMissing(self, name: str) -> str:
        available = self.listVoices()

        if not available:
            return (
                f"Voice {name!r} not found: there are no voice samples in {self._root}.\n"
                f"Add a recording as {self._root / (name + '.wav')}, or several as "
                f"{self._root / name / '*.wav'}.\n"
                "Six to thirty seconds of clean speech works well for cloning."
            )

        message = f"Voice {name!r} not found in {self._root}."
        close = difflib.get_close_matches(name, available, n=3, cutoff=0.6)
        if close:
            message += f" Did you mean: {', '.join(close)}?"
        message += f"\nAvailable voices: {', '.join(available)}"
        return message
