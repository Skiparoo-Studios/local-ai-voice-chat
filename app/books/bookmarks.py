"""Where each book was left off.

Resume is only worth having if it survives the assistant being closed. Somebody
stops a chapter at bedtime and asks for it again the next evening, which is a
different process, so the position goes on disk rather than into memory.

One entry per book, not per chapter: what anyone means by "carry on" is the
last place *that book* was playing, and keeping a position for every chapter
of every book would answer a question nobody asks.

The file is small and written on every stop, which is a few hundred bytes a
few times an evening. It is deliberately not the settings file --- this is
state the assistant maintains, not configuration a person edits, and mixing
the two makes both harder to reason about.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_BOOKMARKS_PATH = Path("outputs/bookmarks.json")

# Resuming a few seconds before the stop makes the join sound deliberate
# rather than clipped, and covers the block already in the device when the
# request to stop arrived.
REWIND_SECONDS = 3.0

# Below this, resuming is not worth the words: start the chapter again.
MINIMUM_WORTH_RESUMING = 10.0


@dataclass(frozen=True, slots=True)
class Bookmark:
    """How far into a book somebody had got."""

    title: str
    chapterNumber: int
    chapterTitle: str
    # Audiobooks measure a position in seconds; a text book counts sentences,
    # because that is what it is read in and seconds would be a guess.
    offsetSeconds: float = 0.0
    sentenceIndex: int = 0

    @property
    def isWorthResuming(self) -> bool:
        return self.offsetSeconds >= MINIMUM_WORTH_RESUMING or self.sentenceIndex > 0

    @property
    def resumeSeconds(self) -> float:
        """Where to start, backed off a little so the join is not abrupt."""
        return max(0.0, self.offsetSeconds - REWIND_SECONDS)

    def describe(self) -> str:
        if self.sentenceIndex:
            return f"{self.chapterTitle}, sentence {self.sentenceIndex}"
        minutes, seconds = divmod(int(self.offsetSeconds), 60)
        return f"{self.chapterTitle}, {minutes} minute(s) {seconds} second(s) in"

    def toDict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "chapterNumber": self.chapterNumber,
            "chapterTitle": self.chapterTitle,
            "offsetSeconds": round(self.offsetSeconds, 2),
            "sentenceIndex": self.sentenceIndex,
        }

    @classmethod
    def fromDict(cls, data: dict[str, object]) -> Bookmark | None:
        try:
            return cls(
                title=str(data["title"]),
                chapterNumber=int(data["chapterNumber"]),  # type: ignore[arg-type]
                chapterTitle=str(data.get("chapterTitle", "")),
                offsetSeconds=float(data.get("offsetSeconds", 0.0)),  # type: ignore[arg-type]
                sentenceIndex=int(data.get("sentenceIndex", 0)),  # type: ignore[arg-type]
            )
        except (KeyError, TypeError, ValueError) as error:
            logger.warning("Ignoring a malformed bookmark: %s", error)
            return None


class Bookmarks:
    """The positions, kept on disk.

    A failure to read or write is logged and swallowed. Losing a bookmark is a
    minor annoyance; refusing to play a book because one could not be saved
    would be a much worse one.
    """

    def __init__(self, path: Path = DEFAULT_BOOKMARKS_PATH) -> None:
        self._path = path
        self._marks: dict[str, Bookmark] = {}
        self._loaded = False

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        if not self._path.is_file():
            return

        try:
            data = json.loads(self._path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("Could not read bookmarks from %s: %s", self._path, error)
            return

        if not isinstance(data, dict):
            logger.warning("%s does not contain an object; ignoring it", self._path)
            return

        for title, entry in data.items():
            if isinstance(entry, dict):
                mark = Bookmark.fromDict(entry)
                if mark is not None:
                    self._marks[_key(title)] = mark

    def save(self, mark: Bookmark) -> None:
        """Record where a book was left, replacing any earlier position."""
        self._load()
        self._marks[_key(mark.title)] = mark

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {mark.title: mark.toDict() for mark in self._marks.values()}
            self._path.write_text(
                json.dumps(payload, indent=4) + "\n", encoding="utf-8"
            )
        except OSError as error:
            logger.warning("Could not save a bookmark to %s: %s", self._path, error)
            return
        logger.info("Bookmarked %s at %s", mark.title, mark.describe())

    def forTitle(self, title: str) -> Bookmark | None:
        self._load()
        return self._marks.get(_key(title))

    def mostRecent(self) -> Bookmark | None:
        """The last book bookmarked, for a bare "carry on"."""
        self._load()
        return next(reversed(self._marks.values()), None) if self._marks else None

    def clear(self, title: str) -> None:
        """Forget a book, which is what finishing a chapter means."""
        self._load()
        if self._marks.pop(_key(title), None) is None:
            return
        try:
            payload = {mark.title: mark.toDict() for mark in self._marks.values()}
            self._path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
        except OSError as error:
            logger.debug("Could not rewrite bookmarks: %s", error)

    def __len__(self) -> int:
        self._load()
        return len(self._marks)

    def describe(self) -> str:
        self._load()
        return f"{len(self._marks)} bookmark(s) in {self._path}"


def _key(title: str) -> str:
    """Case and spacing should not create a second entry for one book."""
    return " ".join(title.lower().split())


__all__ = ["DEFAULT_BOOKMARKS_PATH", "Bookmark", "Bookmarks"]
