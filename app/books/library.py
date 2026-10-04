"""Finding books, and the chapters inside them.

Two libraries, checked in that order because one is a recording of a
professional narrator and the other is the assistant's own voice reading text.
Given both, the recording wins.

    audiobooks   Books/Title [ID]/Title [ID] - 04 - Chapter 3. Name.mp3
    textbooks    TextBooks/name.txt

The folder name supplies the title rather than the file name, because the files carry
subtitles the folders do not --- "Digital Fortress_ A Thriller" against
"Digital Fortress" --- and the shorter one is what somebody says out loud.

Chapters in a text file are found by scanning for headings. They are numbered
by the order they appear rather than by parsing "CHAPTER SEVENTEEN", because
the order is always right and the spelling is not always parseable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.automation.deviceMatcher import tokenise

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = frozenset({".mp3", ".m4b", ".m4a", ".ogg", ".flac", ".wav"})
TEXT_EXTENSIONS = frozenset({".txt"})

# Each file is named "<book> - NN - <chapter>.<ext>". The number orders
# them; the tail is what a listener would call the chapter.
PART_PATTERN = re.compile(r"^(?P<book>.*?)\s+-\s+(?P<number>\d+)\s+-\s+(?P<title>.+)$")

# A trailing catalogue identifier in brackets, which nobody says aloud.
ASIN_PATTERN = re.compile(r"\s*\[[A-Za-z0-9]{8,12}\]\s*$")

# Front and back matter. Present in the file order, but "from the beginning"
# means the story, not the copyright notice.
CREDITS_PATTERN = re.compile(r"\bcredits\b", re.I)

# A heading in a plain text file. Deliberately narrow: a short line that opens
# with a structural word and is not a sentence.
HEADING_PATTERN = re.compile(
    r"^\s*(chapter|part|book|prologue|epilogue|introduction)\b[^.!?]*$", re.I
)

# The longest a line can be and still be a heading rather than prose.
MAXIMUM_HEADING_LENGTH = 60

# A heading that a conversion has welded onto the end of a paragraph, which is
# how "CHAPTER NINE" arrives in a real file: at the end of seven hundred
# characters of prose, on the same line. Required to be upper case and at the
# end of the line, so that a character saying "read chapter nine to me" in
# dialogue is not mistaken for one.
WELDED_HEADING_PATTERN = re.compile(
    r"(?<=[.!?\"'”’])\s+"  # noqa: RUF001 - real prose uses curly quotes
    r"(?P<heading>(?:CHAPTER|PART|BOOK)\s+[A-Z\-]{3,20}\s*)$"
)

# How much of a spoken request a title must account for. Half is too loose,
# since "the" would match everything; all of it is too strict for real speech.
MINIMUM_SCORE = 0.6


@dataclass(frozen=True, slots=True)
class Chapter:
    """One playable or readable division of a book."""

    number: int
    title: str
    # An audiobook has a file per chapter; a text chapter is a range of lines.
    path: Path | None = None
    startLine: int = 0
    endLine: int = 0

    @property
    def isCredits(self) -> bool:
        return bool(CREDITS_PATTERN.search(self.title))

    def describe(self) -> str:
        return f"{self.number}. {self.title}"


@dataclass(frozen=True, slots=True)
class Book:
    """Something that can be read or played."""

    title: str
    path: Path
    chapters: tuple[Chapter, ...] = ()

    @property
    def isAudio(self) -> bool:
        return False

    @property
    def storyChapters(self) -> tuple[Chapter, ...]:
        """Chapters excluding front and back matter.

        Falls back to everything when a book is nothing but credits, so that a
        strange library cannot produce a book with no chapters at all.
        """
        story = tuple(chapter for chapter in self.chapters if not chapter.isCredits)
        return story or self.chapters

    def chapterNumbered(self, number: int) -> Chapter | None:
        for chapter in self.chapters:
            if chapter.number == number:
                return chapter
        return None

    def findChapter(self, spoken: str) -> Chapter | None:
        """Resolve "chapter three", "the third one", or a chapter's name."""
        wanted = spokenNumber(spoken)
        if wanted is not None:
            story = self.storyChapters
            # Counted among the story chapters first: an audiobook carries
            # opening credits as part 1, so "chapter three" is the third
            # chapter, not the file numbered three.
            if 1 <= wanted <= len(story):
                return story[wanted - 1]
            numbered = self.chapterNumbered(wanted)
            if numbered is not None:
                return numbered
            # Falls through rather than giving up. Text chapters are numbered
            # by the order their headings appear, which need not agree with
            # what a heading calls itself: a file missing a chapter has one
            # named "Chapter Nine" sitting in position eight. Matching the
            # name catches what counting cannot.

        tokens = tokenise(spoken)
        if not tokens:
            return None

        best, bestScore = None, 0.0
        for chapter in self.chapters:
            score = overlap(tokens, tokenise(chapter.title))
            if score > bestScore:
                best, bestScore = chapter, score
        return best if bestScore >= MINIMUM_SCORE else None

    def describe(self) -> str:
        kind = "audiobook" if self.isAudio else "text"
        return f"{self.title} ({len(self.chapters)} chapter(s), {kind})"


@dataclass(frozen=True, slots=True)
class AudioBook(Book):
    """A recording, one file per chapter."""

    @property
    def isAudio(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class TextBook(Book):
    """A plain text file, to be read aloud by the assistant."""

    lines: tuple[str, ...] = field(default=(), repr=False)

    def textFor(self, chapter: Chapter) -> str:
        """The prose of one chapter, blank lines removed."""
        body = self.lines[chapter.startLine : chapter.endLine or None]
        return "\n".join(line.strip() for line in body if line.strip())


@dataclass(frozen=True, slots=True)
class BookMatch:
    """What a spoken title resolved to."""

    book: Book | None = None
    alternatives: tuple[Book, ...] = ()

    @property
    def found(self) -> bool:
        return self.book is not None and not self.ambiguous

    @property
    def ambiguous(self) -> bool:
        return len(self.alternatives) > 1


class BookLibrary:
    """The two directories, and what is in them."""

    def __init__(self, audioDirectory: Path, textDirectory: Path) -> None:
        self._audioDirectory = audioDirectory
        self._textDirectory = textDirectory

    @property
    def audioDirectory(self) -> Path:
        return self._audioDirectory

    @property
    def textDirectory(self) -> Path:
        return self._textDirectory

    # --- Discovery --------------------------------------------------------

    def audioBooks(self) -> list[AudioBook]:
        """Every audiobook, one per folder, in alphabetical order."""
        if not self._audioDirectory.is_dir():
            logger.debug("No audiobook directory at %s", self._audioDirectory)
            return []

        books = []
        for folder in sorted(self._audioDirectory.iterdir()):
            if not folder.is_dir():
                continue
            chapters = audioChapters(folder)
            if chapters:
                books.append(
                    AudioBook(title=cleanTitle(folder.name), path=folder, chapters=chapters)
                )
        return books

    def textBooks(self) -> list[TextBook]:
        """Every readable text file. Plain text only, for now."""
        if not self._textDirectory.is_dir():
            logger.debug("No textbook directory at %s", self._textDirectory)
            return []

        books = []
        for path in sorted(self._textDirectory.iterdir()):
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            book = loadTextBook(path)
            if book is not None:
                books.append(book)
        return books

    def all(self) -> list[Book]:
        """Audiobooks first, because a recording beats synthesis."""
        return [*self.audioBooks(), *self.textBooks()]

    # --- Looking things up ------------------------------------------------

    def find(self, spoken: str, *, preferText: bool = False) -> BookMatch:
        """Resolve a spoken title, audiobooks first.

        Several books can answer one request --- seven of them are called
        Harry Potter something --- so the alternatives travel with the answer
        and the caller asks rather than guessing.
        """
        tokens = tokenise(spoken)
        if not tokens:
            return BookMatch()

        shelves = [self.audioBooks(), self.textBooks()]
        if preferText:
            # Asked for by name: "read me the text of harry potter". Without
            # this the seven Harry Potter recordings always answer first and
            # the text library cannot be reached by voice at all.
            shelves.reverse()

        for shelf in shelves:
            scored = [(book, overlap(tokens, tokenise(book.title))) for book in shelf]
            viable = sorted(
                (item for item in scored if item[1] >= MINIMUM_SCORE),
                key=lambda item: (-item[1], item[0].title),
            )
            if not viable:
                continue

            best = viable[0][1]
            # Only equally good matches count as alternatives. "Harry Potter"
            # hits seven books at the same score; "chamber of secrets" hits one
            # better than the rest and is not ambiguous at all.
            tied = tuple(book for book, score in viable if score >= best - 1e-9)
            return BookMatch(book=tied[0], alternatives=tied)

        return BookMatch()

    def unreadableMatch(self, spoken: str) -> Path | None:
        """A file that answers the request but is in a format not read yet.

        "I have The Hobbit but only as an epub" is a far better answer than
        "I couldn't find it", and this library genuinely contains that case.
        """
        if not self._textDirectory.is_dir():
            return None

        tokens = tokenise(spoken)
        if not tokens:
            return None

        best, bestScore = None, 0.0
        for path in sorted(self._textDirectory.iterdir()):
            if path.is_dir() or path.suffix.lower() in TEXT_EXTENSIONS:
                continue
            score = overlap(tokens, tokenise(cleanTitle(path.stem)))
            if score > bestScore:
                best, bestScore = path, score
        return best if bestScore >= MINIMUM_SCORE else None

    def suggestions(self, count: int = 3) -> list[Book]:
        """A few titles to offer when nobody has named one.

        Spread across the shelf rather than taken from the front, so a library
        with seven Harry Potters does not answer with three Harry Potters.
        """
        books = self.all()
        if len(books) <= count:
            return books
        step = len(books) / count
        return [books[int(index * step)] for index in range(count)]

    def describe(self) -> str:
        return (
            f"{len(self.audioBooks())} audiobook(s) in {self._audioDirectory}, "
            f"{len(self.textBooks())} text book(s) in {self._textDirectory}"
        )


# --- Parsing -------------------------------------------------------------


def spokenName(path: Path) -> str:
    """A file name as somebody would say it.

    Library tools decorate names --- "The Hobbit - Tolkien,
    J. R. R._5022" --- and reading that back aloud is absurd. The author and
    the numeric identifier go.
    """
    stem = ASIN_PATTERN.sub("", path.stem).strip()
    stem = re.sub(r"_\d+$", "", stem)
    return stem.split(" - ")[0].strip() or stem


def cleanTitle(name: str) -> str:
    """A folder or file name as somebody would say it."""
    return ASIN_PATTERN.sub("", name).strip()


def audioChapters(folder: Path) -> tuple[Chapter, ...]:
    """The playable files in a book folder, in their numbered order."""
    chapters: list[Chapter] = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        match = PART_PATTERN.match(path.stem)
        if match:
            number = int(match.group("number"))
            # A colon arrives as an underscore, since Windows forbids one in a
            # file name.
            title = cleanTitle(match.group("title")).replace("_", ":")
        else:
            # A single-file book, or a naming scheme this does not know.
            number = len(chapters) + 1
            title = cleanTitle(path.stem)
        chapters.append(Chapter(number=number, title=title, path=path))

    return tuple(sorted(chapters, key=lambda chapter: chapter.number))


def loadTextBook(path: Path) -> TextBook | None:
    """Read a text file and divide it into chapters."""
    try:
        # utf-8-sig because a byte order mark would otherwise become part of
        # the first heading and stop it matching.
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as error:
        logger.warning("Could not read %s: %s", path, error)
        return None

    lines = separateWeldedHeadings(text.splitlines())
    return TextBook(
        title=cleanTitle(path.stem).replace("_", " "),
        path=path,
        chapters=textChapters(lines),
        lines=lines,
    )


def separateWeldedHeadings(lines: list[str]) -> tuple[str, ...]:
    """Put a heading stuck to the end of a paragraph back on its own line.

    Measured against a real file: one chapter heading in seventeen had been
    welded onto the end of the preceding paragraph by whatever produced the
    text. Left alone, that chapter cannot be asked for and the one before it is
    twice the length it should be.
    """
    separated: list[str] = []
    for line in lines:
        match = WELDED_HEADING_PATTERN.search(line)
        if match is None:
            separated.append(line)
            continue

        separated.append(line[: match.start()].rstrip())
        separated.append(match.group("heading").strip())
        logger.debug("Separated a welded heading: %r", match.group("heading").strip())

    return tuple(separated)


def textChapters(lines: tuple[str, ...]) -> tuple[Chapter, ...]:
    """Divide lines into chapters at their headings."""
    marks: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) > MAXIMUM_HEADING_LENGTH:
            continue
        if HEADING_PATTERN.match(stripped):
            marks.append((index, titleAt(lines, index)))

    if not marks:
        # No headings at all, so the whole file is one chapter.
        return (Chapter(number=1, title="the whole book", startLine=0, endLine=len(lines)),)

    chapters = []
    for position, (start, title) in enumerate(marks):
        end = marks[position + 1][0] if position + 1 < len(marks) else len(lines)
        chapters.append(Chapter(number=position + 1, title=title, startLine=start, endLine=end))
    return tuple(chapters)


def titleAt(lines: tuple[str, ...], index: int) -> str:
    """A heading, plus the line after it when that line is a title too.

    Text books commonly put "CHAPTER ONE" on one line and "THE BOY WHO LIVED"
    on the next, and only the pair is worth announcing.
    """
    heading = lines[index].strip()
    for offset in (1, 2):
        if index + offset >= len(lines):
            break
        candidate = lines[index + offset].strip()
        if not candidate:
            continue
        # Prose is long and ends in punctuation; a title does neither.
        if len(candidate) <= MAXIMUM_HEADING_LENGTH and not candidate.endswith(
            (".", "!", "?", ",", ";", ":")
        ):
            spoken = candidate.title() if candidate.isupper() else candidate
            return f"{heading.title() if heading.isupper() else heading}: {spoken}"
        break
    return heading.title() if heading.isupper() else heading


def overlap(query: list[str], candidate: list[str]) -> float:
    """How much of what was said is accounted for by a title.

    A title is also compared with its spaces removed, because a file on disk
    is called harrypotter.txt while the person asking for it says "harry
    potter". Without that the two never meet, and the text library is
    unreachable by voice.
    """
    if not query:
        return 0.0

    found = sum(1 for token in query if token in candidate)
    if found == len(query):
        return 1.0

    joined = "".join(candidate)
    if joined and "".join(query) == joined:
        return 1.0
    # Every spoken word appearing inside one run-together title counts, so
    # "harry potter" matches "harrypotter" but "harry" alone does not reach
    # the threshold on its own strength.
    if joined and all(token in joined for token in query) and len(query) > 1:
        return 1.0

    return found / len(query)


NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20,
}

ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20,
}


def spokenNumber(text: str) -> int | None:
    """The number in "chapter three", "chapter 3" or "the third one"."""
    for token in tokenise(text):
        if token.isdigit():
            return int(token)
        if token in NUMBER_WORDS:
            return NUMBER_WORDS[token]
        if token in ORDINAL_WORDS:
            return ORDINAL_WORDS[token]
    return None


__all__ = [
    "AudioBook",
    "Book",
    "BookLibrary",
    "BookMatch",
    "Chapter",
    "TextBook",
    "cleanTitle",
    "loadTextBook",
    "spokenName",
    "spokenNumber",
]
