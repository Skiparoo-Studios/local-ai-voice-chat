"""Asking for a book, and being read it.

Wraps the ordinary handler rather than replacing it, so "read me a story"
works from normal conversation and everything else carries on reaching the
router. Three turns at most:

    "read me a story"        -> what would you like? here are a few
    "harry potter"           -> I have seven of those. which one?
    "the chamber of secrets" -> which chapter, or from the beginning?
    "from the beginning"     -> reading starts

A book named in the first breath skips straight to the chapter question, and
"stop" ends it at any point.

Audiobooks are searched before text books, because one is a professional
narrator and the other is this assistant reading aloud. A text book is only
reached when no recording answers.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from enum import StrEnum, auto
from typing import TYPE_CHECKING, ClassVar

from app.assistant.conversation import Conversation
from app.assistant.handlers import Response, ResponseHandler
from app.books.bookmarks import Bookmark, Bookmarks
from app.books.library import Book, BookLibrary, spokenName
from app.books.reader import BookReader

if TYPE_CHECKING:
    from app.intelligence.llmProvider import LlmProvider

logger = logging.getLogger(__name__)

# Asking to be read to, without necessarily naming anything.
# What a book may be called. "audio book" as two words is what a recogniser
# usually writes it as, and leaving that out sent "play me an audio book" past
# this handler to the device router, which then asked which light to turn on.
BOOK_WORD = r"(?:audio\s*books?|books?|stor(?:y|ies)|novel)"

REQUEST_PATTERNS = (
    re.compile(
        rf"\b(?:read|play|put\s+on|start|begin)\s+(?:me\s+)?"
        rf"(?:an?|the|some)?\s*{BOOK_WORD}\b",
        re.I,
    ),
    re.compile(r"\bread\s+to\s+me\b", re.I),
    re.compile(r"\btell\s+me\s+a\s+story\b", re.I),
    re.compile(rf"\blisten\s+to\s+(?:an?|the|some)?\s*{BOOK_WORD}\b", re.I),
    re.compile(r"\b(?:read|play)\s+(?:me\s+)?(?:some\s+)?(?:more|of)\b", re.I),
)

# "read me harry potter", "play the chamber of secrets" --- a request with the
# title inside it. The title is whatever follows.
TITLED_PATTERN = re.compile(
    r"^\W*(?:please\s+)?(?:can you\s+)?(?:read|play|put on|start|begin)\s+"
    r"(?:me\s+)?(?:the\s+)?(?:audio\s*book|book|story|novel)?\s*"
    r"(?P<title>.+?)\s*$",
    re.I,
)

# Ending a pending question, where a bare word is unambiguous because nothing
# is playing to talk over it.
CANCEL_STOP_PATTERN = re.compile(
    r"^\W*(stop|quiet|silence|enough|shut up|that.s enough|forget it)\W*$", re.I
)

# Ending a book. Deliberately requires naming it: a narrator saying "stop" is
# common, and one that ends the chapter is worse than having to say two words.
STOP_BOOK_PATTERN = re.compile(
    r"^\W*(?:please\s+)?"
    r"(?:stop|pause|end|cancel|turn off|switch off|shut off|quit)\s+"
    r"(?:the\s+|this\s+|that\s+|my\s+)?"
    r"(?:audio\s*book|book|story|novel|reading|chapter|it)"
    r"\W*$",
    re.I,
)

BEGINNING_PATTERN = re.compile(
    r"\b(beginning|start|first|start of it|the top|begin)\b", re.I
)

# Reading is something you only do to books, so a title given to it is
# claimed even when the library has nothing by that name --- saying so beats
# letting a language model invent a plot. "Play" is not on this list.
READING_VERB_PATTERN = re.compile(r"^\W*(?:please\s+)?(?:can you\s+)?read\b", re.I)

# The only things acted on while a book is playing. Anything else heard then
# is almost certainly the book itself coming back through the microphone.
NEXT_CHAPTER_PATTERN = re.compile(
    r"^\W*(next( chapter)?|skip( ahead)?|move on)\W*$", re.I
)
PREVIOUS_CHAPTER_PATTERN = re.compile(
    r"^\W*((go )?back|previous( chapter)?|last chapter|again)\W*$", re.I
)
WHAT_IS_THIS_PATTERN = re.compile(
    r"^\W*(what(.s| is| are we)? (this|that|playing|reading|we listening to)"
    r"|which (chapter|book)( is (this|it))?)\W*$",
    re.I,
)

# Asking for the written version rather than the recording. Without this the
# seven Harry Potter recordings always answer first and the text library is
# unreachable by voice.
# Longest first: alternation is ordered, and "text" alone would otherwise
# match inside "text book" and leave "book" behind to be taken for part of the
# title.
TEXT_HINT_PATTERN = re.compile(
    r"\b(text\s*books?|text\s*files?|text\s*version|written\s*version"
    r"|text|txt|written|e-?book)\b",
    re.I,
)

# Carrying on where a book was left off.
RESUME_PATTERN = re.compile(
    r"^\W*(?:please\s+)?(resume|continue|carry on|keep going|pick up|"
    r"where (?:we|i) (?:left off|got to)|resume (?:the )?(?:book|audiobook|story))"
    r"(?:\s+(?:the\s+)?(?:book|audiobook|story|reading|listening))?\W*$",
    re.I,
)

CANCEL_PATTERN = re.compile(r"^\W*(never ?mind|forget it|cancel|nothing|no thanks)\W*$", re.I)

# Words that are a request to read but name nothing, so they must not be
# mistaken for a title by TITLED_PATTERN.
NOT_A_TITLE = re.compile(
    rf"^\W*(?:me\s+)?(?:an?|the|some)?\s*(?:{BOOK_WORD}|to\s+me|something|anything)?"
    rf"\W*$",
    re.I,
)


class Waiting(StrEnum):
    """What the handler expects to be told next."""

    nothing = auto()
    book = auto()
    which = auto()
    chapter = auto()


class BookHandler(ResponseHandler):
    """Answers requests to be read to; delegates everything else."""

    name: ClassVar[str] = "books"

    def __init__(
        self,
        base: ResponseHandler,
        library: BookLibrary,
        reader: BookReader,
        bookmarks: Bookmarks | None = None,
    ) -> None:
        self._base = base
        self._library = library
        self._reader = reader
        # Not `bookmarks or Bookmarks()`: Bookmarks defines __len__, so an
        # empty one is falsy and a caller's store would be silently
        # swapped for the default path. Which is exactly what happened.
        self._bookmarks = Bookmarks() if bookmarks is None else bookmarks

        self._waiting = Waiting.nothing
        self._book: Book | None = None
        self._candidates: tuple[Book, ...] = ()

    # --- Properties -------------------------------------------------------

    @property
    def base(self) -> ResponseHandler:
        return self._base

    @property
    def library(self) -> BookLibrary:
        return self._library

    @property
    def reader(self) -> BookReader:
        return self._reader

    @property
    def supportsStreaming(self) -> bool:
        # Its own replies are one sentence and assembled whole. Delegated
        # turns are the base handler's business, and it is asked per turn.
        return False if self._waiting is not Waiting.nothing else self._base.supportsStreaming

    @property
    def usesLlm(self) -> bool:
        return self._base.usesLlm

    @property
    def languageModel(self) -> LlmProvider | None:
        return self._base.languageModel

    @property
    def promptAfterSeconds(self) -> float | None:
        # Never nags. A reader in the middle of a chapter is not a silence to
        # fill, and a pending question is answered when the listener chooses.
        return None

    def __getattr__(self, name: str) -> object:
        """Forward anything unrecognised to the handler being wrapped.

        The API asks a handler for its ``registry`` to list devices, and the
        prompt asks for ``provider``. Inserting a wrapper must not remove
        those.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._base, name)

    # --- Lifecycle --------------------------------------------------------

    async def load(self) -> None:
        await self._base.load()

    async def warmUp(self) -> None:
        await self._base.warmUp()

    async def unload(self) -> None:
        await self._reader.stop()
        await self._base.unload()

    # --- Turns ------------------------------------------------------------

    async def respond(self, text: str, conversation: Conversation) -> Response:
        cleaned = text.strip()
        if not cleaned:
            return Response(text="", handledBy=self.name)

        # Stopping a book has to name it. While a chapter is playing the
        # microphone is full of narration, and a narrator saying "stop" ending
        # the chapter is far worse than having to say two words.
        if self._reader.isReading and STOP_BOOK_PATTERN.match(cleaned):
            return await self._stop()

        # Nothing is playing, so a bare word is unambiguous.
        if self._waiting is not Waiting.nothing and CANCEL_STOP_PATTERN.match(cleaned):
            return await self._stop()

        # While a book is playing, the microphone is hearing the book. Every
        # sentence of it arrives here looking like something a person said,
        # and answering it was catastrophic: the reply opened the speaker at
        # the assistant's sample rate, which closed the stream the book was
        # already writing to, and the audio broke up and then stopped
        # entirely. So nothing but a command is acted on, and anything else
        # is discarded without a word --- an empty reply is never spoken, so
        # there is only ever one writer to the device.
        if self._reader.isReading:
            command = await self._commandWhileReading(cleaned)
            if command is not None:
                return command
            logger.debug("Ignored %r while reading", cleaned[:60])
            return Response(text="", handledBy=f"{self.name}:ignored")

        if RESUME_PATTERN.match(cleaned):
            return await self._resume()

        if self._waiting is not Waiting.nothing:
            answered = await self._continueConversation(cleaned)
            if answered is not None:
                return answered

        opening = await self._openingRequest(cleaned)
        if opening is not None:
            return opening

        return await self._base.respond(cleaned, conversation)

    async def respondStream(
        self, text: str, conversation: Conversation
    ) -> AsyncIterator[str]:
        cleaned = text.strip()
        engaged = self._waiting is not Waiting.nothing or self._reader.isReading
        if engaged or self._looksLikeARequest(cleaned):
            response = await self.respond(cleaned, conversation)
            if response.text:
                yield response.text
            return

        async for fragment in self._base.respondStream(cleaned, conversation):
            yield fragment

    # --- Starting ---------------------------------------------------------

    def _looksLikeARequest(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in REQUEST_PATTERNS)

    async def _openingRequest(self, text: str) -> Response | None:
        """The first turn: a request to be read to, with or without a title.

        "Read me a story" is unmistakable. "Read me Moby Dick" is too, because
        reading is what you do to books. "Play the hallway light" is not, and
        the difference matters: "play" also means music and switches, so a
        title given to it is only claimed when the library can answer it.
        """
        if self._looksLikeARequest(text):
            title = self._titleIn(text)
            return (
                await self._offer(title, preferText=self._wantsText(text))
                if title
                else self._askWhichBook()
            )

        title = self._titleIn(text)
        if not title:
            return None

        if READING_VERB_PATTERN.match(text):
            # "Read" means a book. Answering "I don't have that" is better
            # than handing it to a language model to invent something.
            return await self._offer(title, preferText=self._wantsText(text))

        # An overloaded verb. Only claimed if there is really a book.
        if self._library.find(title).book is not None:
            return await self._offer(title)
        if self._library.unreadableMatch(title) is not None:
            return await self._offer(title)
        return None

    def _wantsText(self, text: str) -> bool:
        return bool(TEXT_HINT_PATTERN.search(text))

    def _titleIn(self, text: str) -> str:
        """The book named inside a request, if one is."""
        match = TITLED_PATTERN.match(text)
        if match is None:
            return ""

        title = match.group("title").strip()
        # "read me a book" names nothing; "read me harry potter" does.
        if NOT_A_TITLE.match(title):
            return ""
        # Strip the framing that survives the pattern. "The text of harry
        # potter" names the same book as "harry potter"; the words asking for
        # the written version choose a shelf and are not part of the title.
        title = re.sub(r"^(me\s+|a\s+|an\s+|the\s+book\s+|the\s+story\s+)", "", title, flags=re.I)
        title = TEXT_HINT_PATTERN.sub(" ", title)
        title = re.sub(
            r"\b(please|for me|to me|out loud|aloud|version|file|of|the)\b",
            " ",
            title,
            flags=re.I,
        )
        return re.sub(r"\s{2,}", " ", title).strip()

    def _askWhichBook(self) -> Response:
        self._waiting = Waiting.book
        offered = self._library.suggestions(3)
        if not offered:
            self._waiting = Waiting.nothing
            return Response(
                text=(
                    "I don't have any books yet. Put audiobooks in the library "
                    "folder and I'll read them."
                ),
                handledBy=f"{self.name}:empty",
            )

        return Response(
            text=(
                "Sure, what book would you like me to read? "
                f"I have {_list(book.title for book in offered)}, and others."
            ),
            handledBy=f"{self.name}:asking",
        )

    # --- Choosing ---------------------------------------------------------

    async def _continueConversation(self, text: str) -> Response | None:
        """Answering a question this handler asked."""
        if CANCEL_PATTERN.match(text):
            self._reset()
            return Response(text="All right, maybe later.", handledBy=f"{self.name}:cancelled")

        if self._waiting in {Waiting.book, Waiting.which}:
            return await self._offer(text)

        if self._waiting is Waiting.chapter:
            if RESUME_PATTERN.match(text):
                book = self._book
                return await self._resume(book.title if book else "")
            return await self._startChapter(text)

        return None

    async def _offer(self, title: str, *, preferText: bool = False) -> Response:
        """Resolve a spoken title and ask what to do with it."""
        match = self._library.find(title, preferText=preferText)

        if match.ambiguous:
            self._waiting = Waiting.which
            self._candidates = match.alternatives
            shown = match.alternatives[:4]
            return Response(
                text=(
                    f"I have {len(match.alternatives)} of those. Did you mean "
                    f"{_list(_distinguish(book, match.alternatives) for book in shown)}?"
                ),
                handledBy=f"{self.name}:ambiguous",
            )

        if not match.found:
            return self._notFound(title)

        self._book = match.book
        # A book with one chapter has nothing to ask about.
        if len(match.book.storyChapters) <= 1:
            return await self._startChapter("from the beginning")

        self._waiting = Waiting.chapter
        mark = self._bookmarks.forTitle(match.book.title)
        if mark is not None and mark.isWorthResuming:
            return Response(
                text=(
                    f"I have {match.book.title}. You were on {mark.describe()}. "
                    "Carry on, a chapter, or start from the beginning?"
                ),
                handledBy=f"{self.name}:chapter",
            )

        return Response(
            text=(
                f"I have {match.book.title}. Which chapter, "
                "or shall I start from the beginning?"
            ),
            handledBy=f"{self.name}:chapter",
        )

    def _notFound(self, title: str) -> Response:
        """Say what is missing, and offer something that is not."""
        self._reset()

        unreadable = self._library.unreadableMatch(title)
        if unreadable is not None:
            return Response(
                text=(
                    f"I have {spokenName(unreadable)} but only as "
                    f"{unreadable.suffix.lstrip('.')}, "
                    "which I can't read yet. I can do plain text and audiobooks."
                ),
                handledBy=f"{self.name}:unsupported",
            )

        offered = self._library.suggestions(2)
        suffix = f" I do have {_list(book.title for book in offered)}." if offered else ""
        return Response(
            text=f"I couldn't find {title} in the library.{suffix}",
            handledBy=f"{self.name}:missing",
        )

    async def _startChapter(self, spoken: str) -> Response:
        """Begin reading, from the beginning or at a named chapter."""
        book = self._book
        if book is None:  # pragma: no cover - guarded by the state machine
            self._reset()
            return self._askWhichBook()

        if BEGINNING_PATTERN.search(spoken) or not spoken.strip():
            chapter = book.storyChapters[0]
        else:
            found = book.findChapter(spoken)
            if found is None:
                return Response(
                    text=(
                        f"I couldn't find that chapter. {book.title} has "
                        f"{len(book.storyChapters)}. Which one, or from the beginning?"
                    ),
                    handledBy=f"{self.name}:noChapter",
                )
            chapter = found

        self._reset()
        await self._reader.begin(book, chapter)

        source = "" if book.isAudio else ", reading it myself"
        return Response(
            text=f"{chapter.title}, from {book.title}{source}. Say stop when you've had enough.",
            handledBy=f"{self.name}:reading",
        )

    # --- While a book is playing ------------------------------------------

    async def _commandWhileReading(self, text: str) -> Response | None:
        """The few things worth acting on mid-chapter, or None to ignore.

        Deliberately a short list. Whatever the recogniser makes of a narrator
        reading Harry Potter, it will not be one of these, and the cost of
        being wrong in the other direction --- talking over the book, or
        stopping it --- is much higher than the cost of ignoring a question
        somebody will simply ask again.
        """
        if NEXT_CHAPTER_PATTERN.match(text):
            return await self._moveChapter(+1)
        if PREVIOUS_CHAPTER_PATTERN.match(text):
            return await self._moveChapter(-1)
        if WHAT_IS_THIS_PATTERN.match(text):
            reading = self._reader.reading
            if reading is None:  # pragma: no cover - reading was just checked
                return None
            return Response(
                text=f"{reading.chapter.title}, from {reading.book.title}.",
                handledBy=f"{self.name}:whatIsThis",
            )
        return None

    async def _moveChapter(self, step: int) -> Response:
        """Skip forward or back within the book being read."""
        reading = self._reader.reading
        if reading is None:  # pragma: no cover - reading was just checked
            return Response(text="", handledBy=f"{self.name}:ignored")

        chapters = reading.book.storyChapters
        try:
            position = chapters.index(reading.chapter)
        except ValueError:
            position = 0

        wanted = position + step
        if not 0 <= wanted < len(chapters):
            edge = "the end" if step > 0 else "the beginning"
            return Response(
                text=f"That's {edge} of {reading.book.title}.",
                handledBy=f"{self.name}:edge",
            )

        chapter = chapters[wanted]
        await self._reader.begin(reading.book, chapter)
        return Response(text=chapter.title, handledBy=f"{self.name}:reading")

    # --- Stopping ---------------------------------------------------------

    async def _stop(self) -> Response:
        reading = self._reader.reading
        wasReading = self._reader.isReading
        await self._reader.stop()
        self._reset()

        if not wasReading:
            return Response(text="All right.", handledBy=f"{self.name}:cancelled")

        # Recorded before saying anything, so that a crash between the two
        # loses the words rather than the place.
        if reading is not None:
            self._bookmarks.save(
                Bookmark(
                    title=reading.book.title,
                    chapterNumber=reading.chapter.number,
                    chapterTitle=reading.chapter.title,
                    offsetSeconds=reading.offsetSeconds,
                    sentenceIndex=reading.sentenceIndex,
                )
            )

        where = f" You were on {reading.chapter.title}." if reading else ""
        return Response(text=f"Stopped.{where}", handledBy=f"{self.name}:stopped")

    # --- Carrying on ------------------------------------------------------

    async def _resume(self, title: str = "") -> Response:
        """Pick a book up where it was left off.

        A bare "carry on" means the last book listened to, which is almost
        always what is meant. Naming one resumes that instead.
        """
        mark = (
            self._bookmarks.forTitle(title) if title else self._bookmarks.mostRecent()
        )
        if mark is None:
            if title:
                return self._notFound(title)
            return Response(
                text="I don't have a book to carry on with. Ask me to read you one.",
                handledBy=f"{self.name}:nothingToResume",
            )

        match = self._library.find(mark.title)
        book = match.book or (match.alternatives[0] if match.alternatives else None)
        if book is None:
            # The library has changed under the bookmark.
            self._bookmarks.clear(mark.title)
            return Response(
                text=f"I can't find {mark.title} any more.",
                handledBy=f"{self.name}:missing",
            )

        chapter = book.chapterNumbered(mark.chapterNumber) or book.storyChapters[0]
        self._reset()

        if not mark.isWorthResuming:
            # Barely started, so the join would be more confusing than useful.
            await self._reader.begin(book, chapter)
            return Response(
                text=f"{chapter.title}, from the beginning.",
                handledBy=f"{self.name}:reading",
            )

        await self._reader.begin(
            book,
            chapter,
            startSeconds=mark.resumeSeconds,
            startSentence=mark.sentenceIndex,
        )
        return Response(
            text=f"Carrying on with {book.title}, {mark.describe()}.",
            handledBy=f"{self.name}:resumed",
        )

    def _reset(self) -> None:
        self._waiting = Waiting.nothing
        self._book = None
        self._candidates = ()

    def describe(self) -> str:
        return f"books ({self._library.describe()}); {self._base.describe()}"


def _list(items) -> str:
    """Join names the way somebody would say them aloud."""
    values = [str(item) for item in items]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return f"{', '.join(values[:-1])} or {values[-1]}"


def _distinguish(book: Book, among: tuple[Book, ...]) -> str:
    """The part of a title that tells it apart from its siblings.

    Reading seven titles that all begin "Harry Potter and the" back at
    somebody is useless. The shared opening is dropped.
    """
    if len(among) < 2:
        return book.title

    words = [title.split() for title in (candidate.title for candidate in among)]
    shared = 0
    for position in range(min(len(item) for item in words)):
        if len({item[position] for item in words}) > 1:
            break
        shared = position + 1

    remainder = " ".join(book.title.split()[shared:])
    return remainder or book.title


__all__ = ["BookHandler", "Waiting"]
