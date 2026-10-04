"""Asking to be read to, and being read to.

The library has its own tests; these are about the conversation and the
playback, driven through fixtures so they need no real books.
"""

from __future__ import annotations

import asyncio

import pytest

from app.assistant.bookHandler import BookHandler
from app.assistant.conversation import Conversation
from app.assistant.handlers import EchoHandler
from app.audio.audioBuffer import AudioBuffer
from app.audio.output import AudioOutput, MemoryAudioOutput
from app.books.library import BookLibrary
from app.books.reader import BookReader


@pytest.fixture
def library(tmp_path):
    audio, text = tmp_path / "Books", tmp_path / "TextBooks"
    audio.mkdir()
    text.mkdir()

    def book(folder: str, parts: list[str]) -> None:
        directory = audio / folder
        directory.mkdir()
        for index, title in enumerate(parts, start=1):
            (directory / f"{folder} - {index:02d} - {title}.mp3").write_bytes(b"")

    book("Harry Potter and the Chamber of Secrets [B01]",
         ["Opening Credits", "Chapter 1_ The Worst Birthday", "Chapter 2_ Dobby"])
    book("Harry Potter and the Goblet of Fire [B02]", ["Chapter 1_ The Riddle House"])
    book("Digital Fortress [B03]", ["Chapter 1", "Chapter 2"])
    (text / "aesop.txt").write_text(
        "CHAPTER ONE\nTHE FOX\nA fox once saw some grapes.\n\n"
        "CHAPTER TWO\nTHE CROW\nA crow sat in a tree.\n",
        encoding="utf-8",
    )
    (text / "The Hobbit - Tolkien, J. R. R._5022.epub").write_bytes(b"nope")
    return BookLibrary(audio, text)


@pytest.fixture
def handler(library):
    return BookHandler(EchoHandler(), library, BookReader(MemoryAudioOutput()))


async def say(handler: BookHandler, text: str) -> str:
    return (await handler.respond(text, Conversation())).text


class TestBeingAsked:
    @pytest.mark.parametrize(
        "request_",
        ["read me a story", "read to me", "tell me a story", "read me a book",
         "listen to a book", "play me an audiobook"],
    )
    async def testARequestWithNoTitleAsksWhichBook(self, handler, request_):
        reply = await say(handler, request_)

        assert "what book would you like" in reply.lower()

    async def testItOffersSomeTitles(self, handler):
        reply = await say(handler, "read me a story")

        assert "Digital Fortress" in reply or "Harry Potter" in reply

    async def testAnEmptyLibrarySaysSoRatherThanAsking(self, tmp_path):
        empty = BookHandler(
            EchoHandler(),
            BookLibrary(tmp_path / "none", tmp_path / "none"),
            BookReader(MemoryAudioOutput()),
        )

        assert "don't have any books" in await say(empty, "read me a story")


class TestChoosingABook:
    async def testAnAmbiguousTitleAsksWhichOne(self, handler):
        await say(handler, "read me a story")

        reply = await say(handler, "harry potter")

        assert "2 of those" in reply

    async def testTheSharedPartOfTitlesIsNotReadBack(self, handler):
        """Repeating "Harry Potter and the" four times is no help to anybody."""
        await say(handler, "read me a story")

        reply = await say(handler, "harry potter")

        assert reply.count("Harry Potter and the") == 0
        assert "Chamber of Secrets" in reply

    async def testNarrowingDownGetsToTheChapterQuestion(self, handler):
        await say(handler, "read me a story")
        await say(handler, "harry potter")

        reply = await say(handler, "chamber of secrets")

        assert "Which chapter" in reply

    async def testATitleInTheFirstBreathSkipsAhead(self, handler):
        reply = await say(handler, "read me digital fortress")

        assert "Which chapter" in reply

    async def testABookWithOneChapterJustStarts(self, handler):
        reply = await say(handler, "read me the goblet of fire")

        assert "Which chapter" not in reply
        assert handler.reader.isReading

    async def testAMissingBookIsSaidPlainly(self, handler):
        reply = await say(handler, "read me moby dick")

        assert "couldn't find" in reply

    async def testAnUnreadableFormatIsExplained(self, handler):
        """The library really does contain an epub, and saying so beats
        pretending the book is not there."""
        reply = await say(handler, "read me the hobbit")

        assert "epub" in reply
        assert "The Hobbit" in reply
        assert "Tolkien" not in reply

    async def testCancellingLetsGo(self, handler):
        await say(handler, "read me a story")

        assert "later" in await say(handler, "never mind")


class TestChoosingAChapter:
    async def testFromTheBeginningStartsAtTheFirstStoryChapter(self, handler):
        await say(handler, "read me the chamber of secrets")

        reply = await say(handler, "from the beginning")

        assert "The Worst Birthday" in reply
        assert "Opening Credits" not in reply

    async def testANamedChapter(self, handler):
        await say(handler, "read me the chamber of secrets")

        assert "Dobby" in await say(handler, "chapter two")

    async def testAChapterThatIsNotThereAsksAgain(self, handler):
        await say(handler, "read me the chamber of secrets")

        reply = await say(handler, "chapter ninety")

        assert "couldn't find that chapter" in reply
        assert handler.reader.isReading is False

    async def testItSaysHowToStop(self, handler):
        await say(handler, "read me the chamber of secrets")

        assert "stop" in (await say(handler, "from the beginning")).lower()


class TestStopping:
    async def testStopEndsTheReading(self, handler):
        await say(handler, "read me the chamber of secrets")
        await say(handler, "from the beginning")
        assert handler.reader.isReading

        reply = await say(handler, "stop audiobook")

        assert "Stopped" in reply
        assert not handler.reader.isReading

    async def testStopIsIgnoredWhenNothingIsHappening(self, handler):
        """Otherwise the router could never act on "stop" itself."""
        reply = await say(handler, "stop audiobook")

        assert reply.startswith("You said:")


class TestDelegation:
    @pytest.mark.parametrize(
        "unrelated",
        ["what is the time", "turn off the kitchen light", "how are you",
         "play the hallway light"],
    )
    async def testEverythingElseReachesTheBaseHandler(self, handler, unrelated):
        assert (await say(handler, unrelated)).startswith("You said:")

    async def testAnOverloadedVerbOnlyCountsWhenThereIsABook(self, handler):
        """"Play" also means music and switches, so a title given to it is
        claimed only when the library can answer."""
        assert (await say(handler, "play the hallway light")).startswith("You said:")
        assert "Which chapter" in await say(handler, "play digital fortress")

    def testTheWrappedHandlerStaysReachable(self, handler):
        """The API asks a handler for its registry; a wrapper must not hide it."""
        assert handler.usesLlm is False
        assert handler.base is not None

    def testItNeverSpeaksUnprompted(self, handler):
        assert handler.promptAfterSeconds is None


class PacedOutput(AudioOutput):
    """Behaves like a speaker: accepting audio takes as long as playing it."""

    def __init__(self, speedUp: float = 200.0) -> None:
        self.played: list[float] = []
        self._speedUp = speedUp

    async def play(self, audio: AudioBuffer) -> None:
        self.played.append(audio.durationSeconds)
        await asyncio.sleep(audio.durationSeconds / self._speedUp)


class TestReadingText:
    """The .txt path, which uses the assistant's own voice."""

    async def testItSynthesisesAndSpeaks(self, library, tmp_path):
        from app.config import Settings
        from app.speech.textToSpeech import createTextToSpeechProvider

        tts = createTextToSpeechProvider(Settings(textToSpeech={"provider": "tone"}))
        await tts.load()
        output = PacedOutput()
        reader = BookReader(output, textToSpeech=tts)
        book = library.textBooks()[0]

        await reader.begin(book, book.chapters[0])
        await reader.wait()

        assert output.played
        assert reader.chaptersRead == 1

    async def testWithoutAVoiceItFailsQuietly(self, library):
        """A missing provider must not raise at whoever asked for the book."""
        reader = BookReader(MemoryAudioOutput(), textToSpeech=None)
        book = library.textBooks()[0]

        await reader.begin(book, book.chapters[0])
        await reader.wait()

        assert reader.chaptersRead == 0


class TestReaderState:
    async def testItWaitsForTheAssistantToStopTalking(self, library):
        """The reply announcing the chapter goes through the same speaker."""
        output = MemoryAudioOutput()
        reader = BookReader(output)
        talking = True
        reader.isBusy = lambda: talking

        book = library.find("digital fortress").book
        await reader.begin(book, book.storyChapters[0])
        await asyncio.sleep(0.25)
        assert output.played == []

        talking = False
        await asyncio.sleep(0.3)
        await reader.stop()

    async def testStartingAgainReplacesWhatWasPlaying(self, library):
        reader = BookReader(MemoryAudioOutput())
        book = library.find("digital fortress").book

        await reader.begin(book, book.chapters[0])
        first = reader.reading
        await reader.begin(book, book.chapters[1])

        assert reader.reading != first
        await reader.stop()

    async def testItDescribesWhatItIsDoing(self, library):
        reader = BookReader(MemoryAudioOutput())
        assert reader.describe() == "not reading"

        book = library.find("digital fortress").book
        await reader.begin(book, book.storyChapters[0])

        assert "reading" in reader.describe()
        await reader.stop()

    async def testAMissingFileIsReportedNotRaised(self, library, tmp_path):
        from app.books.library import Chapter

        reader = BookReader(MemoryAudioOutput())
        book = library.find("digital fortress").book
        absent = Chapter(number=1, title="Gone", path=tmp_path / "nothing.mp3")

        await reader.begin(book, absent)
        await reader.wait()

        assert reader.chaptersRead == 0
