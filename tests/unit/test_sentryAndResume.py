"""Standing down on goodbye, and carrying a book on from where it stopped."""

from __future__ import annotations

import asyncio
import json

import pytest

from app.assistant.bookHandler import BookHandler
from app.assistant.conversation import Conversation
from app.assistant.handlers import EchoHandler, RuleHandler, createResponseHandler
from app.assistant.sentryHandler import SentryHandler
from app.audio.output import MemoryAudioOutput
from app.books.bookmarks import Bookmark, Bookmarks
from app.books.library import BookLibrary
from app.books.reader import BookReader
from app.config import Settings


async def say(handler, text: str) -> str:
    return (await handler.respond(text, Conversation())).text


# --- Sentry mode ---------------------------------------------------------


@pytest.fixture
def sentry():
    return SentryHandler(EchoHandler())


class TestStandingDown:
    @pytest.mark.parametrize(
        "farewell",
        ["goodbye", "good bye", "bye", "good night", "see you later",
         "that's all", "stand down", "okay goodbye", "goodbye then"],
    )
    async def testAFarewellStandsItDown(self, sentry, farewell):
        await say(sentry, farewell)

        assert sentry.isDormant

    async def testItSaysHowToComeBack(self, sentry):
        assert "hello" in (await say(sentry, "goodbye")).lower()

    async def testAFarewellInsideASentenceDoesNotCount(self, sentry):
        """A book read aloud says goodbye rather a lot."""
        await say(sentry, "he said goodbye to her at the station")

        assert not sentry.isDormant

    async def testQuitIsNotAFarewell(self, sentry):
        """Closing the program is a different thing, and the rules handler
        already answers it."""
        await say(sentry, "quit")

        assert not sentry.isDormant


class TestWhileDormant:
    async def testNothingIsAnswered(self, sentry):
        await say(sentry, "goodbye")

        assert await say(sentry, "what is the time") == ""

    async def testItIsSilentRatherThanRefusing(self, sentry):
        """Answering "I am asleep" every time somebody speaks in the room is
        not being dismissed at all."""
        await say(sentry, "goodbye")

        response = await sentry.respond("turn on the lights", Conversation())

        assert response.isEmpty
        assert response.handledBy.endswith("dormant")

    async def testIgnoredUtterancesAreCounted(self, sentry):
        await say(sentry, "goodbye")
        await say(sentry, "one")
        await say(sentry, "two")

        assert sentry.ignored == 2

    async def testItNeverSpeaksFirstWhileDormant(self):
        """Toddler mode fills silences; a dismissed assistant must not."""

        class Chatty(EchoHandler):
            @property
            def promptAfterSeconds(self):
                return 7.0

            async def promptWhenSilent(self):
                from app.assistant.handlers import Response

                return Response(text="still there?")

        handler = SentryHandler(Chatty())
        assert handler.promptAfterSeconds == 7.0

        await say(handler, "goodbye")

        assert handler.promptAfterSeconds is None
        assert await handler.promptWhenSilent() is None


class TestWakingUp:
    @pytest.mark.parametrize(
        "greeting",
        ["hello", "hi", "hey", "good morning", "wake up", "are you there",
         "hello there", "come back"],
    )
    async def testAGreetingWakesIt(self, sentry, greeting):
        await say(sentry, "goodbye")

        await say(sentry, greeting)

        assert not sentry.isDormant

    async def testAGreetingInsideASentenceDoesNotWakeIt(self, sentry):
        await say(sentry, "goodbye")

        await say(sentry, "she said hello to the man in the hat")

        assert sentry.isDormant

    async def testItAnswersAgainOnceAwake(self, sentry):
        await say(sentry, "goodbye")
        await say(sentry, "hello")

        assert (await say(sentry, "what is the time")).startswith("You said:")

    async def testWakingCanBeSilent(self):
        handler = SentryHandler(EchoHandler(), greetOnWaking=False)
        await say(handler, "goodbye")

        assert await say(handler, "hello") == ""
        assert not handler.isDormant

    async def testItCanStartDormant(self):
        handler = SentryHandler(EchoHandler(), dormant=True)

        assert await say(handler, "what is the time") == ""
        await say(handler, "hello")
        assert not handler.isDormant


class TestSentryStopsABook:
    async def testGoodbyeStopsWhateverIsPlaying(self, tmp_path):
        """A book carrying on happily after goodbye is not what anyone means."""
        audio = tmp_path / "Books"
        folder = audio / "Digital Fortress [B0EXAMPLE1]"
        folder.mkdir(parents=True)
        for index in (1, 2):
            name = f"Digital Fortress [B0EXAMPLE1] - {index:02d} - Chapter {index}.mp3"
            (folder / name).write_bytes(b"")
        library = BookLibrary(audio, tmp_path / "TextBooks")
        books = BookHandler(
            RuleHandler(), library, BookReader(MemoryAudioOutput()),
            Bookmarks(tmp_path / "marks.json"),
        )
        handler = SentryHandler(books)

        await say(handler, "read me digital fortress")
        await say(handler, "from the beginning")
        assert books.reader.isReading

        await say(handler, "goodbye")

        assert handler.isDormant
        assert not books.reader.isReading


class TestSentryConfiguration:
    def testItIsOutermost(self, tmp_path):
        """Standing down has to mean everything stops answering, not
        everything except whichever mode is active."""
        settings = Settings(
            assistant={"handler": "rules"},
            books={"audioDirectory": tmp_path, "textDirectory": tmp_path},
        )

        handler = createResponseHandler(settings, audioOutput=MemoryAudioOutput())

        assert isinstance(handler, SentryHandler)

    def testItCanBeTurnedOff(self, tmp_path):
        settings = Settings(assistant={"handler": "rules"}, sentry={"enabled": False})

        handler = createResponseHandler(settings, audioOutput=MemoryAudioOutput())

        assert not isinstance(handler, SentryHandler)

    def testTheWrappedHandlerStaysReachable(self, tmp_path):
        settings = Settings(
            assistant={"handler": "rules"},
            books={"audioDirectory": tmp_path, "textDirectory": tmp_path},
        )

        handler = createResponseHandler(settings, audioOutput=MemoryAudioOutput())

        assert getattr(handler, "reader", None) is not None


# --- Bookmarks and resume ------------------------------------------------


class TestBookmarks:
    def testAPositionSurvivesBeingWrittenAndRead(self, tmp_path):
        """Resume is only worth having if it survives a restart."""
        path = tmp_path / "marks.json"
        Bookmarks(path).save(
            Bookmark(title="Digital Fortress", chapterNumber=2,
                     chapterTitle="Chapter 2", offsetSeconds=930.5)
        )

        recovered = Bookmarks(path).forTitle("Digital Fortress")

        assert recovered.chapterNumber == 2
        assert recovered.offsetSeconds == pytest.approx(930.5)

    def testCaseAndSpacingDoNotCreateASecondEntry(self, tmp_path):
        marks = Bookmarks(tmp_path / "marks.json")
        marks.save(Bookmark(title="Digital Fortress", chapterNumber=1, chapterTitle="one"))

        assert marks.forTitle("digital  fortress") is not None

    def testOneEntryPerBook(self, tmp_path):
        marks = Bookmarks(tmp_path / "marks.json")
        marks.save(Bookmark(title="A Book", chapterNumber=1, chapterTitle="one"))
        marks.save(Bookmark(title="A Book", chapterNumber=5, chapterTitle="five"))

        assert len(marks) == 1
        assert marks.forTitle("A Book").chapterNumber == 5

    def testResumingRewindsALittle(self, tmp_path):
        """So the join sounds deliberate rather than clipped."""
        mark = Bookmark(title="x", chapterNumber=1, chapterTitle="one", offsetSeconds=100.0)

        assert mark.resumeSeconds == pytest.approx(97.0)

    def testItNeverRewindsPastTheStart(self, tmp_path):
        mark = Bookmark(title="x", chapterNumber=1, chapterTitle="one", offsetSeconds=1.0)

        assert mark.resumeSeconds == 0.0

    def testBarelyStartedIsNotWorthResuming(self, tmp_path):
        assert not Bookmark(title="x", chapterNumber=1, chapterTitle="o",
                            offsetSeconds=4.0).isWorthResuming
        assert Bookmark(title="x", chapterNumber=1, chapterTitle="o",
                        offsetSeconds=400.0).isWorthResuming

    def testACorruptFileIsIgnoredRatherThanFatal(self, tmp_path):
        path = tmp_path / "marks.json"
        path.write_text("{ not json", encoding="utf-8")

        assert Bookmarks(path).mostRecent() is None

    def testAnUnwritableLocationIsNotFatal(self, tmp_path):
        """Losing a bookmark is minor; refusing to play a book is not."""
        marks = Bookmarks(tmp_path / "a-file" / "nested" / "marks.json")
        (tmp_path / "a-file").write_text("in the way", encoding="utf-8")

        marks.save(Bookmark(title="x", chapterNumber=1, chapterTitle="one"))

        assert marks.forTitle("x") is not None


@pytest.fixture
def bookHandler(tmp_path):
    audio, text = tmp_path / "Books", tmp_path / "TextBooks"
    folder = audio / "Digital Fortress [B0EXAMPLE1]"
    folder.mkdir(parents=True)
    for index in (1, 2, 3):
        name = f"Digital Fortress [B0EXAMPLE1] - {index:02d} - Chapter {index}.mp3"
        (folder / name).write_bytes(b"")
    text.mkdir()
    return BookHandler(
        EchoHandler(),
        BookLibrary(audio, text),
        BookReader(MemoryAudioOutput()),
        Bookmarks(tmp_path / "marks.json"),
    )


class TestResuming:
    async def testStoppingRecordsWhereYouWere(self, bookHandler, tmp_path):
        await say(bookHandler, "read me digital fortress")
        await say(bookHandler, "from the beginning")
        await say(bookHandler, "stop audiobook")

        assert json.loads((tmp_path / "marks.json").read_text())

    async def testNothingToResumeIsSaidPlainly(self, bookHandler):
        reply = await say(bookHandler, "carry on")

        assert "don't have a book to carry on with" in reply

    @pytest.mark.parametrize(
        "spoken", ["resume", "continue", "carry on", "keep going", "resume the book"]
    )
    async def testSeveralWaysOfAskingToCarryOn(self, bookHandler, spoken):
        bookHandler._bookmarks.save(
            Bookmark(title="Digital Fortress", chapterNumber=2,
                     chapterTitle="Chapter 2", offsetSeconds=600.0)
        )

        assert "Carrying on" in await say(bookHandler, spoken)

    async def testItResumesTheRightChapter(self, bookHandler):
        bookHandler._bookmarks.save(
            Bookmark(title="Digital Fortress", chapterNumber=3,
                     chapterTitle="Chapter 3", offsetSeconds=600.0)
        )

        await say(bookHandler, "carry on")

        assert bookHandler.reader.reading.chapter.number == 3

    async def testItStartsPartWayIn(self, bookHandler):
        bookHandler._bookmarks.save(
            Bookmark(title="Digital Fortress", chapterNumber=2,
                     chapterTitle="Chapter 2", offsetSeconds=600.0)
        )

        await say(bookHandler, "carry on")

        assert bookHandler.reader.reading.startedFrom == pytest.approx(597.0)

    async def testBarelyStartedGoesBackToTheBeginning(self, bookHandler):
        bookHandler._bookmarks.save(
            Bookmark(title="Digital Fortress", chapterNumber=2,
                     chapterTitle="Chapter 2", offsetSeconds=2.0)
        )

        reply = await say(bookHandler, "carry on")

        assert "from the beginning" in reply
        assert bookHandler.reader.reading.startedFrom == 0.0

    async def testTheChapterQuestionOffersToCarryOn(self, bookHandler):
        bookHandler._bookmarks.save(
            Bookmark(title="Digital Fortress", chapterNumber=2,
                     chapterTitle="Chapter 2", offsetSeconds=600.0)
        )

        reply = await say(bookHandler, "read me digital fortress")

        assert "Carry on" in reply
        assert "10 minute" in reply

    async def testCarryOnAnswersThatQuestion(self, bookHandler):
        bookHandler._bookmarks.save(
            Bookmark(title="Digital Fortress", chapterNumber=2,
                     chapterTitle="Chapter 2", offsetSeconds=600.0)
        )
        await say(bookHandler, "read me digital fortress")

        reply = await say(bookHandler, "carry on")

        assert "Carrying on" in reply

    async def testABookmarkForAVanishedBookIsForgotten(self, bookHandler):
        bookHandler._bookmarks.save(
            Bookmark(title="Some Deleted Book", chapterNumber=1,
                     chapterTitle="one", offsetSeconds=600.0)
        )

        reply = await say(bookHandler, "carry on")

        assert "can't find" in reply
        assert bookHandler._bookmarks.forTitle("Some Deleted Book") is None


class TestResumingTextBooks:
    async def testATextBookResumesAtASentence(self, tmp_path):
        """A text book is read in sentences, so seconds would be a guess."""
        audio, text = tmp_path / "Books", tmp_path / "TextBooks"
        audio.mkdir()
        text.mkdir()
        (text / "aesop.txt").write_text(
            "CHAPTER ONE\nTHE FOX\n" + " ".join(f"Sentence number {n}." for n in range(30)),
            encoding="utf-8",
        )
        from app.speech.textToSpeech import createTextToSpeechProvider

        tts = createTextToSpeechProvider(Settings(textToSpeech={"provider": "tone"}))
        await tts.load()
        reader = BookReader(MemoryAudioOutput(), textToSpeech=tts)
        library = BookLibrary(audio, text)
        book = library.textBooks()[0]

        await reader.begin(book, book.chapters[0], startSentence=5)
        await asyncio.sleep(0.4)
        await reader.stop()

        assert reader.reading is None or reader.reading.sentenceIndex >= 5
