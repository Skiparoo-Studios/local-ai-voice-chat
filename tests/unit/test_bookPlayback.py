"""Two bugs found by using the thing, and the fixes for them.

The first was serious: asking for a book played it for a few seconds, the
audio broke up, and afterwards nothing played at all. The cause was the
assistant answering the book. Continuous listening heard the narrator, ran a
turn, and the reply opened the speaker at the assistant's sample rate --- which
closed the stream the book was already writing to. Two writers, one device.

The second was that a text book could not be reached by voice.
"""

from __future__ import annotations

import asyncio

import pytest

from app.assistant.bookHandler import BookHandler
from app.assistant.conversation import Conversation
from app.assistant.handlers import EchoHandler
from app.audio.audioBuffer import AudioBuffer
from app.audio.output import MemoryAudioOutput
from app.books.library import BookLibrary
from app.books.reader import BookReader


@pytest.fixture
def library(tmp_path):
    audio, text = tmp_path / "Books", tmp_path / "TextBooks"
    audio.mkdir()
    text.mkdir()

    folder = audio / "Digital Fortress [B03]"
    folder.mkdir()
    for index, title in enumerate(["Chapter 1", "Chapter 2", "Chapter 3"], start=1):
        (folder / f"Digital Fortress [B03] - {index:02d} - {title}.mp3").write_bytes(b"")

    # Named the way a file on disk actually is, with no space in it.
    (text / "harrypotter.txt").write_text(
        "CHAPTER ONE\nTHE BOY WHO LIVED\nMr Dursley was perfectly normal.\n\n"
        "CHAPTER TWO\nTHE VANISHING GLASS\nNearly ten years had passed.\n",
        encoding="utf-8",
    )
    return BookLibrary(audio, text)


@pytest.fixture
def handler(library):
    return BookHandler(EchoHandler(), library, BookReader(MemoryAudioOutput()))


async def say(handler: BookHandler, text: str) -> str:
    return (await handler.respond(text, Conversation())).text


async def startReading(handler: BookHandler) -> None:
    await say(handler, "read me digital fortress")
    await say(handler, "from the beginning")
    assert handler.reader.isReading


class TestTheAssistantDoesNotAnswerTheBook:
    """The bug: the narrator was heard, answered, and the reply tore down the
    stream the book was playing through."""

    async def testSpeechHeardWhileReadingIsIgnored(self, handler):
        await startReading(handler)

        # What the recogniser makes of a narrator mid-chapter.
        reply = await say(handler, "the fog was thick over the compound that night")

        assert reply == ""

    async def testAnIgnoredTurnSaysNothingAtAll(self, handler):
        """Empty is the point: the runtime speaks nothing, so there is never a
        second writer to the speaker."""
        await startReading(handler)

        response = await handler.respond("he turned the page slowly", Conversation())

        assert response.isEmpty
        assert response.handledBy.endswith("ignored")

    async def testTheBookKeepsPlayingThrough(self, handler):
        await startReading(handler)

        await say(handler, "some narration that is not a command")

        assert handler.reader.isReading

    async def testAQuestionIsIgnoredRatherThanAnswered(self, handler):
        """A cost of the fix, and the right trade: being talked over is worse
        than having to ask again."""
        await startReading(handler)

        assert await say(handler, "what is the time") == ""

    async def testNothingIsIgnoredWhenNoBookIsPlaying(self, handler):
        assert (await say(handler, "what is the time")).startswith("You said:")


class TestCommandsThatStillWorkWhileReading:
    async def testStoppingTheBookWorks(self, handler):
        """The whole point of having a stop command."""
        await startReading(handler)

        reply = await say(handler, "stop audiobook")

        assert "Stopped" in reply
        assert not handler.reader.isReading

    @pytest.mark.parametrize(
        "spoken",
        ["stop audiobook", "stop the audiobook", "stop audio book", "stop the book",
         "stop reading", "pause the book", "stop it", "turn off the audiobook"],
    )
    async def testSeveralWaysOfStoppingTheBook(self, handler, spoken):
        await startReading(handler)

        await say(handler, spoken)

        assert not handler.reader.isReading

    @pytest.mark.parametrize("spoken", ["stop", "quiet", "enough", "silence"])
    async def testABareStopNoLongerEndsTheBook(self, handler, spoken):
        """A narrator says "stop" often enough that a bare one ending the
        chapter is worse than having to say two words. The microphone is full
        of narration while a book plays."""
        await startReading(handler)

        assert await say(handler, spoken) == ""
        assert handler.reader.isReading

    async def testNextChapter(self, handler):
        await startReading(handler)
        first = handler.reader.reading.chapter

        reply = await say(handler, "next chapter")

        assert handler.reader.reading.chapter != first
        assert "Chapter 2" in reply

    async def testPreviousChapter(self, handler):
        await startReading(handler)
        await say(handler, "next chapter")

        reply = await say(handler, "go back")

        assert "Chapter 1" in reply

    async def testTheEndOfTheBookIsReportedNotWrappedAround(self, handler):
        await startReading(handler)
        await say(handler, "next")
        await say(handler, "next")

        reply = await say(handler, "next")

        assert "the end" in reply

    async def testAskingWhatIsPlaying(self, handler):
        await startReading(handler)

        reply = await say(handler, "what is this")

        assert "Digital Fortress" in reply

    async def testANarratorSayingSomethingLongIsNotACommand(self, handler):
        """The command patterns are anchored, so prose containing the word
        cannot trigger them."""
        await startReading(handler)

        await say(handler, "she told him to stop the car immediately")

        assert handler.reader.isReading


class TestReachingTheTextLibrary:
    """The bug: a file called harrypotter.txt could not be asked for, because
    "harry potter" is two words and the title is one."""

    def testAJoinedTitleMatchesSpokenWords(self, library):
        assert library.find("harrypotter").book is not None
        assert library.find("harry potter", preferText=True).book.title == "harrypotter"

    def testOneWordIsNotEnoughToMatchAJoinedTitle(self, library):
        """Otherwise "harry" alone would match, and so would half of anything."""
        assert library.find("harry", preferText=True).book is None

    async def testAskingForTheTextVersionReachesIt(self, handler):
        reply = await say(handler, "read me the text of harry potter")

        assert "harrypotter" in reply

    @pytest.mark.parametrize(
        "spoken",
        ["read me the text of harry potter",
         "read me the harry potter text file",
         "read the written version of harry potter"],
    )
    async def testSeveralWaysOfAskingForText(self, handler, spoken):
        assert "harrypotter" in await say(handler, spoken)

    async def testRecordingsStillWinWhenNoTextIsAsked(self, handler):
        """A professional narrator beats the assistant reading aloud."""
        reply = await say(handler, "read me digital fortress")

        assert "Digital Fortress" in reply
        assert "harrypotter" not in reply

    def testTheWordsAskingForTextAreNotPartOfTheTitle(self, handler):
        assert handler._titleIn("read me the text of harry potter") == "harry potter"


class TestOneWriterAtATime:
    """Defence in depth. Even with the handler silent, nothing should be able
    to close the stream another writer is using."""

    async def testConcurrentWritesAtDifferentRatesAreSerialised(self):
        from app.audio.output import SoundDeviceOutput

        events: list[str] = []

        class FakeStream:
            def __init__(self, samplerate, **kwargs):
                self.rate = samplerate
                events.append(f"open{samplerate}")

            def start(self):
                pass

            def write(self, samples):
                events.append(f"write{self.rate}")

            def stop(self):
                pass

            def close(self):
                events.append(f"close{self.rate}")

        class FakeSoundDevice:
            def OutputStream(self, samplerate, **kwargs):
                return FakeStream(samplerate, **kwargs)

        import numpy

        output = SoundDeviceOutput.__new__(SoundDeviceOutput)
        output._device = None
        output._soundDevice = FakeSoundDevice()
        output._numpy = numpy
        output._stream = None
        output._format = None
        # Deliberately not assigning a lock: the class holds a reentrant one,
        # and a plain Lock here deadlocks, because _writeBlocking holds it
        # while _ensureStream closes and reopens the stream.

        book = AudioBuffer(data=b"\x00\x00" * 500, sampleRate=44100)
        reply = AudioBuffer(data=b"\x00\x00" * 500, sampleRate=24000)

        await asyncio.gather(
            *(output.play(book) for _ in range(4)),
            *(output.play(reply) for _ in range(4)),
        )

        # A write must never land between another writer's open and close of a
        # different rate --- that is the corruption. Serialised, every write
        # matches the stream open immediately before it.
        current = None
        for event in events:
            if event.startswith("open"):
                current = event[len("open") :]
            elif event.startswith("write"):
                assert event[len("write") :] == current, events
