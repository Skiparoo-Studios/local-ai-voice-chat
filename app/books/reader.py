"""Reading a book aloud, for as long as it takes.

Everything else the assistant says is a sentence or two. A chapter is twenty
minutes, and one of the files in this library is seventy-two, so reading is
not a turn --- it is a background activity that has to be interruptible and
must not hold the assistant's turn open while it runs.

Two sources, one path out:

    an audiobook   decoded from disk in blocks and written to the speaker
    a text book    split into sentences, synthesised, and spoken

Audio is decoded in blocks rather than loaded whole. A seventy-two minute
chapter is ninety-five megabytes as sixteen-bit mono, and holding that to play
it would be careless when soundfile will stream it at constant memory.

The reader waits for the assistant to stop talking before it starts, because
the reply announcing the chapter goes through the same speaker. Without that
the announcement and the first sentence of the book would overlap.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.assistant.sentences import splitIntoSentences
from app.audio.audioBuffer import AudioBuffer
from app.audio.output import AudioOutput
from app.books.library import Book, Chapter, TextBook

if TYPE_CHECKING:
    from app.speech.textToSpeech import TextToSpeechProvider

logger = logging.getLogger(__name__)

# Five seconds of audio at a time. Long enough that decoding is negligible,
# short enough that a request to stop is acted on almost at once, since a
# block already written to the device has to finish playing.
BLOCK_SECONDS = 5.0

# How often to check whether the assistant has finished announcing the
# chapter before the book itself starts.
SETTLE_POLL_SECONDS = 0.1

# Giving up rather than talking over the assistant for ever.
SETTLE_TIMEOUT_SECONDS = 30.0

# Blocks decoded ahead of the speaker. Two is enough to keep it fed and small
# enough that stopping wastes almost nothing.
QUEUE_DEPTH = 2

# How long the decoding thread waits for room before checking whether it has
# been told to stop.
HANDOVER_POLL_SECONDS = 0.2


@dataclass(slots=True)
class Reading:
    """What is being read, and how far in.

    Not frozen: the position advances as the audio is written, and a caller
    asking where the book had got to is asking about this object.
    """

    book: Book
    chapter: Chapter
    # Seconds of the chapter played, counting whatever it started from.
    offsetSeconds: float = 0.0
    # Sentences of a text chapter spoken, which is what it is read in.
    sentenceIndex: int = 0
    startedFrom: float = 0.0

    def describe(self) -> str:
        return f"{self.chapter.title} of {self.book.title}"


class BookReader:
    """Plays a chapter of an audiobook, or reads a chapter of text aloud."""

    def __init__(
        self,
        output: AudioOutput,
        *,
        textToSpeech: TextToSpeechProvider | None = None,
        voice: str | None = None,
        language: str | None = "en",
        isBusy: Callable[[], bool] | None = None,
        blockSeconds: float = BLOCK_SECONDS,
        volume: float = 1.0,
    ) -> None:
        self._output = output
        self._tts = textToSpeech
        self._voice = voice
        self._language = language
        # Asked before starting, so the book does not begin over the top of
        # the assistant announcing it. Public and reassignable because the
        # runtime that knows the answer is built after this is, and the reader
        # deliberately does not import it to find out.
        self.isBusy: Callable[[], bool] = isBusy or (lambda: False)
        self._blockSeconds = blockSeconds
        self._volume = max(0.0, min(1.0, volume))

        self._task: asyncio.Task[None] | None = None
        self._reading: Reading | None = None
        self.chaptersRead = 0

    # --- State ------------------------------------------------------------

    @property
    def isReading(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def reading(self) -> Reading | None:
        return self._reading if self.isReading else None

    def describe(self) -> str:
        if self._reading is None or not self.isReading:
            return "not reading"
        return f"reading {self._reading.describe()}"

    # --- Control ----------------------------------------------------------

    async def begin(
        self,
        book: Book,
        chapter: Chapter,
        *,
        startSeconds: float = 0.0,
        startSentence: int = 0,
    ) -> Reading:
        """Start reading, replacing anything already being read.

        ``startSeconds`` and ``startSentence`` carry a bookmark back in, so
        resuming is the same operation as starting with an offset.
        """
        await self.stop()

        reading = Reading(
            book=book,
            chapter=chapter,
            offsetSeconds=startSeconds,
            sentenceIndex=startSentence,
            startedFrom=startSeconds,
        )
        self._reading = reading
        self._task = asyncio.create_task(self._read(reading))
        logger.info("Reading %s", reading.describe())
        return reading

    async def stop(self) -> None:
        """Stop reading now, discarding audio already buffered."""
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        # After the task is gone, so nothing writes to the device afterwards.
        with contextlib.suppress(Exception):
            await self._output.stop()

    async def wait(self) -> None:
        """Wait for the current chapter to finish. Used by tests and the CLI."""
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    # --- Reading ----------------------------------------------------------

    async def _read(self, reading: Reading) -> None:
        try:
            await self._waitUntilQuiet()
            if isinstance(reading.book, TextBook):
                await self._readText(reading.book, reading)
            else:
                await self._playAudio(reading)
            self.chaptersRead += 1
            logger.info("Finished %s", reading.describe())
        except asyncio.CancelledError:
            logger.info("Stopped reading %s", reading.describe())
            raise
        except Exception as error:  # noqa: BLE001 - reported, never raised at a listener
            logger.error("Reading %s failed: %s", reading.describe(), error)

    async def _waitUntilQuiet(self) -> None:
        """Let the assistant finish announcing the chapter."""
        waited = 0.0
        while self.isBusy() and waited < SETTLE_TIMEOUT_SECONDS:
            await asyncio.sleep(SETTLE_POLL_SECONDS)
            waited += SETTLE_POLL_SECONDS

    # --- An audiobook -----------------------------------------------------

    async def _playAudio(self, reading: Reading) -> None:
        """Decode a chapter in blocks and write it to the speaker."""
        chapter = reading.chapter
        if chapter.path is None:
            raise FileNotFoundError("This chapter has no audio file")

        soundfile = _importSoundFile()
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=QUEUE_DEPTH)
        loop = asyncio.get_running_loop()
        finished = object()
        path = chapter.path
        # Told to the decoding thread when nobody is going to read the queue
        # again. Without it the thread waits for room in a queue that is full
        # and has no consumer, which leaks a thread per stopped chapter and
        # leaves the interpreter unable to exit --- measured, and the reason
        # this is not simply a blocking put.
        stopping = threading.Event()
        startSeconds = reading.startedFrom

        def offer(item: Any) -> bool:
            """Hand one item over, giving up if the consumer has gone."""
            future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            while not stopping.is_set():
                try:
                    future.result(timeout=HANDOVER_POLL_SECONDS)
                    return True
                except concurrent.futures.TimeoutError:
                    continue
                except Exception:  # noqa: BLE001 - the loop is gone; so are we
                    return False
            future.cancel()
            return False

        def decode() -> None:
            try:
                with soundfile.SoundFile(str(path)) as handle:
                    rate = handle.samplerate
                    channels = handle.channels
                    frames = max(1, int(rate * self._blockSeconds))
                    if startSeconds > 0:
                        handle.seek(min(int(rate * startSeconds), max(0, len(handle) - 1)))
                    while not stopping.is_set():
                        block = handle.read(frames, dtype="int16", always_2d=True)
                        if not len(block):
                            break
                        if self._volume < 1.0:
                            # Integer arithmetic on the samples: quieter audio
                            # leaves the microphone room to hear a command over
                            # it, which is the whole point of the setting.
                            block = (block * self._volume).astype(block.dtype)
                        audio = AudioBuffer(
                            data=block.tobytes(), sampleRate=rate, channels=channels
                        )
                        if not offer(audio):
                            return
            except Exception as error:  # noqa: BLE001 - handed to the consumer
                offer(error)
            else:
                offer(finished)

        worker = asyncio.create_task(asyncio.to_thread(decode))
        await self._output.beginUtterance()
        try:
            while True:
                item = await queue.get()
                if item is finished:
                    break
                if isinstance(item, Exception):
                    raise item
                await self._output.play(item)
                # Counted after the write returns, so the position reflects
                # what has been heard rather than what has been decoded.
                reading.offsetSeconds += item.durationSeconds
        finally:
            # Set before cancelling, so the thread is already on its way out.
            stopping.set()
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await worker
            await self._output.endUtterance()

    # --- A text book ------------------------------------------------------

    async def _readText(self, book: TextBook, reading: Reading) -> None:
        """Synthesise a chapter a sentence at a time and speak it.

        Sentence by sentence rather than all at once so that stopping is
        prompt and no more is synthesised than gets heard. A chapter is five
        thousand words; generating all of it before saying any would take
        minutes and waste nearly all of it.
        """
        if self._tts is None:
            raise RuntimeError("Reading text aloud needs a text-to-speech provider")

        chapter = reading.chapter
        text = book.textFor(chapter)
        if not text.strip():
            raise ValueError(f"{chapter.title} has no text in it")

        sentences = splitIntoSentences(text)
        # A bookmark in a text book counts sentences, so resuming is a matter
        # of skipping the ones already spoken.
        start = min(reading.sentenceIndex, len(sentences))

        await self._output.beginUtterance()
        try:
            for position, sentence in enumerate(sentences[start:], start=start):
                async for audio in self._tts.synthesiseStream(
                    sentence, voice=self._voice, language=self._language
                ):
                    await self._output.play(audio)
                reading.sentenceIndex = position + 1
        finally:
            await self._output.endUtterance()


def _importSoundFile() -> Any:
    try:
        import soundfile
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Playing audiobooks requires the 'soundfile' package.\n"
            'Install with: pip install -e ".[books]"'
        ) from error
    return soundfile


__all__ = ["BookReader", "Reading"]
