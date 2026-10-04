"""Stopping playback must not close a stream somebody is writing to.

Reported as "the server stopped for some unknown reason" when an audiobook was
stopped. There was no traceback because there was no Python exception:
stopping cancels a play() that is already inside a blocking write on a worker
thread, then calls abort() and close() from another, and PortAudio closing a
stream underneath a write in flight takes the process down natively.

The fake stream here reports the violation instead of crashing, which is the
only way to test for it without risking the test runner.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time

import pytest

from app.audio.audioBuffer import AudioBuffer
from app.audio.output import SoundDeviceOutput


class ObservingStream:
    """A stream that notices being closed or aborted mid-write."""

    def __init__(self, samplerate: int, violations: list[str], **kwargs: object) -> None:
        self.rate = samplerate
        self.violations = violations
        self.writing = False
        self.closed = False

    def start(self) -> None:
        pass

    def write(self, samples: object) -> None:
        if self.closed:
            self.violations.append("write after close")
        self.writing = True
        # Long enough that a stop landing concurrently really overlaps.
        time.sleep(0.05)
        if self.closed:
            self.violations.append("closed during write")
        self.writing = False

    def stop(self) -> None:
        pass

    def abort(self) -> None:
        if self.writing:
            self.violations.append("abort during write")

    def close(self) -> None:
        if self.writing:
            self.violations.append("close during write")
        self.closed = True


def buildOutput(violations: list[str]) -> SoundDeviceOutput:
    import numpy

    class FakeSoundDevice:
        def OutputStream(self, samplerate: int, **kwargs: object) -> ObservingStream:
            return ObservingStream(samplerate, violations, **kwargs)

    output = SoundDeviceOutput.__new__(SoundDeviceOutput)
    output._device = None
    output._soundDevice = FakeSoundDevice()
    output._numpy = numpy
    output._stream = None
    output._format = None
    return output


class TestStoppingDuringAWrite:
    async def testAbortWaitsForTheWriteInFlight(self):
        """The crash: stop() closed the device under a blocking write."""
        violations: list[str] = []
        output = buildOutput(violations)
        audio = AudioBuffer(data=b"\x00\x00" * 2000, sampleRate=44100)

        async def playSeveral() -> None:
            for _ in range(6):
                await output.play(audio)

        writer = asyncio.create_task(playSeveral())
        # Part way through a write, which is when stopping a book lands.
        await asyncio.sleep(0.06)
        await output.stop()

        writer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await writer

        assert violations == []

    async def testRepeatedStopsAreSafe(self):
        """Stop, start, stop is the ordinary rhythm of using this."""
        violations: list[str] = []
        output = buildOutput(violations)
        audio = AudioBuffer(data=b"\x00\x00" * 500, sampleRate=22050)

        for _ in range(4):
            task = asyncio.create_task(output.play(audio))
            await asyncio.sleep(0.01)
            await output.stop()
            task.cancel()
            # Whether the write finished first or was cancelled is timing, and
            # not what this is about; the violations are.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        assert violations == []

    async def testTwoRatesAtOnceStillNeverTearsAWriteApart(self):
        """A book at 44.1 kHz and a reply at 24 both wanting the device."""
        violations: list[str] = []
        output = buildOutput(violations)
        book = AudioBuffer(data=b"\x00\x00" * 800, sampleRate=44100)
        reply = AudioBuffer(data=b"\x00\x00" * 800, sampleRate=24000)

        await asyncio.gather(
            *(output.play(book) for _ in range(3)),
            *(output.play(reply) for _ in range(3)),
        )

        assert violations == []

    def testTheLockIsReentrant(self):
        """_writeBlocking holds it while _ensureStream may close and reopen,
        so a plain Lock would deadlock on itself."""
        assert isinstance(SoundDeviceOutput._lock, type(threading.RLock()))

    async def testClosingIsSafeWhileWriting(self):
        violations: list[str] = []
        output = buildOutput(violations)
        audio = AudioBuffer(data=b"\x00\x00" * 2000, sampleRate=44100)

        task = asyncio.create_task(output.play(audio))
        await asyncio.sleep(0.01)
        await asyncio.to_thread(output.close)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

        assert violations == []
