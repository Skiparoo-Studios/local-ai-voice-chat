"""Playback through the system device.

Uses a stand-in for PortAudio so the behaviour that broke can be asserted:
a streamed reply must reach the device as one continuous stream, not as a
series of playbacks that cut each other off.
"""

from __future__ import annotations

import numpy
import pytest

from app.audio.audioBuffer import AudioBuffer
from app.audio.output import AudioOutputError, SoundDeviceOutput

RATE = 24000


class FakeStream:
    """Records what is written, and whether it was started and stopped."""

    def __init__(self, samplerate: int, channels: int, **_: object) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.written: list[numpy.ndarray] = []
        self.started = False
        self.stopped = False
        self.closed = False
        self.aborted = False

    def start(self) -> None:
        self.started = True

    def write(self, samples) -> None:  # mirrors sounddevice's signature
        if not self.started or self.closed:
            raise RuntimeError("wrote to a stream that was not open")
        self.written.append(numpy.array(samples))

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True

    def abort(self) -> None:
        self.aborted = True


class FakeSoundDevice:
    """Stands in for the sounddevice module."""

    def __init__(self) -> None:
        self.streams: list[FakeStream] = []
        self.failOnOpen = False

    def OutputStream(self, **kwargs) -> FakeStream:  # mirrors sounddevice's name
        if self.failOnOpen:
            raise RuntimeError("no such device")
        stream = FakeStream(**kwargs)
        self.streams.append(stream)
        return stream


def buildOutput() -> tuple[SoundDeviceOutput, FakeSoundDevice]:
    output = SoundDeviceOutput.__new__(SoundDeviceOutput)
    fake = FakeSoundDevice()
    output._device = None
    output._soundDevice = fake
    output._numpy = numpy
    output._stream = None
    output._format = None
    return output, fake


def piece(seconds: float = 0.1) -> AudioBuffer:
    return AudioBuffer.tone(440.0, seconds, RATE)


class TestContinuity:
    """The bug: each piece of a streamed reply cut off the one before it."""

    async def testOneStreamServesEveryPiece(self):
        output, fake = buildOutput()

        for _ in range(6):
            await output.play(piece())

        assert len(fake.streams) == 1
        assert len(fake.streams[0].written) == 6

    async def testNoSampleIsLost(self):
        output, fake = buildOutput()
        pieces = [piece(0.1) for _ in range(5)]

        for item in pieces:
            await output.play(item)

        written = numpy.concatenate(fake.streams[0].written)
        expected = numpy.concatenate(
            [numpy.frombuffer(item.data, dtype="<i2") for item in pieces]
        )
        assert numpy.array_equal(written, expected)

    async def testTheStreamStaysOpenBetweenPieces(self):
        output, fake = buildOutput()

        await output.play(piece())
        await output.play(piece())

        assert not fake.streams[0].closed

    async def testEmptyAudioIsIgnored(self):
        output, fake = buildOutput()

        await output.play(AudioBuffer(data=b"", sampleRate=RATE))

        assert fake.streams == []


class TestFormatChanges:
    async def testADifferentRateOpensANewStream(self):
        output, fake = buildOutput()

        await output.play(AudioBuffer.tone(440.0, 0.1, 24000))
        await output.play(AudioBuffer.tone(440.0, 0.1, 16000))

        assert len(fake.streams) == 2
        assert fake.streams[0].closed
        assert fake.streams[1].samplerate == 16000

    async def testTheSameRateReusesTheStream(self):
        output, fake = buildOutput()

        await output.play(AudioBuffer.tone(440.0, 0.1, 24000))
        await output.play(AudioBuffer.tone(880.0, 0.1, 24000))

        assert len(fake.streams) == 1


class TestInterruption:
    async def testStopDiscardsBufferedAudio(self):
        """Barge-in must stop talking now, not once the buffer drains."""
        output, fake = buildOutput()
        await output.play(piece())

        await output.stop()

        assert fake.streams[0].aborted
        assert fake.streams[0].closed

    async def testStopWithNothingPlayingIsHarmless(self):
        output, _ = buildOutput()

        await output.stop()

    async def testPlayingAgainAfterStopOpensAFreshStream(self):
        output, fake = buildOutput()
        await output.play(piece())
        await output.stop()

        await output.play(piece())

        assert len(fake.streams) == 2


class TestFailures:
    async def testAnUnopenableDeviceIsReportedUsefully(self):
        output, fake = buildOutput()
        fake.failOnOpen = True

        with pytest.raises(AudioOutputError, match="--list-devices"):
            await output.play(piece())

    async def testNonSixteenBitAudioIsRejected(self):
        output, _ = buildOutput()

        with pytest.raises(AudioOutputError, match="16-bit"):
            await output.play(
                AudioBuffer(data=b"\x00", sampleRate=RATE, sampleWidth=1)
            )

    def testCloseReleasesTheDevice(self):
        output, fake = buildOutput()
        output._ensureStream(RATE, 1)

        output.close()

        assert fake.streams[0].closed

    def testCloseWithNoStreamIsHarmless(self):
        output, _ = buildOutput()

        output.close()
