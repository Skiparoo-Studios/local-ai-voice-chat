"""Audio capture abstraction and resampling."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.audio import AudioBuffer, MemoryAudioInput, WavFileInput
from app.audio.input import AudioInputError, resampleBuffer


class TestWavFileInput:
    async def testReadsAWavFile(self, tmp_path: Path):
        path = tmp_path / "speech.wav"
        AudioBuffer.tone(440.0, 0.5, 16000).writeWav(path)

        audio = await WavFileInput(path).record()

        assert audio.sampleRate == 16000
        assert audio.durationSeconds == pytest.approx(0.5, abs=0.01)

    async def testResamplesToTheTargetRate(self, tmp_path: Path):
        path = tmp_path / "speech.wav"
        AudioBuffer.tone(440.0, 0.5, 44100).writeWav(path)

        audio = await WavFileInput(path, targetSampleRate=16000).record()

        assert audio.sampleRate == 16000
        assert audio.durationSeconds == pytest.approx(0.5, abs=0.02)

    async def testMissingFileIsReported(self, tmp_path: Path):
        with pytest.raises(AudioInputError, match="not found"):
            await WavFileInput(tmp_path / "absent.wav").record()


class TestMemoryAudioInput:
    async def testReturnsQueuedBuffers(self):
        first = AudioBuffer.tone(440.0, 0.1, 16000)
        second = AudioBuffer.tone(880.0, 0.2, 16000)
        source = MemoryAudioInput([first, second])

        assert (await source.record()).data == first.data
        assert (await source.record()).data == second.data

    async def testFallsBackToSilenceWhenEmpty(self):
        audio = await MemoryAudioInput().record()

        assert audio.durationSeconds > 0

    async def testCountsRecordings(self):
        source = MemoryAudioInput()

        await source.record()
        await source.record()

        assert source.recordCount == 2

    async def testAcceptsAStopSignal(self):
        """The interface must take a stop signal even where it is ignored."""
        stop = asyncio.Event()

        audio = await MemoryAudioInput().record(stopWhen=stop, maximumSeconds=1.0)

        assert audio is not None


class TestMicrophoneStall:
    """A microphone that leaves the bus must not leave the caller waiting.

    Without this the process stays alive and connected but deaf: no audio, and
    no crash for systemd to restart. It matters most for an unattended device
    with a physical switch on the microphone.
    """

    def buildMicrophone(self, stallSeconds: float):
        from app.audio.input import SoundDeviceInput

        # Constructed without __init__ so the test needs no sound card.
        microphone = SoundDeviceInput.__new__(SoundDeviceInput)
        microphone._queue = asyncio.Queue()
        microphone._stallSeconds = stallSeconds
        return microphone

    async def testADeviceThatStopsDeliveringIsReported(self):
        microphone = self.buildMicrophone(0.05)

        with pytest.raises(AudioInputError, match="No audio from the microphone"):
            await microphone._nextFrame(microphone._queue)

    async def testTheMessageSaysWhatToLookAt(self):
        microphone = self.buildMicrophone(0.05)

        with pytest.raises(AudioInputError) as caught:
            await microphone._nextFrame(microphone._queue)

        message = str(caught.value)
        assert "unplugged or switched off at the connector" in message
        assert "captureStallSeconds" in message

    async def testAMutedButConnectedDeviceIsNotAStall(self):
        """Muting silences the signal; the device keeps streaming zeroes."""
        microphone = self.buildMicrophone(0.05)
        silence = b"\x00\x00" * 512
        microphone._queue.put_nowait(silence)

        assert await microphone._nextFrame(microphone._queue) == silence

    async def testWaitingIndefinitelyIsStillAvailable(self):
        microphone = self.buildMicrophone(0.0)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                microphone._nextFrame(microphone._queue), timeout=0.05
            )

    def testTheStallComesFromSettings(self):
        from app.audio.input import createAudioInput
        from app.config import Settings

        microphone = createAudioInput(
            Settings(audio={"captureStallSeconds": 12.5, "outputBackend": "file"})
        )

        assert microphone._stallSeconds == 12.5


class TestResampling:
    def testDownsamplePreservesDuration(self):
        original = AudioBuffer.tone(440.0, 1.0, 48000)

        resampled = resampleBuffer(original, 16000)

        assert resampled.sampleRate == 16000
        assert resampled.durationSeconds == pytest.approx(1.0, abs=0.01)

    def testUpsamplePreservesDuration(self):
        original = AudioBuffer.tone(440.0, 0.5, 8000)

        resampled = resampleBuffer(original, 16000)

        assert resampled.sampleRate == 16000
        assert resampled.durationSeconds == pytest.approx(0.5, abs=0.01)

    def testMatchingRateIsUnchanged(self):
        original = AudioBuffer.tone(440.0, 0.5, 16000)

        assert resampleBuffer(original, 16000) is original

    def testEmptyBufferIsHandled(self):
        empty = AudioBuffer(data=b"", sampleRate=44100)

        assert resampleBuffer(empty, 16000).data == b""

    def testStereoIsDownmixedToMono(self):
        """Whisper wants mono; capture devices often default to stereo."""
        stereo = AudioBuffer(data=b"\x00\x10\x00\x20" * 100, sampleRate=16000, channels=2)

        result = resampleBuffer(stereo, 16000)

        assert result.channels == 1
        assert result.frameCount == 100

    def testStereoIsDownmixedAndResampledTogether(self):
        stereo = AudioBuffer(data=b"\x00\x10\x00\x20" * 48000, sampleRate=48000, channels=2)

        result = resampleBuffer(stereo, 16000)

        assert result.channels == 1
        assert result.sampleRate == 16000
        assert result.durationSeconds == pytest.approx(1.0, abs=0.01)

    def testResamplingKeepsSignalEnergy(self):
        """A resampler that produced silence would still pass a duration check."""
        original = AudioBuffer.tone(440.0, 0.5, 48000, amplitude=0.5)

        resampled = resampleBuffer(original, 16000)

        assert max(abs(sample) for sample in resampled.toFloatSamples()) > 0.2
