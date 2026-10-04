"""Audio buffer format handling."""

from __future__ import annotations

import wave
from io import BytesIO
from pathlib import Path

import pytest

from app.audio import AudioBuffer


class TestValidation:
    def testRejectsPartialFrames(self):
        with pytest.raises(ValueError, match="whole number"):
            AudioBuffer(data=b"\x00\x01\x02", sampleRate=16000, channels=1, sampleWidth=2)

    def testRejectsNonPositiveSampleRate(self):
        with pytest.raises(ValueError, match="sampleRate"):
            AudioBuffer(data=b"", sampleRate=0)

    def testRejectsUnsupportedSampleWidth(self):
        with pytest.raises(ValueError, match="sampleWidth"):
            AudioBuffer(data=b"", sampleRate=16000, sampleWidth=5)

    def testAcceptsStereoFrames(self):
        buffer = AudioBuffer(data=b"\x00\x00\x00\x00", sampleRate=16000, channels=2)

        assert buffer.frameCount == 1


class TestDuration:
    def testFrameCountAndDuration(self):
        buffer = AudioBuffer(data=b"\x00\x00" * 16000, sampleRate=16000)

        assert buffer.frameCount == 16000
        assert buffer.durationSeconds == pytest.approx(1.0)

    def testEmptyBufferHasZeroDuration(self):
        assert AudioBuffer(data=b"", sampleRate=16000).durationSeconds == 0.0

    def testSilenceHasRequestedDuration(self):
        assert AudioBuffer.silence(0.5, 16000).durationSeconds == pytest.approx(0.5)


class TestFloatConversion:
    def testRoundTripPreservesApproximateValues(self):
        original = [0.0, 0.5, -0.5, 1.0, -1.0]
        buffer = AudioBuffer.fromFloatSamples(original, 16000)

        assert buffer.toFloatSamples() == pytest.approx(original, abs=1e-4)

    def testClipsRatherThanWraps(self):
        """Out-of-range values must saturate; wrapping would turn loud into noise."""
        buffer = AudioBuffer.fromFloatSamples([2.0, -2.0], 16000)
        decoded = buffer.toFloatSamples()

        assert decoded[0] == pytest.approx(1.0, abs=1e-4)
        assert decoded[1] == pytest.approx(-1.0, abs=1e-4)

    def testToFloatSamplesRejectsNon16Bit(self):
        buffer = AudioBuffer(data=b"\x00", sampleRate=16000, sampleWidth=1)

        with pytest.raises(ValueError, match="16-bit"):
            buffer.toFloatSamples()


class TestWav:
    def testWavBytesAreReadableByTheWaveModule(self):
        buffer = AudioBuffer.fromFloatSamples([0.1] * 100, 22050)

        with wave.open(BytesIO(buffer.toWavBytes()), "rb") as handle:
            assert handle.getframerate() == 22050
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getnframes() == 100

    def testWavRoundTrip(self):
        original = AudioBuffer.fromFloatSamples([0.25, -0.25] * 50, 24000)
        restored = AudioBuffer.fromWavBytes(original.toWavBytes())

        assert restored.sampleRate == original.sampleRate
        assert restored.data == original.data

    def testWriteWavCreatesParentDirectories(self, tmp_path: Path):
        path = tmp_path / "nested" / "deeper" / "out.wav"
        AudioBuffer.silence(0.1, 16000).writeWav(path)

        assert path.is_file()
        assert AudioBuffer.fromWavFile(path).durationSeconds == pytest.approx(0.1)


class TestTone:
    def testToneHasRequestedDuration(self):
        tone = AudioBuffer.tone(440.0, 0.25, 22050)

        assert tone.durationSeconds == pytest.approx(0.25, abs=0.01)

    def testToneFadesInAndOutToAvoidClicks(self):
        samples = AudioBuffer.tone(440.0, 0.25, 22050).toFloatSamples()

        assert abs(samples[0]) < 0.01
        assert abs(samples[-1]) < 0.01
        assert max(abs(sample) for sample in samples) > 0.1


class TestConcatenate:
    def testJoinsBuffers(self):
        first = AudioBuffer.silence(0.1, 16000)
        second = AudioBuffer.silence(0.2, 16000)

        assert first.concatenate(second).durationSeconds == pytest.approx(0.3)

    def testRejectsMismatchedFormats(self):
        with pytest.raises(ValueError, match="different formats"):
            AudioBuffer.silence(0.1, 16000).concatenate(AudioBuffer.silence(0.1, 22050))

    def testConcatenatingOntoEmptyWorks(self):
        empty = AudioBuffer(data=b"", sampleRate=16000)
        tone = AudioBuffer.tone(440.0, 0.1, 16000)

        assert empty.concatenate(tone).data == tone.data
