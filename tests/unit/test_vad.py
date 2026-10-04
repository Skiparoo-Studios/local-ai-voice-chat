"""Voice activity detection and utterance segmentation."""

from __future__ import annotations

import pytest

from app.audio.audioBuffer import AudioBuffer
from app.audio.input import splitIntoFrames
from app.audio.vad import (
    EnergyVadProvider,
    SegmenterSettings,
    SegmentState,
    SpeechSegmenter,
    UnknownVadProviderError,
    createVadProvider,
    resolveVadProviderClass,
    rootMeanSquare,
)
from app.config import Settings

RATE = 16000
FRAME_MS = 32


def frame(*, loud: bool) -> AudioBuffer:
    """One 32 ms frame, either speech-like or silent."""
    if loud:
        return AudioBuffer.tone(220.0, FRAME_MS / 1000, RATE, amplitude=0.5, fadeSeconds=0.0)
    return AudioBuffer.silence(FRAME_MS / 1000, RATE)


def feedFrames(segmenter: SpeechSegmenter, pattern: str, vad: EnergyVadProvider):
    """Drive the segmenter with a pattern of 's' (speech) and '.' (silence)."""
    results = []
    for symbol in pattern:
        piece = frame(loud=symbol == "s")
        results.append(segmenter.feed(piece, vad.isSpeech(piece)))
    return results


class TestEnergyVad:
    def testLoudFrameIsSpeech(self):
        assert EnergyVadProvider(threshold=0.02).isSpeech(frame(loud=True))

    def testSilentFrameIsNot(self):
        assert not EnergyVadProvider(threshold=0.02).isSpeech(frame(loud=False))

    def testThresholdIsRespected(self):
        assert not EnergyVadProvider(threshold=0.99).isSpeech(frame(loud=True))

    def testRootMeanSquareOfSilenceIsZero(self):
        assert rootMeanSquare(frame(loud=False)) == 0.0

    def testRootMeanSquareOfEmptyFrameIsZero(self):
        assert rootMeanSquare(AudioBuffer(data=b"", sampleRate=RATE)) == 0.0


class TestRegistry:
    def testResolvesProviders(self):
        assert resolveVadProviderClass("energy") is EnergyVadProvider

    def testUnknownProviderListsAvailableOnes(self):
        with pytest.raises(UnknownVadProviderError, match="Available:"):
            resolveVadProviderClass("nope")

    def testConfigurationSelectsTheProvider(self):
        provider = createVadProvider(Settings(vad={"provider": "energy"}))

        assert isinstance(provider, EnergyVadProvider)


class TestSegmentation:
    """Ending on silence rather than a timer is the point of this stage."""

    def testSpeechThenSilenceCompletesAnUtterance(self):
        vad = EnergyVadProvider(threshold=0.02)
        segmenter = SpeechSegmenter(
            SegmenterSettings(startFrames=2, silenceFrames=3, prerollFrames=2)
        )

        results = feedFrames(segmenter, "ssssssssss...", vad)

        assert results[-1].isComplete
        assert results[-1].utterance is not None
        assert results[-1].reason == "silence"

    def testCaptureDoesNotStartOnOneNoisyFrame(self):
        vad = EnergyVadProvider(threshold=0.02)
        segmenter = SpeechSegmenter(SegmenterSettings(startFrames=3))

        results = feedFrames(segmenter, "s.s.s.", vad)

        assert all(result.state is SegmentState.waiting for result in results)

    def testPrerollKeepsAudioFromBeforeDetection(self):
        """Without this the first word is clipped, because speech is always
        recognised slightly after it began."""
        vad = EnergyVadProvider(threshold=0.02)
        withPreroll = SpeechSegmenter(
            SegmenterSettings(
                startFrames=2, silenceFrames=3, prerollFrames=5, minimumSeconds=0.0
            )
        )
        withoutPreroll = SpeechSegmenter(
            SegmenterSettings(
                startFrames=2, silenceFrames=3, prerollFrames=0, minimumSeconds=0.0
            )
        )

        pattern = ".....ssssssss..."
        longer = feedFrames(withPreroll, pattern, vad)[-1]
        shorter = feedFrames(withoutPreroll, pattern, vad)[-1]

        assert longer.utterance.durationSeconds > shorter.utterance.durationSeconds

    def testSilenceAloneNeverCompletes(self):
        vad = EnergyVadProvider(threshold=0.02)
        segmenter = SpeechSegmenter()

        results = feedFrames(segmenter, "." * 60, vad)

        assert not any(result.isComplete for result in results)

    def testBriefPauseDoesNotEndTheUtterance(self):
        """A pause mid-sentence must not be treated as the end."""
        vad = EnergyVadProvider(threshold=0.02)
        segmenter = SpeechSegmenter(
            SegmenterSettings(startFrames=2, silenceFrames=10, prerollFrames=2)
        )

        results = feedFrames(segmenter, "sssss..ssssss", vad)

        assert not any(result.isComplete for result in results)

    def testMaximumLengthEndsTheUtterance(self):
        vad = EnergyVadProvider(threshold=0.02)
        segmenter = SpeechSegmenter(
            SegmenterSettings(
                startFrames=1, silenceFrames=99, maximumSeconds=0.2, minimumSeconds=0.0
            )
        )

        results = feedFrames(segmenter, "s" * 20, vad)

        completed = [result for result in results if result.isComplete]
        assert completed
        assert completed[0].reason == "reached the maximum length"

    def testVeryShortUtteranceIsDiscarded(self):
        """A cough should not become a transcription request."""
        vad = EnergyVadProvider(threshold=0.02)
        segmenter = SpeechSegmenter(
            SegmenterSettings(
                startFrames=1, silenceFrames=2, prerollFrames=0, minimumSeconds=1.0
            )
        )

        results = feedFrames(segmenter, "ss...", vad)

        assert results[-1].isComplete
        assert results[-1].utterance is None
        assert results[-1].reason == "too short"

    def testResetReturnsToWaiting(self):
        vad = EnergyVadProvider(threshold=0.02)
        segmenter = SpeechSegmenter(SegmenterSettings(startFrames=1))
        feedFrames(segmenter, "sss", vad)

        segmenter.reset()

        assert segmenter.state is SegmentState.waiting
        assert segmenter.capturedSeconds == 0.0

    def testUtteranceKeepsTheFrameFormat(self):
        vad = EnergyVadProvider(threshold=0.02)
        segmenter = SpeechSegmenter(
            SegmenterSettings(
                startFrames=1, silenceFrames=2, prerollFrames=0, minimumSeconds=0.0
            )
        )

        results = feedFrames(segmenter, "sssss...", vad)

        assert results[-1].utterance.sampleRate == RATE
        assert results[-1].utterance.channels == 1

    def testDefaultMinimumRejectsAVeryBriefNoise(self):
        """The 0.3s default is load-bearing: a cough is not a request."""
        vad = EnergyVadProvider(threshold=0.02)
        segmenter = SpeechSegmenter(SegmenterSettings(startFrames=1, silenceFrames=2))

        results = feedFrames(segmenter, "ss...", vad)

        assert results[-1].isComplete
        assert results[-1].utterance is None


class TestFrameSplitting:
    def testBufferIsCutIntoEqualFrames(self):
        buffer = AudioBuffer.silence(1.0, RATE)

        frames = splitIntoFrames(buffer, FRAME_MS)

        assert len(frames) == 31  # 1000 / 32, remainder discarded
        assert all(len(piece.data) == len(frames[0].data) for piece in frames)

    def testShortRemainderIsDiscarded(self):
        """Models want uniform frames, so a partial one is dropped."""
        buffer = AudioBuffer.silence(0.05, RATE)

        frames = splitIntoFrames(buffer, FRAME_MS)

        assert len(frames) == 1

    def testBufferShorterThanOneFrameProducesNothing(self):
        assert splitIntoFrames(AudioBuffer.silence(0.01, RATE), FRAME_MS) == []

    def testFramesPreserveTheFormat(self):
        frames = splitIntoFrames(AudioBuffer.silence(0.5, RATE), FRAME_MS)

        assert all(piece.sampleRate == RATE for piece in frames)
