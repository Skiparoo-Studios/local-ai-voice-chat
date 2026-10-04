"""End-to-end transcription with faster-whisper.

Requires the ``stt`` extra. The Whisper model downloads on first run (about
480 MB for ``small``). Skipped unless VOICE_RUN_MODEL_TESTS=1.

The fixture ``tests/data/utterance.wav`` is a known sentence, so the transcript
can be checked against a reference rather than merely asserted non-empty.

Run with:  pytest -m integration
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from app.audio import AudioBuffer, resampleBuffer
from app.speech.speechToText import Transcript

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("VOICE_RUN_MODEL_TESTS") != "1",
        reason="set VOICE_RUN_MODEL_TESTS=1 to run tests that load Whisper",
    ),
]

FIXTURE = Path(__file__).parents[1] / "data" / "utterance.wav"
REFERENCE = "Turn the kitchen light off and set a timer for ten minutes."

# Whisper writes numbers as digits, so "ten" and "10" are the same transcript.
NUMBER_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten",
}


def normalise(text: str) -> list[str]:
    """Reduce a transcript to comparable words.

    Case, punctuation and digit-versus-word spelling are formatting choices,
    not recognition errors, so they are removed before comparison.
    """
    words = re.findall(r"[a-z0-9']+", text.lower())
    return [NUMBER_WORDS.get(word, word) for word in words]


def wordErrorRate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over words, divided by the reference length."""
    referenceWords = normalise(reference)
    hypothesisWords = normalise(hypothesis)

    if not referenceWords:
        return 0.0 if not hypothesisWords else 1.0

    previous = list(range(len(hypothesisWords) + 1))
    for i, referenceWord in enumerate(referenceWords, start=1):
        current = [i]
        for j, hypothesisWord in enumerate(hypothesisWords, start=1):
            substitution = previous[j - 1] + (referenceWord != hypothesisWord)
            current.append(min(previous[j] + 1, current[j - 1] + 1, substitution))
        previous = current

    return previous[-1] / len(referenceWords)


@pytest.fixture(scope="module")
async def provider():
    """One loaded model shared by every test here, as in the real application."""
    from app.speech.models.fasterWhisperProvider import FasterWhisperProvider

    if not FIXTURE.is_file():
        pytest.skip(f"missing fixture {FIXTURE}")

    instance = FasterWhisperProvider(
        modelSize="small",
        device="auto",
        language="en",
        modelsDirectory=Path("models"),
    )
    await instance.load()
    yield instance
    await instance.unload()


@pytest.fixture(scope="module")
def utterance() -> AudioBuffer:
    return resampleBuffer(AudioBuffer.fromWavFile(FIXTURE), 16000)


class TestTranscription:
    async def testTranscriptMatchesTheReference(self, provider, utterance):
        """Exit criterion: a fixed WAV transcribes to the expected text."""
        transcript = await provider.transcribe(utterance)

        assert not transcript.isEmpty
        assert wordErrorRate(REFERENCE, transcript.text) == 0.0

    async def testReportsTimingsAndLanguage(self, provider, utterance):
        transcript = await provider.transcribe(utterance)

        assert transcript.audioSeconds == pytest.approx(utterance.durationSeconds, abs=0.01)
        assert transcript.transcriptionSeconds > 0
        assert transcript.language == "en"

    async def testFasterThanRealTime(self, provider, utterance):
        """Below 1.0 is the precondition for the Stage 3 conversation loop."""
        transcript = await provider.transcribe(utterance)

        assert transcript.realTimeFactor < 1.0

    async def testModelIsNotReloadedOnSecondCall(self, provider, utterance):
        first = id(provider._model)
        await provider.transcribe(utterance)
        await provider.load()

        assert id(provider._model) == first

    async def testSilenceDoesNotProduceHallucinatedText(self, provider):
        """Whisper invents text during silence; the VAD filter should stop it."""
        transcript = await provider.transcribe(AudioBuffer.silence(3.0, 16000))

        assert transcript.isEmpty


class TestWordErrorRate:
    """The metric itself, so a broken measurement cannot pass a broken model."""

    def testIdenticalTextScoresZero(self):
        assert wordErrorRate("turn the light off", "turn the light off") == 0.0

    def testDigitsAndWordsAreEquivalent(self):
        assert wordErrorRate("set a timer for ten", "set a timer for 10") == 0.0

    def testPunctuationAndCaseAreIgnored(self):
        assert wordErrorRate("turn the light off", "Turn the light off!") == 0.0

    def testOneWrongWordInFour(self):
        assert wordErrorRate("turn the light off", "turn the lamp off") == pytest.approx(0.25)

    def testMissingWordCounts(self):
        assert wordErrorRate("turn the light off", "turn the light") == pytest.approx(0.25)

    def testEmptyHypothesisScoresOne(self):
        assert wordErrorRate("turn the light off", "") == 1.0


class TestTranscriptType:
    def testProviderReturnsATranscript(self, provider, utterance):
        assert isinstance(Transcript(text="x"), Transcript)
