"""End-to-end Piper synthesis, against the real model.

Requires the ``piper`` extra. The voice downloads on first run, which is about
60 MB rather than XTTS's 1.8 GB, and needs no licence acceptance. Skipped when
unavailable, so these sit in the normal suite without breaking a machine that
has not installed it.

Run with:  pytest -m integration
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.audio import AudioBuffer

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("VOICE_RUN_MODEL_TESTS") != "1",
        reason="set VOICE_RUN_MODEL_TESTS=1 to run tests that load Piper",
    ),
]

VOICE = "en_GB-cori-medium"


@pytest.fixture(scope="module")
async def provider():
    """One loaded voice shared by every test here, as in the real application."""
    from app.speech.models.piperProvider import PiperProvider

    instance = PiperProvider(VOICE, modelsDirectory=Path("models") / "piper")
    await instance.load()
    yield instance
    await instance.unload()


class TestSynthesis:
    async def testItProducesAudibleAudio(self, provider):
        audio = await provider.synthesise("The kitchen light is off.")

        assert isinstance(audio, AudioBuffer)
        assert audio.durationSeconds > 0.5
        assert audio.sampleRate == provider.sampleRate

    async def testLongerTextProducesLongerAudio(self, provider):
        # Deliberately not "Yes." --- see testAKnownVoiceQuirk below.
        short = await provider.synthesise("Hello.")
        long = await provider.synthesise(
            "The kitchen light is off, and I have set a timer for ten minutes."
        )

        assert long.durationSeconds > short.durationSeconds

    async def testVeryShortUtterancesAlwaysProduceAudio(self, provider):
        """Piper is unreliable on one-word input, so check it never breaks.

        Duration is deliberately not asserted. "Yes." on en_GB-cori-medium is
        sometimes 0.7 s and sometimes three and a half, and one run in nine
        transcribed as "Yes! vvvvvvvvvvvv" --- a VITS duration predictor with
        almost no phonemes to work from. Raw piper produces the same bytes, so
        it is the model rather than this provider, and asserting on it would
        only give a test that fails at random.

        What must hold is that nothing crashes and something audible comes
        back. The instability itself is documented for users to work around by
        phrasing or voice, not defended against here.
        """
        for text in ("Yes.", "No.", "Done.", "Okay!"):
            for _ in range(2):
                audio = await provider.synthesise(text)
                assert audio.durationSeconds > 0.1, text
                assert audio.sampleRate == provider.sampleRate

    async def testItIsFasterThanRealTimeOnACpu(self, provider):
        """The reason Piper is worth having on a Pi. Measured at 0.054 on this
        machine; the threshold is loose so a slow CI box does not fail."""
        import time

        started = time.perf_counter()
        audio = await provider.synthesise(
            "The kitchen light is off, and I have set a timer for ten minutes."
        )
        elapsed = time.perf_counter() - started

        assert elapsed < audio.durationSeconds

    async def testEachSentenceArrivesSeparately(self, provider):
        pieces = [
            piece
            async for piece in provider.synthesiseStream(
                "The light is off. The timer is set. Anything else?"
            )
        ]

        assert len(pieces) >= 2

    async def testStreamingBeginsBeforeGenerationFinishes(self, provider):
        import time

        started = time.perf_counter()
        first = None
        total = 0.0
        async for piece in provider.synthesiseStream(
            "The light is off. The timer is set. Anything else?"
        ):
            if first is None:
                first = time.perf_counter() - started
            total += piece.durationSeconds

        assert first is not None
        assert first < total

    async def testTheSpeakingRateCanBeSlowed(self):
        """Worth having for a small child."""
        from app.speech.models.piperProvider import PiperProvider

        directory = Path("models") / "piper"
        normal = PiperProvider(VOICE, modelsDirectory=directory)
        slow = PiperProvider(VOICE, modelsDirectory=directory, lengthScale=1.5)
        await normal.load()
        await slow.load()

        text = "What sound does a sheep make?"
        assert (await slow.synthesise(text)).durationSeconds > (
            await normal.synthesise(text)
        ).durationSeconds

        await normal.unload()
        await slow.unload()


class TestItIsUnderstandable:
    """Synthesise, then transcribe, and compare. The strongest check available
    without a person listening: it proves the audio is speech saying the right
    words, not merely that bytes came back."""

    async def testWhisperHearsWhatPiperSaid(self, provider):
        from app.audio.input import resampleBuffer
        from app.config import Settings
        from app.speech.speechToText import createSpeechToTextProvider

        recogniser = createSpeechToTextProvider(
            Settings(speechToText={"provider": "faster-whisper", "model": "small"})
        )
        await recogniser.load()
        try:
            spoken = "Turn the kitchen light off."
            audio = await provider.synthesise(spoken)
            heard = await recogniser.transcribe(
                resampleBuffer(audio, recogniser.requiredSampleRate), language="en"
            )

            assert heard.text.strip().lower().rstrip(".") == spoken.lower().rstrip(".")
        finally:
            await recogniser.unload()
