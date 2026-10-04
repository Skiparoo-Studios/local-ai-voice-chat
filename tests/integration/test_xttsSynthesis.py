"""End-to-end XTTS synthesis.

Requires the ``tts`` extra and the XTTS-v2 model, which downloads on first run
(about 1.8 GB) and needs ``COQUI_TOS_AGREED=1``. Skipped when unavailable, so
these can sit in the normal suite without breaking a machine that only runs the
API.

Run with:  pytest -m integration
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.audio import AudioBuffer
from app.speech.voices import VoiceLibrary

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("VOICE_RUN_MODEL_TESTS") != "1",
        reason="set VOICE_RUN_MODEL_TESTS=1 to run tests that load XTTS",
    ),
]


@pytest.fixture(scope="module")
async def provider(tmp_path_factory):
    """One loaded model shared by every test here, as in the real application."""
    from app.speech.models.xttsProvider import XttsProvider

    instance = XttsProvider(
        voices=VoiceLibrary(Path("voices")),
        modelsDirectory=Path("models"),
        device="auto",
    )
    await instance.load()
    yield instance
    await instance.unload()


class TestSynthesis:
    async def testSynthesisProducesAudibleAudio(self, provider):
        audio = await provider.synthesise("Turn the kitchen light off.")

        assert isinstance(audio, AudioBuffer)
        assert audio.durationSeconds > 0.5
        assert audio.sampleRate == provider.sampleRate
        # Silence would mean the model ran but produced nothing useful.
        assert max(abs(sample) for sample in audio.toFloatSamples()) > 0.01

    async def testLongTextIsChunkedAndJoined(self, provider):
        sentence = "The assistant runs entirely on local hardware. "
        audio = await provider.synthesise(sentence * 6)

        assert audio.durationSeconds > 5.0

    async def testModelIsNotReloadedOnSecondCall(self, provider):
        """Exit criterion: the model loads once and stays resident."""
        first = id(provider._model)
        await provider.synthesise("One.")
        await provider.load()

        assert id(provider._model) == first

    async def testDifferentSpeakersProduceDifferentAudio(self, provider):
        available = provider.availableVoices()
        if len(available) < 2:
            pytest.skip("needs at least two voices")

        first = await provider.synthesise("Hello there.", voice=available[0])
        second = await provider.synthesise("Hello there.", voice=available[1])

        assert first.data != second.data


class TestClonedVoice:
    async def testClonesFromASample(self, provider):
        voices = VoiceLibrary(Path("voices"))
        local = voices.listVoices()
        if not local:
            pytest.skip("no voice samples in voices/")

        audio = await provider.synthesise("This is a cloned voice.", voice=local[0])

        assert audio.durationSeconds > 0.5

    async def testEmbeddingIsCachedToDisk(self, provider):
        voices = VoiceLibrary(Path("voices"))
        local = voices.listVoices()
        if not local:
            pytest.skip("no voice samples in voices/")

        await provider.synthesise("Warming the cache.", voice=local[0])

        assert list(voices.cacheDirectory.glob(f"{local[0]}-*.pth"))
