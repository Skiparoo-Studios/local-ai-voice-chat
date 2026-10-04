"""Speech-to-text abstraction, registry and the scripted provider."""

from __future__ import annotations

import pytest

from app.audio import AudioBuffer
from app.config import Settings
from app.speech.models.fasterWhisperProvider import (
    COMPUTE_TYPE_FOR_DEVICE,
    FasterWhisperProvider,
)
from app.speech.models.scriptedProvider import ScriptedProvider
from app.speech.speechToText import (
    STT_PROVIDER_REGISTRY,
    SpeechToTextProvider,
    Transcript,
    TranscriptionError,
    UnknownSttProviderError,
    createSpeechToTextProvider,
    registerSttProvider,
    resolveSttProviderClass,
)


class TestRegistry:
    def testResolvesRegisteredProvider(self):
        assert resolveSttProviderClass("scripted") is ScriptedProvider

    def testUnknownProviderListsAvailableOnes(self):
        with pytest.raises(UnknownSttProviderError, match="Available:"):
            resolveSttProviderClass("does-not-exist")

    def testWhisperIsRegisteredWithoutBeingImported(self):
        assert "faster-whisper" in STT_PROVIDER_REGISTRY

    def testProvidersCanBeRegistered(self):
        registerSttProvider("custom", "app.speech.models.scriptedProvider:ScriptedProvider")
        try:
            assert resolveSttProviderClass("custom") is ScriptedProvider
        finally:
            STT_PROVIDER_REGISTRY.pop("custom", None)


class TestProviderSelection:
    """Exit criterion: swapping providers is configuration, not code."""

    def testConfigurationSelectsTheProvider(self):
        settings = Settings(speechToText={"provider": "scripted"})
        provider = createSpeechToTextProvider(settings)

        assert isinstance(provider, ScriptedProvider)
        assert isinstance(provider, SpeechToTextProvider)

    def testUnknownConfiguredProviderIsRejected(self):
        settings = Settings(speechToText={"provider": "nope"})

        with pytest.raises(UnknownSttProviderError):
            createSpeechToTextProvider(settings)

    def testWhisperReceivesConfiguredSettings(self):
        settings = Settings(
            speechToText={
                "provider": "faster-whisper",
                "model": "tiny",
                "device": "cpu",
                "language": "fr",
                "beamSize": 1,
            }
        )
        provider = createSpeechToTextProvider(settings)

        assert provider._modelSize == "tiny"
        assert provider._devicePreference == "cpu"
        assert provider._language == "fr"
        assert provider._beamSize == 1


class TestScriptedProvider:
    async def testReturnsConfiguredPhrasesInOrder(self):
        provider = ScriptedProvider(phrases=("first", "second"))
        await provider.load()
        audio = AudioBuffer.silence(1.0, 16000)

        assert (await provider.transcribe(audio)).text == "first"
        assert (await provider.transcribe(audio)).text == "second"

    async def testPhrasesRepeatAfterTheLast(self):
        provider = ScriptedProvider(phrases=("only",))
        await provider.load()
        audio = AudioBuffer.silence(1.0, 16000)

        assert (await provider.transcribe(audio)).text == "only"
        assert (await provider.transcribe(audio)).text == "only"

    async def testReportsTheAudioDuration(self):
        provider = ScriptedProvider()
        await provider.load()

        transcript = await provider.transcribe(AudioBuffer.silence(2.0, 16000))

        assert transcript.audioSeconds == pytest.approx(2.0)

    async def testLoadAndUnloadTrackState(self):
        provider = ScriptedProvider()
        assert not provider.isLoaded

        await provider.load()
        assert provider.isLoaded

        await provider.unload()
        assert not provider.isLoaded


class TestTranscript:
    def testEmptyTranscriptIsDetected(self):
        assert Transcript(text="").isEmpty
        assert Transcript(text="   ").isEmpty
        assert not Transcript(text="hello").isEmpty

    def testRealTimeFactorIsTranscriptionOverAudio(self):
        transcript = Transcript(text="x", audioSeconds=4.0, transcriptionSeconds=1.0)

        assert transcript.realTimeFactor == pytest.approx(0.25)

    def testZeroLengthAudioDoesNotDivideByZero(self):
        assert Transcript(text="x", audioSeconds=0.0).realTimeFactor == 0.0

    def testDescriptionMentionsBothDurations(self):
        description = Transcript(
            text="x", audioSeconds=4.0, transcriptionSeconds=1.0
        ).describe()

        assert "1.00s" in description
        assert "4.00s" in description
        assert "RTF" in description


class TestWhisperConfiguration:
    """Behaviour that does not require the model to be loaded."""

    def testComputeTypeDefaultsPerDevice(self):
        provider = FasterWhisperProvider(computeType="auto")

        assert provider._resolveComputeType("cuda") == COMPUTE_TYPE_FOR_DEVICE["cuda"]
        assert provider._resolveComputeType("cpu") == COMPUTE_TYPE_FOR_DEVICE["cpu"]

    def testExplicitComputeTypeIsHonoured(self):
        provider = FasterWhisperProvider(computeType="int8_float16")

        assert provider._resolveComputeType("cuda") == "int8_float16"

    async def testTranscribingBeforeLoadingIsRejected(self):
        provider = FasterWhisperProvider()

        with pytest.raises(TranscriptionError, match="not loaded"):
            await provider.transcribe(AudioBuffer.silence(1.0, 16000))

    async def testWrongSampleRateIsRejectedWithGuidance(self):
        """Silently transcribing 44 kHz audio would just produce poor results."""
        provider = FasterWhisperProvider()
        provider._model = object()

        with pytest.raises(TranscriptionError, match="16000 Hz"):
            await provider.transcribe(AudioBuffer.silence(1.0, 44100))

    async def testEmptyAudioReturnsAnEmptyTranscript(self):
        provider = FasterWhisperProvider()
        provider._model = object()

        transcript = await provider.transcribe(AudioBuffer(data=b"", sampleRate=16000))

        assert transcript.isEmpty

    def testDescribeMentionsTheUnresolvedDevice(self):
        assert "unresolved" in FasterWhisperProvider(device="auto").describe()

    def testRequiredSampleRateIsWhisperNative(self):
        assert FasterWhisperProvider().requiredSampleRate == 16000
