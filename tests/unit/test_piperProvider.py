"""The Piper provider, without downloading a 60 MB voice.

Synthesis itself is exercised in tests/integration/test_piperSynthesis.py,
which needs the real model. Everything here is the wiring around it: voice
resolution, configuration, and the failures a user is most likely to meet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.audio.audioBuffer import AudioBuffer
from app.config import Settings
from app.speech.models.piperProvider import (
    DEFAULT_VOICE,
    PiperProvider,
    _splitSpeaker,
)
from app.speech.textToSpeech import (
    PROVIDER_REGISTRY,
    ProviderUnavailableError,
    createTextToSpeechProvider,
    resolveProviderClass,
)
from app.speech.voices import VoiceLibrary


class FakeChunk:
    """What piper yields: raw int16 bytes and the format they are in."""

    def __init__(self, data: bytes, sampleRate: int = 22050) -> None:
        self.audio_int16_bytes = data
        self.sample_rate = sampleRate
        self.sample_width = 2
        self.sample_channels = 1


class FakeVoice:
    """Stands in for piper.PiperVoice."""

    def __init__(self, chunks: int = 2, sampleRate: int = 22050) -> None:
        self._chunks = chunks
        self.config = type("Config", (), {"sample_rate": sampleRate, "num_speakers": 1})()
        self.calls: list[tuple[str, object]] = []

    def synthesize(self, text: str, config=None, **kwargs):
        self.calls.append((text, config))
        for index in range(self._chunks):
            yield FakeChunk(bytes(220 * (index + 1)), self.config.sample_rate)


def buildProvider(tmp_path: Path, voice=None, **kwargs) -> PiperProvider:
    provider = PiperProvider(
        kwargs.pop("voiceName", DEFAULT_VOICE), modelsDirectory=tmp_path, **kwargs
    )
    if voice is not None:
        provider._voices[DEFAULT_VOICE] = voice
        provider._sampleRate = voice.config.sample_rate
    return provider


class TestRegistration:
    def testPiperIsRegistered(self):
        assert "piper" in PROVIDER_REGISTRY

    def testItResolvesToTheProviderClass(self):
        assert resolveProviderClass("piper") is PiperProvider

    def testSelectingItIsConfiguration(self, tmp_path):
        settings = Settings(
            textToSpeech={"provider": "piper"}, paths={"models": tmp_path}
        )

        provider = createTextToSpeechProvider(settings, VoiceLibrary(tmp_path))

        assert isinstance(provider, PiperProvider)

    def testItDoesNotDragInXtts(self):
        """The whole point of the registry: choosing one provider must not
        import another's dependencies."""
        import sys

        assert "app.speech.models.piperProvider" in sys.modules
        assert "TTS" not in sys.modules


class TestConfiguration:
    def testThePlaceholderVoiceFallsBackToARealOne(self, tmp_path):
        """settings.example.json ships voice "default", which names no Piper
        voice and would otherwise be looked up and fail."""
        settings = Settings(
            textToSpeech={"provider": "piper", "voice": "default"},
            paths={"models": tmp_path},
        )

        provider = PiperProvider.fromSettings(settings, VoiceLibrary(tmp_path))

        assert provider._voiceName == DEFAULT_VOICE

    def testAConfiguredVoiceIsUsed(self, tmp_path):
        settings = Settings(
            textToSpeech={"provider": "piper", "voice": "en_US-lessac-medium"},
            paths={"models": tmp_path},
        )

        provider = PiperProvider.fromSettings(settings, VoiceLibrary(tmp_path))

        assert provider._voiceName == "en_US-lessac-medium"

    def testVoicesLiveUnderTheModelsDirectory(self, tmp_path):
        settings = Settings(textToSpeech={"provider": "piper"}, paths={"models": tmp_path})

        provider = PiperProvider.fromSettings(settings, VoiceLibrary(tmp_path))

        assert provider._modelsDirectory == tmp_path / "piper"

    def testCudaIsReportedRatherThanAttempted(self, tmp_path, caplog):
        """Piper is faster than real time on a CPU, and onnxruntime-gpu is a
        separate package. Saying so beats failing to load."""
        import logging

        settings = Settings(
            textToSpeech={"provider": "piper", "device": "cuda"}, paths={"models": tmp_path}
        )

        with caplog.at_level(logging.INFO):
            provider = PiperProvider.fromSettings(settings, VoiceLibrary(tmp_path))

        assert "cpu" in provider.describe()
        assert "device is ignored" in caplog.text

    def testTheSpeechRateIsPassedThrough(self, tmp_path):
        settings = Settings(
            textToSpeech={"provider": "piper", "speechRate": 1.3}, paths={"models": tmp_path}
        )

        assert PiperProvider.fromSettings(settings, VoiceLibrary(tmp_path))._lengthScale == 1.3

    def testNoSpeechRateLeavesTheVoiceAlone(self, tmp_path):
        settings = Settings(textToSpeech={"provider": "piper"}, paths={"models": tmp_path})

        provider = PiperProvider.fromSettings(settings, VoiceLibrary(tmp_path))

        assert provider._lengthScale is None
        assert provider._synthesisConfig(None) is None


class TestMultiSpeakerVoices:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("en_GB-cori-medium", ("en_GB-cori-medium", None)),
            ("en_US-libritts-high#42", ("en_US-libritts-high", 42)),
            ("en_US-libritts-high#0", ("en_US-libritts-high", 0)),
        ],
    )
    def testASpeakerIndexIsSeparatedFromTheVoice(self, name, expected):
        assert _splitSpeaker(name) == expected

    def testAnUnreadableSpeakerIndexIsIgnoredNotFatal(self, tmp_path, caplog):
        assert _splitSpeaker("en_US-libritts-high#lots") == ("en_US-libritts-high", None)


class TestSynthesis:
    async def testStreamingYieldsOneBufferPerChunk(self, tmp_path):
        provider = buildProvider(tmp_path, voice=FakeVoice(chunks=3))

        pieces = [audio async for audio in provider.synthesiseStream("Three sentences here.")]

        assert len(pieces) == 3
        assert all(isinstance(piece, AudioBuffer) for piece in pieces)
        assert all(piece.sampleRate == 22050 for piece in pieces)

    async def testTheWholeUtteranceIsTheChunksJoined(self, tmp_path):
        voice = FakeVoice(chunks=3)
        provider = buildProvider(tmp_path, voice=voice)

        whole = await provider.synthesise("Three sentences here.")

        assert len(whole.data) == 220 + 440 + 660

    async def testStreamingIsAdvertised(self, tmp_path):
        """Piper emits per sentence, so the runtime should take that path."""
        assert buildProvider(tmp_path, voice=FakeVoice()).supportsStreaming

    async def testEmptyTextProducesNoAudioAndNoCall(self, tmp_path):
        voice = FakeVoice()
        provider = buildProvider(tmp_path, voice=voice)

        assert (await provider.synthesise("   ")).durationSeconds == 0
        assert voice.calls == []

    async def testLanguageIsAcceptedAndIgnored(self, tmp_path):
        """A Piper voice is trained for one language; the voice chooses it."""
        provider = buildProvider(tmp_path, voice=FakeVoice())

        assert (await provider.synthesise("Hello", language="fr")).data

    async def testAFailureDuringSynthesisReachesTheCaller(self, tmp_path):
        from app.speech.textToSpeech import SpeechSynthesisError

        class BrokenVoice(FakeVoice):
            def synthesize(self, text, config=None, **kwargs):
                raise RuntimeError("onnxruntime fell over")
                yield  # pragma: no cover

        provider = buildProvider(tmp_path, voice=BrokenVoice())

        with pytest.raises(SpeechSynthesisError, match="onnxruntime fell over"):
            await provider.synthesise("Hello")

    async def testTheSampleRateFollowsTheVoice(self, tmp_path):
        provider = buildProvider(tmp_path, voice=FakeVoice(sampleRate=16000))

        assert provider.sampleRate == 16000


class TestMissingVoices:
    async def testDownloadingCanBeRefused(self, tmp_path):
        """A machine that should never reach the network says what to run."""
        provider = PiperProvider(
            "en_GB-cori-medium", modelsDirectory=tmp_path, allowDownload=False
        )

        with pytest.raises(ProviderUnavailableError, match="download_voices"):
            await provider.load()

    async def testTheErrorNamesTheVoiceAndTheDirectory(self, tmp_path):
        provider = PiperProvider("en_XX-nonesuch", modelsDirectory=tmp_path, allowDownload=False)

        with pytest.raises(ProviderUnavailableError) as caught:
            await provider.load()

        assert "en_XX-nonesuch" in str(caught.value)
        assert str(tmp_path) in str(caught.value)

    def testAnAlreadyPresentVoiceIsNotDownloaded(self, tmp_path):
        (tmp_path / "en_GB-cori-medium.onnx").write_bytes(b"not really a model")
        provider = PiperProvider(
            "en_GB-cori-medium", modelsDirectory=tmp_path, allowDownload=False
        )

        assert provider._resolveModelPath("en_GB-cori-medium").is_file()


class TestVoiceListing:
    def testDownloadedVoicesAreListed(self, tmp_path):
        for name in ("en_GB-cori-medium", "en_US-lessac-medium"):
            (tmp_path / f"{name}.onnx").write_bytes(b"x")
            (tmp_path / f"{name}.onnx.json").write_text("{}")
        provider = PiperProvider(modelsDirectory=tmp_path)

        assert provider.availableVoices() == ["en_GB-cori-medium", "en_US-lessac-medium"]

    def testAnAbsentDirectoryListsNothing(self, tmp_path):
        provider = PiperProvider(modelsDirectory=tmp_path / "nothing here")

        assert provider.availableVoices() == []

    def testItDescribesItself(self, tmp_path):
        description = PiperProvider("en_GB-cori-medium", modelsDirectory=tmp_path).describe()

        assert "piper" in description
        assert "en_GB-cori-medium" in description
