"""XTTS speaker resolution.

These tests substitute a fake model for the real one so that voice resolution,
which is ordinary logic, can be tested without torch or a 2 GB download. The
model's own inference is exercised by the integration tests instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.audio import AudioBuffer
from app.speech.models.xttsProvider import (
    DEFAULT_BUILT_IN_SPEAKER,
    SpeakerSource,
    XttsProvider,
)
from app.speech.voices import VoiceLibrary, VoiceNotFoundError

BUILT_IN_SPEAKERS = {
    DEFAULT_BUILT_IN_SPEAKER: {"gpt_cond_latent": "latent", "speaker_embedding": "embedding"},
    "Daisy Studious": {"gpt_cond_latent": "latent2", "speaker_embedding": "embedding2"},
}


class FakeSpeakerManager:
    def __init__(self, speakers: dict) -> None:
        self.speakers = speakers


class FakeModel:
    def __init__(self, speakers: dict | None = None) -> None:
        self.speaker_manager = FakeSpeakerManager(speakers if speakers is not None else {})


def makeProvider(tmp_path: Path, *, speakers: dict | None = BUILT_IN_SPEAKERS) -> XttsProvider:
    provider = XttsProvider(voices=VoiceLibrary(tmp_path / "voices"), voice="default")
    provider._model = FakeModel(speakers)
    return provider


def makeSample(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    AudioBuffer.silence(0.1, 22050).writeWav(path)
    return path


class TestBuiltInSpeakers:
    def testDefaultResolvesToABuiltInSpeaker(self, tmp_path: Path):
        """The assistant must have a working voice before anyone records one."""
        source = makeProvider(tmp_path).resolveSpeaker("default")

        assert source.builtIn
        assert source.name == DEFAULT_BUILT_IN_SPEAKER

    def testBuiltInSpeakerResolvesByName(self, tmp_path: Path):
        source = makeProvider(tmp_path).resolveSpeaker("Daisy Studious")

        assert source == SpeakerSource(name="Daisy Studious", builtIn=True)

    def testBuiltInConditioningComesFromTheModel(self, tmp_path: Path):
        provider = makeProvider(tmp_path)

        latent, embedding = provider._conditioningFor(
            SpeakerSource(name=DEFAULT_BUILT_IN_SPEAKER, builtIn=True)
        )

        assert (latent, embedding) == ("latent", "embedding")

    def testBuiltInConditioningIsCached(self, tmp_path: Path):
        provider = makeProvider(tmp_path)
        source = SpeakerSource(name=DEFAULT_BUILT_IN_SPEAKER, builtIn=True)

        provider._conditioningFor(source)

        assert DEFAULT_BUILT_IN_SPEAKER in provider._conditioningCache


class TestClonedVoices:
    def testLocalSampleResolvesToClonedVoice(self, tmp_path: Path):
        provider = makeProvider(tmp_path)
        makeSample(tmp_path / "voices" / "michael.wav")

        source = provider.resolveSpeaker("michael")

        assert not source.builtIn
        assert len(source.samples) == 1

    def testLocalSampleOverridesABuiltInOfTheSameName(self, tmp_path: Path):
        """Adding a recording must be all it takes to replace a default."""
        provider = makeProvider(tmp_path)
        makeSample(tmp_path / "voices" / f"{DEFAULT_BUILT_IN_SPEAKER}.wav")

        source = provider.resolveSpeaker(DEFAULT_BUILT_IN_SPEAKER)

        assert not source.builtIn

    def testLocalDefaultOverridesTheBuiltInAlias(self, tmp_path: Path):
        provider = makeProvider(tmp_path)
        makeSample(tmp_path / "voices" / "default.wav")

        source = provider.resolveSpeaker("default")

        assert not source.builtIn
        assert source.name == "default"


class TestResolutionFailures:
    def testUnknownVoiceMentionsBuiltInSpeakers(self, tmp_path: Path):
        provider = makeProvider(tmp_path)

        with pytest.raises(VoiceNotFoundError, match="Built-in speakers"):
            provider.resolveSpeaker("nobody")

    def testUnknownVoiceWithNoBuiltInsFallsBackToTheLibraryMessage(self, tmp_path: Path):
        provider = makeProvider(tmp_path, speakers={})

        with pytest.raises(VoiceNotFoundError, match="no voice samples"):
            provider.resolveSpeaker("nobody")

    def testDefaultFailsWhenNoBuiltInsAndNoSamples(self, tmp_path: Path):
        provider = makeProvider(tmp_path, speakers={})

        with pytest.raises(VoiceNotFoundError):
            provider.resolveSpeaker("default")


class TestAvailableVoices:
    def testListsClonedVoicesBeforeBuiltInSpeakers(self, tmp_path: Path):
        provider = makeProvider(tmp_path)
        makeSample(tmp_path / "voices" / "michael.wav")

        available = provider.availableVoices()

        assert available[0] == "michael"
        assert DEFAULT_BUILT_IN_SPEAKER in available

    def testNamesAreNotDuplicated(self, tmp_path: Path):
        provider = makeProvider(tmp_path)
        makeSample(tmp_path / "voices" / f"{DEFAULT_BUILT_IN_SPEAKER}.wav")

        available = provider.availableVoices()

        assert available.count(DEFAULT_BUILT_IN_SPEAKER) == 1

    def testNoModelMeansNoBuiltInSpeakers(self, tmp_path: Path):
        provider = XttsProvider(voices=VoiceLibrary(tmp_path / "voices"))

        assert provider.availableVoices() == []


class TestClonedConditioning:
    """The cloned-voice path, which built-in speakers return before reaching."""

    def testConditioningIsComputedFromTheRecordings(self, tmp_path: Path):
        provider = makeProvider(tmp_path)
        provider._model = _RecordingModel()
        provider._torch = _FakeTorch()
        provider._device = "cpu"
        makeSample(tmp_path / "voices" / "michael" / "one.wav")
        makeSample(tmp_path / "voices" / "michael" / "two.wav")

        source = provider.resolveSpeaker("michael")
        latent, embedding = provider._computeConditioning(source)

        assert latent == "latents"
        assert embedding == "embedding"
        assert len(provider._model.embeddingCalls) == 2

    def testSpeakerSourceReportsItsSampleCount(self):
        """Logging the count used an attribute only Voice had."""
        source = SpeakerSource(name="michael", samples=(Path("a.wav"), Path("b.wav")))

        assert source.sampleCount == 2
        assert SpeakerSource(name="built-in", builtIn=True).sampleCount == 0


class _RecordingModel:
    """A model that records what it was conditioned on."""

    def __init__(self) -> None:
        self.speaker_manager = FakeSpeakerManager(BUILT_IN_SPEAKERS)
        self.embeddingCalls: list[object] = []

    def get_speaker_embedding(self, audio, rate):
        self.embeddingCalls.append(audio)
        return "embedding"

    def get_gpt_cond_latents(self, audio, rate, length, chunk_length):
        return "latents"


class _FakeTorch:
    """Enough of torch for the conditioning path."""

    float32 = "float32"

    class _Tensor:
        def __init__(self, values) -> None:
            self.values = values

        def unsqueeze(self, _dimension):
            return self

        def clip(self, _low, _high):
            return self

        def to(self, _device):
            return self

        def __getitem__(self, _item):
            return self

    def tensor(self, values, dtype=None):
        return self._Tensor(values)

    def cat(self, tensors, dim):
        return tensors[0]

    def stack(self, values):
        return _FakeStack(values)

    def inference_mode(self):
        import contextlib

        return contextlib.nullcontext()


class _FakeStack:
    def __init__(self, values) -> None:
        self.values = values

    def mean(self, dim):
        return self.values[0]


class TestConditioningSettings:
    """The method defaults ignore most of a recording; the config does not."""

    def testSettingsComeFromTheModelConfig(self, tmp_path: Path):
        provider = makeProvider(tmp_path)
        provider._model.config = _FakeConfig()

        settings = provider._conditioningSettings()

        assert settings["gpt_cond_len"] == 30
        assert settings["max_ref_length"] == 30
        assert settings["gpt_cond_chunk_len"] == 4
        assert settings["sound_norm_refs"] is False

    def testAbsentConfigYieldsNoOverrides(self, tmp_path: Path):
        provider = makeProvider(tmp_path)
        provider._model.config = None

        assert provider._conditioningSettings() == {}

    def testCacheKeyChangesWithTheSettings(self, tmp_path: Path):
        """Embeddings computed under different settings are different things."""
        makeSample(tmp_path / "voices" / "michael.wav")

        first = makeProvider(tmp_path)
        first._model.config = _FakeConfig()
        second = makeProvider(tmp_path)
        second._model.config = _FakeConfig(gptCondLen=6)

        source = first.resolveSpeaker("michael")
        assert first._cachePathFor(source) != second._cachePathFor(source)

    def testBuiltInSpeakersNeedNoConditioningSettings(self, tmp_path: Path):
        provider = makeProvider(tmp_path)
        provider._model.config = _FakeConfig()

        latent, embedding = provider._conditioningFor(
            SpeakerSource(name=DEFAULT_BUILT_IN_SPEAKER, builtIn=True)
        )

        assert (latent, embedding) == ("latent", "embedding")


class _FakeConfig:
    """Stands in for the XTTS config object."""

    def __init__(self, gptCondLen: int = 30) -> None:
        self.gpt_cond_len = gptCondLen
        self.gpt_cond_chunk_len = 4
        self.max_ref_len = 30
        self.sound_norm_refs = False


class TestWaveformConversion:
    """inference returns numpy; inference_stream yields tensors on the GPU."""

    def testNumpyArrayIsConverted(self):
        numpy = pytest.importorskip("numpy")
        from app.speech.models.xttsProvider import _waveformToBuffer

        audio = _waveformToBuffer(numpy.zeros(2400, dtype=numpy.float32), 24000)

        assert audio.frameCount == 2400
        assert audio.sampleRate == 24000

    def testTensorIsBroughtBackToTheHost(self):
        """A CUDA tensor cannot be handed to numpy directly."""
        from app.speech.models.xttsProvider import _waveformToBuffer

        class FakeTensor:
            """Stands in for a tensor that must be detached and moved."""

            def __init__(self) -> None:
                self.detached = False

            def detach(self):
                self.detached = True
                return self

            def cpu(self):
                import numpy

                return numpy.zeros(1200, dtype="float32")

        tensor = FakeTensor()
        audio = _waveformToBuffer(tensor, 24000)

        assert tensor.detached
        assert audio.frameCount == 1200

    def testValuesAreClipped(self):
        numpy = pytest.importorskip("numpy")
        from app.speech.models.xttsProvider import _waveformToBuffer

        audio = _waveformToBuffer(numpy.array([2.0, -2.0], dtype=numpy.float32), 24000)
        samples = audio.toFloatSamples()

        assert samples[0] == pytest.approx(1.0, abs=1e-4)
        assert samples[1] == pytest.approx(-1.0, abs=1e-4)


class TestLifecycle:
    async def testSynthesisBeforeLoadingIsRejected(self, tmp_path: Path):
        provider = XttsProvider(voices=VoiceLibrary(tmp_path))

        with pytest.raises(Exception, match="not loaded"):
            await provider.synthesise("hello")

    def testDescribeMentionsTheUnresolvedDevice(self, tmp_path: Path):
        provider = XttsProvider(voices=VoiceLibrary(tmp_path), device="auto")

        assert "unresolved" in provider.describe()
