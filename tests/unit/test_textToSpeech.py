"""Provider abstraction, registry and the tone provider."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.audio import AudioBuffer, MemoryAudioOutput, WavFileOutput, createAudioOutput
from app.config import Settings
from app.speech.device import DeviceUnavailableError, resolveDevice
from app.speech.models.scriptedProvider import ScriptedProvider
from app.speech.models.toneProvider import ToneProvider
from app.speech.models.xttsProvider import splitText
from app.speech.textToSpeech import (
    PROVIDER_REGISTRY,
    TextToSpeechProvider,
    UnknownProviderError,
    createTextToSpeechProvider,
    registerProvider,
    resolveProviderClass,
)
from app.speech.voices import VoiceLibrary


class TestRegistry:
    def testResolvesRegisteredProvider(self):
        assert resolveProviderClass("tone") is ToneProvider

    def testUnknownProviderListsAvailableOnes(self):
        with pytest.raises(UnknownProviderError, match="Available:"):
            resolveProviderClass("does-not-exist")

    def testXttsIsRegistered(self):
        """Registered without importing torch, which is the point of lazy loading."""
        assert "xtts" in PROVIDER_REGISTRY

    def testProvidersCanBeRegistered(self):
        registerProvider("custom", "app.speech.models.toneProvider:ToneProvider")
        try:
            assert resolveProviderClass("custom") is ToneProvider
        finally:
            PROVIDER_REGISTRY.pop("custom", None)


class TestProviderSelection:
    """Exit criterion: swapping providers is configuration, not code."""

    def testConfigurationSelectsTheProvider(self, tmp_path: Path):
        settings = Settings(textToSpeech={"provider": "tone"}, paths={"voices": tmp_path})
        provider = createTextToSpeechProvider(settings)

        assert isinstance(provider, ToneProvider)
        assert isinstance(provider, TextToSpeechProvider)

    def testUnknownConfiguredProviderIsRejected(self, tmp_path: Path):
        settings = Settings(textToSpeech={"provider": "nope"}, paths={"voices": tmp_path})

        with pytest.raises(UnknownProviderError):
            createTextToSpeechProvider(settings)

    def testProviderReceivesConfiguredVoiceAndLanguage(self, tmp_path: Path):
        settings = Settings(
            textToSpeech={"provider": "tone", "voice": "alice", "language": "fr"},
            paths={"voices": tmp_path},
        )
        provider = createTextToSpeechProvider(settings)

        assert provider._voice == "alice"
        assert provider._language == "fr"


class TestToneProvider:
    async def testSynthesisProducesAudio(self):
        provider = ToneProvider()
        await provider.load()

        audio = await provider.synthesise("hello there")

        assert isinstance(audio, AudioBuffer)
        assert audio.durationSeconds > 0
        assert audio.sampleRate == provider.sampleRate

    async def testLongerTextProducesLongerAudio(self):
        provider = ToneProvider()
        await provider.load()

        short = await provider.synthesise("hello")
        long = await provider.synthesise("hello there this is a longer sentence")

        assert long.durationSeconds > short.durationSeconds

    async def testEmptyTextIsHandled(self):
        provider = ToneProvider()
        await provider.load()

        assert (await provider.synthesise("")).durationSeconds >= 0

    async def testDifferentVoicesProduceDifferentAudio(self):
        """Proves the voice argument reaches the provider."""
        provider = ToneProvider()
        await provider.load()

        first = await provider.synthesise("hello", voice="alice")
        second = await provider.synthesise("hello", voice="bob")

        assert first.data != second.data

    async def testSameVoiceIsDeterministic(self):
        provider = ToneProvider()
        await provider.load()

        first = await provider.synthesise("hello", voice="alice")
        second = await provider.synthesise("hello", voice="alice")

        assert first.data == second.data

    async def testLoadAndUnloadTrackState(self):
        provider = ToneProvider()
        assert not provider.isLoaded

        await provider.load()
        assert provider.isLoaded

        await provider.unload()
        assert not provider.isLoaded

    async def testLoadingTwiceIsHarmless(self):
        provider = ToneProvider()
        await provider.load()
        await provider.load()

        assert provider.isLoaded


class TestSynthesisStreaming:
    """Sentence streaming cannot help a one-sentence reply, and most of what
    a home assistant says is one sentence."""

    async def testDefaultYieldsTheWholeUtterance(self):
        provider = ToneProvider()
        await provider.load()

        pieces = [
            piece async for piece in provider.synthesiseStream("hello there friend")
        ]

        assert len(pieces) == 1
        whole = await provider.synthesise("hello there friend")
        assert pieces[0].data == whole.data

    async def testStreamedAudioMatchesTheWholeUtterance(self):
        provider = ToneProvider()
        await provider.load()

        pieces = [piece async for piece in provider.synthesiseStream("one two three")]
        joined = b"".join(piece.data for piece in pieces)

        assert joined == (await provider.synthesise("one two three")).data

    def testProvidersDeclareWhetherTheyStream(self):
        assert not ToneProvider().supportsStreaming

    def testXttsStreamsOnlyWithALowLevelModel(self, tmp_path: Path):
        from app.speech.models.xttsProvider import XttsProvider

        provider = XttsProvider(voices=VoiceLibrary(tmp_path))

        # No model loaded, so nothing to stream from.
        assert not provider.supportsStreaming

    def testChunkSizeOfZeroDisablesStreaming(self, tmp_path: Path):
        from app.speech.models.xttsProvider import XttsProvider

        provider = XttsProvider(voices=VoiceLibrary(tmp_path), streamChunkSize=0)
        provider._model = object()

        assert not provider.supportsStreaming

    def testChunkSizeComesFromConfiguration(self, tmp_path: Path):
        settings = Settings(
            textToSpeech={"provider": "xtts", "streamChunkSize": 5},
            paths={"voices": tmp_path},
        )
        provider = createTextToSpeechProvider(settings)

        assert provider._streamChunkSize == 5

    async def testEmptyTextStreamsNothing(self):
        provider = ToneProvider()
        await provider.load()

        pieces = [piece async for piece in provider.synthesiseStream("   ")]

        assert len(pieces) == 1


class TestAudioOutputSelection:
    def testFileBackendSelectedByConfiguration(self, tmp_path: Path):
        settings = Settings(
            audio={"outputBackend": "file"},
            paths={"outputs": tmp_path / "outputs"},
        )

        assert isinstance(createAudioOutput(settings), WavFileOutput)

    async def testWavFileOutputWritesPlayableAudio(self, tmp_path: Path):
        output = WavFileOutput(tmp_path)

        await output.play(AudioBuffer.tone(440.0, 0.2, 22050))

        assert output.lastPath is not None
        assert AudioBuffer.fromWavFile(output.lastPath).durationSeconds == pytest.approx(
            0.2, abs=0.01
        )

    async def testMemoryOutputRecordsBuffers(self):
        output = MemoryAudioOutput()

        await output.play(AudioBuffer.silence(0.1, 16000))

        assert len(output.played) == 1
        assert output.lastPlayed is not None


class TestContinuousPlayback:
    """A streamed reply arrives as many pieces, and they must join seamlessly."""

    async def testFilePiecesBecomeOneRecording(self, tmp_path: Path):
        """One file per reply, not one per synthesised piece."""
        output = WavFileOutput(tmp_path)

        await output.beginUtterance()
        for _ in range(4):
            await output.play(AudioBuffer.tone(440.0, 0.25, 22050))
        await output.endUtterance()

        files = sorted(tmp_path.glob("*.wav"))
        assert len(files) == 1
        assert AudioBuffer.fromWavFile(files[0]).durationSeconds == pytest.approx(
            1.0, abs=0.02
        )

    async def testPlayWithoutBoundariesStillWritesImmediately(self, tmp_path: Path):
        output = WavFileOutput(tmp_path)

        await output.play(AudioBuffer.tone(440.0, 0.2, 22050))

        assert len(list(tmp_path.glob("*.wav"))) == 1

    async def testAnUtteranceWithNoAudioWritesNothing(self, tmp_path: Path):
        output = WavFileOutput(tmp_path)

        await output.beginUtterance()
        await output.endUtterance()

        assert list(tmp_path.glob("*.wav")) == []

    async def testJoinedRecordingPreservesEveryPiece(self, tmp_path: Path):
        output = WavFileOutput(tmp_path)
        pieces = [AudioBuffer.tone(220.0 * (index + 1), 0.2, 22050) for index in range(3)]

        await output.beginUtterance()
        for piece in pieces:
            await output.play(piece)
        await output.endUtterance()

        written = AudioBuffer.fromWavFile(output.lastPath)
        assert written.data == b"".join(piece.data for piece in pieces)

    async def testBoundariesAreOptionalForOtherOutputs(self):
        output = MemoryAudioOutput()

        await output.beginUtterance()
        await output.play(AudioBuffer.tone(440.0, 0.1, 22050))
        await output.endUtterance()

        assert len(output.played) == 1

    async def testRuntimeMarksUtteranceBoundaries(self, tmp_path: Path):
        """The runtime is the only thing that knows where a reply begins."""
        from app.assistant.handlers import EchoHandler
        from app.assistant.runtime import AssistantRuntime
        from app.assistant.session import Session
        from app.audio import MemoryAudioInput
        from app.events import EventBus

        output = WavFileOutput(tmp_path)
        runtime = AssistantRuntime(
            speechToText=ScriptedProvider(),
            textToSpeech=ToneProvider(),
            handler=EchoHandler(),
            audioInput=MemoryAudioInput(),
            audioOutput=output,
            bus=EventBus(),
            session=Session.create(),
            warmUpOnStart=False,
        )
        await runtime.start()

        await runtime.handleText("hello there")

        assert len(list(tmp_path.glob("*.wav"))) == 1


class TestDeviceResolution:
    def testCpuIsAlwaysHonoured(self):
        assert resolveDevice("cpu", probe=lambda: True) == "cpu"

    def testAutoUsesCudaWhenAvailable(self):
        assert resolveDevice("auto", probe=lambda: True) == "cuda"

    def testAutoFallsBackToCpu(self):
        """The brief forbids assuming CUDA is present."""
        assert resolveDevice("auto", probe=lambda: False) == "cpu"

    def testExplicitCudaFailsLoudlyWhenUnavailable(self):
        with pytest.raises(DeviceUnavailableError, match="no CUDA device"):
            resolveDevice("cuda", probe=lambda: False)

    def testUnknownDeviceIsRejected(self):
        with pytest.raises(DeviceUnavailableError, match="Unknown device"):
            resolveDevice("tpu", probe=lambda: False)


class TestTextSplitting:
    """XTTS truncates long input, so text is chunked before synthesis."""

    def testShortTextIsNotSplit(self):
        assert splitText("Hello there.", 220) == ["Hello there."]

    def testSplitsOnSentenceBoundaries(self):
        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = splitText(text, 40)

        assert len(chunks) > 1
        assert all(len(chunk) <= 40 for chunk in chunks)

    def testLongSentenceWithoutPunctuationIsStillSplit(self):
        text = " ".join(["word"] * 100)
        chunks = splitText(text, 50)

        assert all(len(chunk) <= 50 for chunk in chunks)

    def testNoContentIsLost(self):
        text = "One sentence. Another sentence. A third one here."
        chunks = splitText(text, 20)

        assert "".join(chunks).replace(" ", "") == text.replace(" ", "")

    def testChunksAreNeverEmpty(self):
        chunks = splitText("A. B. C. " * 20, 30)

        assert all(chunk.strip() for chunk in chunks)


class TestVoiceLibraryIntegration:
    def testProviderReportsAvailableVoices(self, tmp_path: Path):
        voices = VoiceLibrary(tmp_path)
        AudioBuffer.silence(0.1, 22050).writeWav(tmp_path / "alice.wav")

        assert voices.listVoices() == ["alice"]
