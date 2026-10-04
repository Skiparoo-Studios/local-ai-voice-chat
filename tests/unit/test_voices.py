"""Voice sample discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.audio import AudioBuffer
from app.speech.voices import VoiceLibrary, VoiceNotFoundError


def makeSample(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    AudioBuffer.silence(0.1, 22050).writeWav(path)
    return path


@pytest.fixture
def library(tmp_path: Path) -> VoiceLibrary:
    return VoiceLibrary(tmp_path / "voices")


class TestResolution:
    def testResolvesSingleFileVoice(self, library: VoiceLibrary):
        makeSample(library.root / "michael.wav")

        voice = library.get("michael")

        assert voice.name == "michael"
        assert voice.sampleCount == 1

    def testResolvesDirectoryVoiceWithSeveralSamples(self, library: VoiceLibrary):
        makeSample(library.root / "michael" / "one.wav")
        makeSample(library.root / "michael" / "two.wav")

        voice = library.get("michael")

        assert voice.sampleCount == 2
        assert [path.name for path in voice.samples] == ["one.wav", "two.wav"]

    def testDirectoryTakesPrecedenceOverFile(self, library: VoiceLibrary):
        makeSample(library.root / "michael.wav")
        makeSample(library.root / "michael" / "better.wav")

        assert library.get("michael").samples[0].name == "better.wav"

    def testOtherAudioExtensionsAreAccepted(self, library: VoiceLibrary):
        makeSample(library.root / "michael.flac")

        assert library.get("michael").sampleCount == 1

    def testEmptyDirectoryIsNotAVoice(self, library: VoiceLibrary):
        (library.root / "empty").mkdir(parents=True)

        with pytest.raises(VoiceNotFoundError):
            library.get("empty")

    def testHasReportsExistence(self, library: VoiceLibrary):
        makeSample(library.root / "michael.wav")

        assert library.has("michael")
        assert not library.has("someone-else")


class TestListing:
    def testListsFileAndDirectoryVoices(self, library: VoiceLibrary):
        makeSample(library.root / "alice.wav")
        makeSample(library.root / "bob" / "sample.wav")

        assert library.listVoices() == ["alice", "bob"]

    def testMissingRootListsNothing(self, library: VoiceLibrary):
        assert library.listVoices() == []

    def testCacheDirectoryIsNotAVoice(self, library: VoiceLibrary):
        makeSample(library.root / ".cache" / "michael-abc.pth")
        makeSample(library.root / "alice.wav")

        assert library.listVoices() == ["alice"]

    def testNonAudioFilesAreIgnored(self, library: VoiceLibrary):
        library.root.mkdir(parents=True)
        (library.root / "notes.txt").write_text("hello", encoding="utf-8")

        assert library.listVoices() == []


class TestErrorMessages:
    def testMissingVoiceWithNoLibraryExplainsHowToAddOne(self, library: VoiceLibrary):
        with pytest.raises(VoiceNotFoundError, match="no voice samples"):
            library.get("michael")

    def testMissingVoiceSuggestsCloseMatches(self, library: VoiceLibrary):
        makeSample(library.root / "michael.wav")

        with pytest.raises(VoiceNotFoundError, match="Did you mean: michael"):
            library.get("micheal")

    def testMissingVoiceListsAvailableOnes(self, library: VoiceLibrary):
        makeSample(library.root / "alice.wav")

        with pytest.raises(VoiceNotFoundError, match="Available voices: alice"):
            library.get("zebra")

    def testEmptyNameIsRejected(self, library: VoiceLibrary):
        with pytest.raises(VoiceNotFoundError, match="No voice name"):
            library.get("")


class TestCacheDirectory:
    def testCacheDirectorySitsUnderTheVoicesRoot(self, library: VoiceLibrary):
        assert library.cacheDirectory.parent == library.root
        assert library.cacheDirectory.name == ".cache"
