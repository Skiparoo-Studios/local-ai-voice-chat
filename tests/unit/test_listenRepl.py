"""Transcription prompt behaviour, exercised through the scripted provider."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.audio import AudioBuffer, MemoryAudioInput
from app.events import EventBus, TranscriptionCompleted, TranscriptionStarted
from app.speech.listenRepl import ListenRepl
from app.speech.models.scriptedProvider import ScriptedProvider


@pytest.fixture
async def repl(tmp_path: Path):
    provider = ScriptedProvider(phrases=("turn the kitchen light off",))
    await provider.load()
    source = MemoryAudioInput([AudioBuffer.tone(440.0, 1.0, 16000)])
    instance = ListenRepl(
        provider,
        source,
        EventBus(),
        language="en",
        maximumSeconds=30.0,
        outputsDirectory=tmp_path / "outputs",
    )
    return instance, source


class TestTranscription:
    async def testTranscribesCapturedAudio(self, repl, capsys):
        instance, _ = repl

        transcript = await instance.transcribe(AudioBuffer.tone(440.0, 1.0, 16000))

        assert transcript.text == "turn the kitchen light off"
        assert "turn the kitchen light off" in capsys.readouterr().out

    async def testPublishesTranscriptionEvents(self, tmp_path: Path):
        provider = ScriptedProvider(phrases=("hello",))
        await provider.load()
        bus = EventBus()
        started: list[TranscriptionStarted] = []
        completed: list[TranscriptionCompleted] = []
        bus.subscribe(TranscriptionStarted, started.append)
        bus.subscribe(TranscriptionCompleted, completed.append)

        instance = ListenRepl(
            provider,
            MemoryAudioInput(),
            bus,
            language="en",
            maximumSeconds=30.0,
            outputsDirectory=tmp_path,
        )

        await instance.transcribe(AudioBuffer.tone(440.0, 1.0, 16000))

        assert len(started) == 1
        assert completed[0].text == "hello"

    async def testEmptyTranscriptIsReported(self, tmp_path: Path, capsys):
        provider = ScriptedProvider(phrases=("",))
        await provider.load()
        instance = ListenRepl(
            provider,
            MemoryAudioInput(),
            EventBus(),
            language="en",
            maximumSeconds=30.0,
            outputsDirectory=tmp_path,
        )

        await instance.transcribe(AudioBuffer.silence(1.0, 16000))

        assert "no speech detected" in capsys.readouterr().out


class TestFileTranscription:
    async def testTranscribesAWavFile(self, repl, tmp_path: Path):
        instance, _ = repl
        path = tmp_path / "utterance.wav"
        AudioBuffer.tone(440.0, 1.0, 16000).writeWav(path)

        transcript = await instance.transcribeFile(path)

        assert transcript.text == "turn the kitchen light off"

    async def testResamplesAFileToTheProviderRate(self, repl, tmp_path: Path):
        """A 44 kHz file must not reach a provider that requires 16 kHz."""
        instance, _ = repl
        path = tmp_path / "utterance.wav"
        AudioBuffer.tone(440.0, 1.0, 44100).writeWav(path)

        await instance.transcribeFile(path)

        assert instance._lastAudio is not None
        assert instance._lastAudio.sampleRate == 16000

    async def testMissingFileIsReported(self, repl, tmp_path: Path):
        instance, _ = repl

        with pytest.raises(Exception, match="not found"):
            await instance.transcribeFile(tmp_path / "absent.wav")


class TestCommands:
    async def testQuitStopsTheLoop(self, repl):
        instance, _ = repl
        instance._running = True

        await instance._handleCommand(":quit")

        assert not instance._running

    async def testSaveWritesTheLastRecording(self, repl, tmp_path: Path):
        instance, _ = repl
        instance._lastAudio = AudioBuffer.tone(440.0, 0.5, 16000)

        await instance._handleCommand(f":save {tmp_path / 'saved.wav'}")

        assert AudioBuffer.fromWavFile(tmp_path / "saved.wav").durationSeconds > 0

    async def testSaveWithNothingRecordedIsHandled(self, repl, capsys):
        instance, _ = repl

        await instance._handleCommand(":save")

        assert "nothing to save" in capsys.readouterr().out

    async def testFileCommandWithoutArgumentShowsUsage(self, repl, capsys):
        instance, _ = repl

        await instance._handleCommand(":file")

        assert "usage:" in capsys.readouterr().out

    async def testInfoReportsProviderDetails(self, repl, capsys):
        instance, _ = repl

        await instance._handleCommand(":info")

        output = capsys.readouterr().out
        assert "provider:" in output
        assert "scripted" in output

    async def testUnknownCommandIsReported(self, repl, capsys):
        instance, _ = repl

        await instance._handleCommand(":nonsense")

        assert "unknown command" in capsys.readouterr().out
