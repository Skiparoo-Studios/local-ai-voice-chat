"""REPL behaviour, exercised through the tone provider."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.audio import AudioBuffer, MemoryAudioOutput
from app.events import EventBus, SpeechGenerationCompleted, SpeechGenerationStarted
from app.speech.models.toneProvider import ToneProvider
from app.speech.repl import SpeechRepl, SynthesisTiming, cleanInputLine
from app.speech.voices import VoiceLibrary


@pytest.fixture
async def repl(tmp_path: Path):
    provider = ToneProvider()
    await provider.load()
    output = MemoryAudioOutput()
    voices = VoiceLibrary(tmp_path / "voices")
    instance = SpeechRepl(
        provider,
        output,
        voices,
        EventBus(),
        voice="default",
        language="en",
        outputsDirectory=tmp_path / "outputs",
    )
    return instance, output


class TestSpeaking:
    async def testSpeakingSendsAudioToTheOutput(self, repl):
        instance, output = repl

        await instance.speakOnce("hello there")

        assert len(output.played) == 1
        assert output.played[0].durationSeconds > 0

    async def testSpeakingPublishesLifecycleEvents(self, tmp_path: Path):
        provider = ToneProvider()
        await provider.load()
        bus = EventBus()
        started: list[SpeechGenerationStarted] = []
        completed: list[SpeechGenerationCompleted] = []
        bus.subscribe(SpeechGenerationStarted, started.append)
        bus.subscribe(SpeechGenerationCompleted, completed.append)

        instance = SpeechRepl(
            provider,
            MemoryAudioOutput(),
            VoiceLibrary(tmp_path),
            bus,
            voice="default",
            language="en",
            outputsDirectory=tmp_path,
        )

        await instance.speakOnce("hello")

        assert [event.text for event in started] == ["hello"]
        assert completed[0].audioSeconds > 0
        assert completed[0].durationSeconds >= 0


class TestCommands:
    async def testQuitStopsTheLoop(self, repl):
        instance, _ = repl
        instance._running = True

        await instance._handleCommand(":quit")

        assert not instance._running

    async def testVoiceCommandSwitchesVoice(self, repl):
        instance, _ = repl

        await instance._handleCommand(":voice high")

        assert instance._voice == "high"

    async def testVoiceCommandRejectsUnknownVoice(self, repl, capsys):
        instance, _ = repl

        await instance._handleCommand(":voice nonexistent")

        assert instance._voice == "default"
        assert "unknown voice" in capsys.readouterr().out

    async def testSaveWritesTheLastUtterance(self, repl, tmp_path: Path):
        instance, _ = repl
        await instance.speakOnce("hello")

        await instance._handleCommand(f":save {tmp_path / 'saved.wav'}")

        saved = AudioBuffer.fromWavFile(tmp_path / "saved.wav")
        assert saved.durationSeconds > 0

    async def testSaveWithNothingSpokenIsHandled(self, repl, capsys):
        instance, _ = repl

        await instance._handleCommand(":save")

        assert "nothing to save" in capsys.readouterr().out

    async def testSaveWithoutPathDerivesOneFromTheText(self, repl, tmp_path: Path):
        instance, _ = repl
        await instance.speakOnce("hello there")

        await instance._handleCommand(":save")

        assert (tmp_path / "outputs" / "hello-there.wav").is_file()

    async def testUnknownCommandIsReported(self, repl, capsys):
        instance, _ = repl

        await instance._handleCommand(":nonsense")

        assert "unknown command" in capsys.readouterr().out

    async def testInfoReportsProviderDetails(self, repl, capsys):
        instance, _ = repl

        await instance._handleCommand(":info")

        output = capsys.readouterr().out
        assert "provider:" in output
        assert "tone" in output

    async def testVoicesCommandListsProviderVoices(self, repl, capsys):
        instance, _ = repl

        await instance._handleCommand(":voices")

        assert "default" in capsys.readouterr().out


class TestInputCleaning:
    """Windows shells prepend a BOM to the first line of piped input."""

    def testByteOrderMarkIsStripped(self):
        assert cleanInputLine("\ufeff:info") == ":info"

    def testZeroWidthSpaceIsStripped(self):
        assert cleanInputLine("\u200b:quit") == ":quit"

    def testOrdinaryWhitespaceIsStillStripped(self):
        assert cleanInputLine("  hello there  ") == "hello there"

    def testPlainTextIsUnchanged(self):
        assert cleanInputLine("Turn the kitchen light off.") == "Turn the kitchen light off."

    async def testCommandWithByteOrderMarkIsNotSpoken(self, repl):
        """The bug this guards: a prefixed ':info' was synthesised as speech."""
        instance, output = repl

        line = cleanInputLine("\ufeff:quit")
        instance._running = True
        if line.startswith(":"):
            await instance._handleCommand(line)
        else:
            await instance.speakOnce(line)

        assert not instance._running
        assert output.played == []


class TestTiming:
    def testRealTimeFactorIsSynthesisOverAudio(self):
        timing = SynthesisTiming(synthesisSeconds=0.5, audioSeconds=2.0)

        assert timing.realTimeFactor == pytest.approx(0.25)

    def testZeroLengthAudioDoesNotDivideByZero(self):
        assert SynthesisTiming(synthesisSeconds=0.5, audioSeconds=0.0).realTimeFactor == 0.0

    def testDescriptionMentionsBothDurations(self):
        description = SynthesisTiming(synthesisSeconds=0.5, audioSeconds=2.0).describe()

        assert "0.50s" in description
        assert "2.00s" in description
        assert "RTF" in description
