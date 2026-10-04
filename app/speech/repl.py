"""Interactive speech synthesis prompt.

The Stage 1 milestone: type text, hear it spoken, with the model loaded once
and reused for every utterance.
"""

from __future__ import annotations

import asyncio
import logging
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

from app.audio.audioBuffer import AudioBuffer
from app.audio.output import AudioOutput, AudioOutputError
from app.events import (
    AssistantState,
    AssistantStateChanged,
    EventBus,
    SpeechGenerationCompleted,
    SpeechGenerationStarted,
)
from app.speech.textToSpeech import SpeechSynthesisError, TextToSpeechProvider
from app.speech.voices import VoiceLibrary, VoiceNotFoundError

logger = logging.getLogger(__name__)

# Byte order mark and zero-width space. Neither counts as whitespace, so
# str.strip leaves them attached to the first character of a line.
INVISIBLE_PREFIXES = "\ufeff\u200b"

HELP_TEXT = """
Type text to have it spoken. Commands:

  :help            show this message
  :voices          list available voices
  :voice <name>    switch voice
  :save [path]     save the last utterance as a WAV file
  :info            show provider, device and voice
  :quit            exit (Ctrl-C and Ctrl-D also work)
""".strip()


def cleanInputLine(line: str) -> str:
    """Normalise a line read from the terminal.

    Windows shells prepend a byte order mark to the first line when input is
    piped rather than typed. It is not whitespace, so ``strip`` leaves it in
    place and a leading ``:command`` stops looking like a command and gets
    spoken aloud instead.
    """
    return line.lstrip(INVISIBLE_PREFIXES).strip()


@dataclass(slots=True)
class SynthesisTiming:
    """How long an utterance took, and how that compares to its length."""

    synthesisSeconds: float
    audioSeconds: float

    @property
    def realTimeFactor(self) -> float:
        """Synthesis time per second of audio. Below 1.0 is faster than real time."""
        if self.audioSeconds <= 0:
            return 0.0
        return self.synthesisSeconds / self.audioSeconds

    def describe(self) -> str:
        return (
            f"{self.synthesisSeconds:.2f}s for {self.audioSeconds:.2f}s of audio "
            f"(RTF {self.realTimeFactor:.2f})"
        )


class SpeechRepl:
    """Reads lines from the terminal and speaks them."""

    def __init__(
        self,
        provider: TextToSpeechProvider,
        output: AudioOutput,
        voices: VoiceLibrary,
        bus: EventBus,
        *,
        voice: str,
        language: str,
        outputsDirectory: Path,
    ) -> None:
        self._provider = provider
        self._output = output
        self._voices = voices
        self._bus = bus
        self._voice = voice
        self._language = language
        self._outputsDirectory = outputsDirectory
        self._lastAudio: AudioBuffer | None = None
        self._lastText: str = ""
        self._running = False

    async def run(self) -> int:
        """Loop until the user exits. Returns a process exit code."""
        self._running = True
        print(HELP_TEXT)
        print()

        while self._running:
            try:
                line = await asyncio.to_thread(input, "> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            line = cleanInputLine(line)
            if not line:
                continue

            try:
                if line.startswith(":"):
                    await self._handleCommand(line)
                else:
                    await self._speak(line)
            except KeyboardInterrupt:
                # Interrupt the utterance, not the session.
                print("\n(interrupted)")
                await self._output.stop()
            except (SpeechSynthesisError, AudioOutputError, VoiceNotFoundError) as error:
                print(f"error: {error}")

        return 0

    # --- Speaking --------------------------------------------------------

    async def speakOnce(self, text: str) -> AudioBuffer | None:
        """Speak a single line without entering the loop."""
        await self._speak(text)
        return self._lastAudio

    async def _speak(self, text: str) -> None:
        await self._bus.publish(
            SpeechGenerationStarted(text=text, voice=self._voice)
        )
        await self._bus.publish(AssistantStateChanged(state=AssistantState.speaking))

        started = time.perf_counter()
        audio = await self._provider.synthesise(
            text, voice=self._voice, language=self._language
        )
        timing = SynthesisTiming(
            synthesisSeconds=time.perf_counter() - started,
            audioSeconds=audio.durationSeconds,
        )

        self._lastAudio = audio
        self._lastText = text
        print(f"  {timing.describe()}")

        await self._bus.publish(
            SpeechGenerationCompleted(
                voice=self._voice,
                audioSeconds=timing.audioSeconds,
                durationSeconds=timing.synthesisSeconds,
            )
        )

        await self._output.play(audio)
        await self._bus.publish(AssistantStateChanged(state=AssistantState.idle))

    # --- Commands --------------------------------------------------------

    async def _handleCommand(self, line: str) -> None:
        command, _, argument = line[1:].partition(" ")
        command = command.lower()
        argument = argument.strip()

        if command in {"quit", "exit", "q"}:
            self._running = False
        elif command in {"help", "h", "?"}:
            print(HELP_TEXT)
        elif command == "voices":
            self._printVoices()
        elif command == "voice":
            self._switchVoice(argument)
        elif command == "save":
            self._save(argument)
        elif command == "info":
            self._printInfo()
        else:
            print(f"unknown command {command!r}; try :help")

    def _printVoices(self) -> None:
        print(f"  current: {self._voice}")

        available = self._provider.availableVoices()
        if not available:
            print(f"  no voices found in {self._voices.root}")
            print("  add a recording as <name>.wav, or several as <name>/*.wav")
            return

        # Providers with built-in speakers offer dozens, so wrap rather than
        # printing one per line.
        print(f"  available ({len(available)}):")
        for line in textwrap.wrap(", ".join(available), width=72):
            print(f"    {line}")

    def _switchVoice(self, name: str) -> None:
        if not name:
            print(f"  current voice: {self._voice}")
            return
        available = self._provider.availableVoices()
        if available and name not in available:
            print(f"  unknown voice {name!r}; available: {', '.join(available)}")
            return
        self._voice = name
        print(f"  voice set to {name}")

    def _save(self, argument: str) -> None:
        if self._lastAudio is None:
            print("  nothing to save yet")
            return

        path = Path(argument) if argument else self._defaultSavePath()
        self._lastAudio.writeWav(path)
        print(f"  saved {self._lastAudio.durationSeconds:.2f}s to {path}")

    def _defaultSavePath(self) -> Path:
        slug = "".join(
            character if character.isalnum() else "-" for character in self._lastText.lower()
        ).strip("-")[:40] or "utterance"
        return self._outputsDirectory / f"{slug}.wav"

    def _printInfo(self) -> None:
        print(f"  provider: {self._provider.describe()}")
        print(f"  output:   {self._output.describe()}")
        print(f"  voice:    {self._voice}")
        print(f"  language: {self._language}")
        print(f"  rate:     {self._provider.sampleRate} Hz")
