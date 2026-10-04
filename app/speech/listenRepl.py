"""Interactive transcription prompt.

The Stage 2 milestone: press Enter, speak, press Enter again, see the
transcript. Capture stops on a signal rather than after a fixed duration, so
Stage 6 can replace the "press Enter" trigger with voice activity detection
without changing the capture layer.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from app.audio.audioBuffer import AudioBuffer
from app.audio.input import AudioInput, AudioInputError, resampleBuffer
from app.events import (
    AssistantState,
    AssistantStateChanged,
    EventBus,
    SpeechEnded,
    SpeechStarted,
    TranscriptionCompleted,
    TranscriptionStarted,
)
from app.speech.repl import cleanInputLine
from app.speech.speechToText import SpeechToTextProvider, Transcript, TranscriptionError

logger = logging.getLogger(__name__)

HELP_TEXT = """
Press Enter to start recording, then Enter again to stop. Commands:

  :help            show this message
  :save [path]     save the last recording as a WAV file
  :file <path>     transcribe a WAV file instead of the microphone
  :info            show provider, device and language
  :quit            exit (Ctrl-C and Ctrl-D also work)
""".strip()


class ListenRepl:
    """Records from the microphone and prints what was said."""

    def __init__(
        self,
        provider: SpeechToTextProvider,
        audioInput: AudioInput,
        bus: EventBus,
        *,
        language: str | None,
        maximumSeconds: float,
        outputsDirectory: Path,
    ) -> None:
        self._provider = provider
        self._input = audioInput
        self._bus = bus
        self._language = language
        self._maximumSeconds = maximumSeconds
        self._outputsDirectory = outputsDirectory
        self._lastAudio: AudioBuffer | None = None
        self._running = False

    async def run(self) -> int:
        self._running = True
        print(HELP_TEXT)
        print()

        while self._running:
            try:
                line = await asyncio.to_thread(input, "[Enter to record] > ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            line = cleanInputLine(line)

            try:
                if line.startswith(":"):
                    await self._handleCommand(line)
                elif line:
                    print("  type nothing and press Enter to record; :help for commands")
                else:
                    await self.captureAndTranscribe()
            except KeyboardInterrupt:
                print("\n(interrupted)")
            except (TranscriptionError, AudioInputError) as error:
                print(f"error: {error}")

        return 0

    # --- Capture ---------------------------------------------------------

    async def captureAndTranscribe(self) -> Transcript:
        """Record until the user presses Enter, then transcribe."""
        audio = await self._record()
        self._lastAudio = audio

        if not audio.data:
            print("  nothing was recorded")
            return Transcript(text="")

        print(f"  captured {audio.durationSeconds:.2f}s")
        return await self.transcribe(audio)

    async def _record(self) -> AudioBuffer:
        stop = asyncio.Event()

        await self._bus.publish(AssistantStateChanged(state=AssistantState.listening))
        await self._bus.publish(SpeechStarted())

        recording = asyncio.create_task(
            self._input.record(stopWhen=stop, maximumSeconds=self._maximumSeconds)
        )

        # Wait for a second Enter, but do not hang if capture stops itself
        # after reaching the maximum duration.
        waiting = asyncio.create_task(asyncio.to_thread(input, "  recording... [Enter to stop] "))
        await asyncio.wait({recording, waiting}, return_when=asyncio.FIRST_COMPLETED)

        stop.set()
        audio = await recording
        if not waiting.done():
            print("  (reached the maximum recording length)")
            waiting.cancel()

        await self._bus.publish(SpeechEnded(durationSeconds=audio.durationSeconds))
        return audio

    async def transcribe(self, audio: AudioBuffer) -> Transcript:
        await self._bus.publish(AssistantStateChanged(state=AssistantState.thinking))
        await self._bus.publish(TranscriptionStarted())

        started = time.perf_counter()
        transcript = await self._provider.transcribe(audio, language=self._language)
        elapsed = time.perf_counter() - started

        if transcript.isEmpty:
            print("  (no speech detected)")
        else:
            print(f"  {transcript.text}")
        print(f"  {transcript.describe()}" if transcript.audioSeconds else f"  {elapsed:.2f}s")

        await self._bus.publish(
            TranscriptionCompleted(
                text=transcript.text,
                language=transcript.language,
                durationSeconds=elapsed,
            )
        )
        await self._bus.publish(AssistantStateChanged(state=AssistantState.idle))
        return transcript

    async def transcribeFile(self, path: Path) -> Transcript:
        """Transcribe a WAV file, resampling it to what the provider needs."""
        if not path.is_file():
            raise AudioInputError(f"Audio file not found: {path}")

        buffer = await asyncio.to_thread(AudioBuffer.fromWavFile, path)
        audio = resampleBuffer(buffer, self._provider.requiredSampleRate)
        self._lastAudio = audio
        print(f"  {path} ({audio.durationSeconds:.2f}s)")
        return await self.transcribe(audio)

    # --- Commands --------------------------------------------------------

    async def _handleCommand(self, line: str) -> None:
        command, _, argument = line[1:].partition(" ")
        command = command.lower()
        argument = argument.strip()

        if command in {"quit", "exit", "q"}:
            self._running = False
        elif command in {"help", "h", "?"}:
            print(HELP_TEXT)
        elif command == "save":
            self._save(argument)
        elif command == "file":
            if not argument:
                print("  usage: :file <path to a WAV file>")
            else:
                await self.transcribeFile(Path(argument))
        elif command == "info":
            self._printInfo()
        else:
            print(f"unknown command {command!r}; try :help")

    def _save(self, argument: str) -> None:
        if self._lastAudio is None:
            print("  nothing to save yet")
            return

        path = Path(argument) if argument else self._outputsDirectory / "recording.wav"
        self._lastAudio.writeWav(path)
        print(f"  saved {self._lastAudio.durationSeconds:.2f}s to {path}")

    def _printInfo(self) -> None:
        print(f"  provider: {self._provider.describe()}")
        print(f"  input:    {self._input.describe()}")
        print(f"  language: {self._language or 'auto-detect'}")
        print(f"  rate:     {self._provider.requiredSampleRate} Hz")
