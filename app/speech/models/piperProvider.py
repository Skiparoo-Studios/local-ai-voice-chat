"""Piper text-to-speech.

The provider abstraction exists so that the synthesiser is a choice rather than
an assumption, and this is the second real implementation that proves it. It
answers two problems XTTS has.

The first is licensing. XTTS-v2 is under the Coqui Public Model Licence, which
restricts the model *and its output* to non-commercial use. Piper's voices are
published under their own terms --- mostly MIT or Creative Commons --- and
nothing here restricts what the generated audio may be used for. The Piper
engine itself is GPL-3.0, which is a copyleft obligation on anyone distributing
a combined work, not a limit on use or output. See NOTICE.md, and note that
neither the engine nor its voices are distributed with this project.

The second is hardware. XTTS wants a GPU and takes 26.6 s to load. Piper is an
ONNX model that loads in about two seconds and synthesises at roughly a
seventeenth of real time on a CPU, which is what makes it the sensible choice
on a Raspberry Pi, or on any machine where the GPU is busy running the language
model.

What it gives up is voice cloning. Piper voices are trained, not conditioned on
a recording, so the thirty seconds of your own speech that XTTS turns into a
voice has no equivalent here. Pick the provider that matches what you want:
your own voice, or a permissive licence and a CPU.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from app.audio.audioBuffer import AudioBuffer
from app.speech.textToSpeech import (
    ProviderUnavailableError,
    SpeechSynthesisError,
    TextToSpeechProvider,
)
from app.speech.voices import VoiceLibrary

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from app.config import Settings

logger = logging.getLogger(__name__)

# "medium" is the quality tier: low is 20 MB and audibly worse, high is slower
# for a difference most people do not notice through a kitchen speaker.
#
# cori rather than alba, which was the first choice. Synthesising "what sound
# does a sheep make?" and transcribing it back, alba --- a Scottish voice ---
# produced "ship" on one run in two, where cori was right both times. Toddler
# mode asks that exact question, so the vowel matters more here than it looks.
DEFAULT_VOICE = "en_GB-cori-medium"

# Piper voices live in their own directory under paths.models, so they do not
# mix with whatever coqui-tts puts in TTS_HOME.
VOICES_SUBDIRECTORY = "piper"

# A multi-speaker voice selects its speaker after a hash: "en_US-libritts-high#42".
SPEAKER_SEPARATOR = "#"

# Deep enough to keep the speaker fed, shallow enough that a fast synthesiser
# does not run far ahead of playback and hold a whole reply in memory.
STREAM_QUEUE_DEPTH = 8

MISSING_PACKAGE = (
    "Piper text-to-speech requires the 'piper-tts' package.\n"
    'Install with: pip install -e ".[piper]"\n'
    "\n"
    "Note that piper-tts is GPL-3.0-or-later, because it bundles espeak-ng as "
    "its phonemiser. That is a copyleft obligation on distributing a combined "
    "work; it does not restrict use, and it does not restrict the audio it "
    "generates. See NOTICE.md."
)


class PiperProvider(TextToSpeechProvider):
    """Synthesises with a Piper ONNX voice, on the CPU."""

    name: ClassVar[str] = "piper"

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        *,
        modelsDirectory: Path | None = None,
        allowDownload: bool = True,
        lengthScale: float | None = None,
    ) -> None:
        self._voiceName = voice or DEFAULT_VOICE
        self._modelsDirectory = modelsDirectory or Path("models") / VOICES_SUBDIRECTORY
        self._allowDownload = allowDownload
        self._lengthScale = lengthScale
        # One loaded voice per name. Switching voices mid-session is rare, but
        # reloading a 60 MB model on every turn to support it would not be.
        self._voices: dict[str, Any] = {}
        self._sampleRate = 22050

    @classmethod
    def fromSettings(cls, settings: Settings, voices: VoiceLibrary) -> PiperProvider:
        """Build from configuration.

        Piper voices are models rather than speaker recordings, so the
        VoiceLibrary --- which manages the reference audio XTTS conditions on
        --- has nothing to contribute and is accepted only to satisfy the
        interface.
        """
        configured = settings.textToSpeech.voice
        # "default" is the placeholder in settings.example.json and names no
        # Piper voice, so it falls through to one that exists.
        voice = DEFAULT_VOICE if configured in {"", "default"} else configured

        if settings.textToSpeech.device == "cuda":
            # Not an error, and not worth failing over. Piper synthesises at a
            # seventeenth of real time on a CPU, so a GPU buys nothing here and
            # would need onnxruntime-gpu installed to be used at all.
            logger.info("Piper runs on the CPU; textToSpeech.device is ignored")

        return cls(
            voice,
            modelsDirectory=settings.paths.models / VOICES_SUBDIRECTORY,
            allowDownload=settings.textToSpeech.allowVoiceDownload,
            lengthScale=settings.textToSpeech.speechRate or None,
        )

    # --- Properties -------------------------------------------------------

    @property
    def sampleRate(self) -> int:
        return self._sampleRate

    @property
    def isLoaded(self) -> bool:
        return bool(self._voices)

    @property
    def supportsStreaming(self) -> bool:
        # Piper emits one chunk per sentence, so a multi-sentence reply starts
        # playing after the first rather than the last. Measured, the first of
        # three sentences arrived in 0.11 s against 0.26 s for all of them.
        return True

    # --- Lifecycle --------------------------------------------------------

    async def load(self) -> None:
        if self._voiceName in self._voices:
            return
        await self._loadVoice(self._voiceName)

    async def unload(self) -> None:
        self._voices.clear()

    async def _loadVoice(self, name: str) -> Any:
        """Load a voice, downloading it first if it is not already here."""
        modelName, _ = _splitSpeaker(name)
        if modelName in self._voices:
            return self._voices[modelName]

        path = await asyncio.to_thread(self._resolveModelPath, modelName)
        voice = await asyncio.to_thread(self._loadBlocking, path)

        self._voices[modelName] = voice
        self._sampleRate = int(voice.config.sample_rate)
        logger.info(
            "Piper voice %s loaded (%d Hz, %d speaker(s))",
            modelName,
            self._sampleRate,
            voice.config.num_speakers,
        )
        return voice

    def _loadBlocking(self, path: Path) -> Any:
        piper = _importPiper()
        try:
            return piper.PiperVoice.load(path)
        except Exception as error:
            raise ProviderUnavailableError(f"Could not load Piper voice {path}: {error}") from error

    def _resolveModelPath(self, modelName: str) -> Path:
        """The .onnx for this voice, fetching it if it is absent."""
        path = self._modelsDirectory / f"{modelName}.onnx"
        if path.is_file():
            return path

        if not self._allowDownload:
            raise ProviderUnavailableError(
                f"Piper voice {modelName!r} is not in {self._modelsDirectory} and "
                "textToSpeech.allowVoiceDownload is false.\n"
                f"Fetch it with: python -m piper.download_voices {modelName}"
            )

        self._modelsDirectory.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading Piper voice %s to %s", modelName, self._modelsDirectory)

        try:
            from piper.download_voices import download_voice

            download_voice(modelName, self._modelsDirectory)
        except ImportError as error:
            raise ProviderUnavailableError(MISSING_PACKAGE) from error
        except Exception as error:
            raise ProviderUnavailableError(
                f"Could not download Piper voice {modelName!r}: {error}\n"
                "Check the name against: python -m piper.download_voices --list"
            ) from error

        if not path.is_file():
            raise ProviderUnavailableError(
                f"Piper reported success but {path} is missing"
            )
        return path

    # --- Synthesis --------------------------------------------------------

    async def synthesise(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
    ) -> AudioBuffer:
        """Generate a whole utterance.

        ``language`` is ignored: a Piper voice is trained for one language, so
        the voice name chooses it. Accepting the argument keeps the interface
        uniform rather than making callers know which provider they have.
        """
        cleaned = text.strip()
        if not cleaned:
            return AudioBuffer(data=b"", sampleRate=self._sampleRate)

        pieces = [audio async for audio in self.synthesiseStream(cleaned, voice=voice)]
        if not pieces:
            return AudioBuffer(data=b"", sampleRate=self._sampleRate)

        return AudioBuffer(
            data=b"".join(piece.data for piece in pieces),
            sampleRate=pieces[0].sampleRate,
            channels=pieces[0].channels,
        )

    async def synthesiseStream(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
    ) -> AsyncIterator[AudioBuffer]:
        """Yield one buffer per sentence, as each is generated.

        Piper's generator is synchronous and blocking, so it runs in a worker
        thread and hands pieces back through a bounded queue --- the same
        arrangement the XTTS provider uses, for the same reason.
        """
        cleaned = text.strip()
        if not cleaned:
            return

        requested = voice or self._voiceName
        loaded = await self._loadVoice(requested)
        _, speakerId = _splitSpeaker(requested)

        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=STREAM_QUEUE_DEPTH)
        loop = asyncio.get_running_loop()
        finished = object()

        def produce() -> None:
            try:
                for chunk in loaded.synthesize(cleaned, self._synthesisConfig(speakerId)):
                    audio = AudioBuffer(
                        data=chunk.audio_int16_bytes,
                        sampleRate=int(chunk.sample_rate),
                        channels=int(chunk.sample_channels),
                        sampleWidth=int(chunk.sample_width),
                    )
                    asyncio.run_coroutine_threadsafe(queue.put(audio), loop).result()
            except Exception as error:  # noqa: BLE001 - reported to the consumer
                asyncio.run_coroutine_threadsafe(queue.put(error), loop).result()
            else:
                asyncio.run_coroutine_threadsafe(queue.put(finished), loop).result()

        worker = asyncio.create_task(asyncio.to_thread(produce))
        try:
            while True:
                item = await queue.get()
                if item is finished:
                    break
                if isinstance(item, Exception):
                    raise SpeechSynthesisError(f"Piper synthesis failed: {item}") from item
                yield item
        finally:
            await worker

    def _synthesisConfig(self, speakerId: int | None) -> Any:
        """Speaker and speaking rate, or None when both are default."""
        if speakerId is None and self._lengthScale is None:
            return None

        piper = _importPiper()
        settings: dict[str, Any] = {}
        if speakerId is not None:
            settings["speaker_id"] = speakerId
        if self._lengthScale is not None:
            # Piper's length_scale stretches time, so larger is slower. Naming
            # the setting speechRate would invert it for anyone reading the
            # config, so it is passed through under Piper's own meaning.
            settings["length_scale"] = self._lengthScale
        return piper.SynthesisConfig(**settings)

    # --- Voices -----------------------------------------------------------

    def availableVoices(self) -> list[str]:
        """Voices already downloaded. Many more can be fetched by name."""
        if not self._modelsDirectory.is_dir():
            return []
        return sorted(path.stem for path in self._modelsDirectory.glob("*.onnx"))

    def describe(self) -> str:
        return f"piper ({self._voiceName}) at {self._sampleRate} Hz on cpu"


def _splitSpeaker(name: str) -> tuple[str, int | None]:
    """Separate "voice#3" into the voice and its speaker index."""
    modelName, separator, speaker = name.partition(SPEAKER_SEPARATOR)
    if not separator:
        return modelName, None

    try:
        return modelName, int(speaker)
    except ValueError:
        logger.warning("Ignoring unreadable speaker id in %r", name)
        return modelName, None


def _importPiper() -> Any:
    try:
        import piper
    except ImportError as error:
        raise ProviderUnavailableError(MISSING_PACKAGE) from error
    return piper
