"""Audio output devices.

The assistant must run on a workstation with speakers, on a headless server,
and in tests, so playback sits behind an interface with backends for each.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from app.audio.audioBuffer import AudioBuffer

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


class AudioOutputError(Exception):
    """Raised when audio cannot be played."""


class AudioOutput(ABC):
    """Somewhere generated audio can be sent."""

    @abstractmethod
    async def play(self, audio: AudioBuffer) -> None:
        """Play or persist ``audio``, returning once it has been handled."""

    async def beginUtterance(self) -> None:  # noqa: B027 - optional hook
        """Mark the start of one reply.

        Synthesis streams, so a reply reaches ``play`` as several pieces. An
        output that needs to treat them as one thing --- a file, a recording,
        a client expecting a single response --- needs to know where the
        boundaries are, and only the runtime does.
        """

    async def endUtterance(self) -> None:  # noqa: B027 - optional hook
        """Mark the end of one reply."""

    async def stop(self) -> None:  # noqa: B027 - optional hook, not every output can stop
        """Interrupt playback in progress. Silent no-op if there is none."""

    def close(self) -> None:  # noqa: B027 - optional hook, not every output holds a device
        """Release any device held by this output."""

    def describe(self) -> str:
        return type(self).__name__


class MemoryAudioOutput(AudioOutput):
    """Keeps buffers in memory. Intended for tests."""

    def __init__(self) -> None:
        self.played: list[AudioBuffer] = []

    async def play(self, audio: AudioBuffer) -> None:
        self.played.append(audio)

    @property
    def lastPlayed(self) -> AudioBuffer | None:
        return self.played[-1] if self.played else None


class WavFileOutput(AudioOutput):
    """Writes each utterance to a WAV file.

    Used on headless machines, and whenever generated audio needs inspecting.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._counter = 0
        self._pending: list[AudioBuffer] | None = None
        self.lastPath: Path | None = None

    async def beginUtterance(self) -> None:
        """Collect the pieces of one reply so they become one file."""
        self._pending = []

    async def endUtterance(self) -> None:
        pending, self._pending = self._pending, None
        if pending:
            await self._write(_join(pending))

    async def play(self, audio: AudioBuffer) -> None:
        if self._pending is not None:
            self._pending.append(audio)
            return
        await self._write(audio)

    async def _write(self, audio: AudioBuffer) -> None:
        self._counter += 1
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        path = self._directory / f"{stamp}-{self._counter:03d}.wav"
        await asyncio.to_thread(audio.writeWav, path)
        self.lastPath = path
        logger.info("Wrote %.2fs of audio to %s", audio.durationSeconds, path)

    def describe(self) -> str:
        return f"WAV files in {self._directory}"


class SoundDeviceOutput(AudioOutput):
    """Plays through the system audio device via PortAudio.

    Audio is written into one continuous stream rather than played buffer by
    buffer. Since synthesis streams, a reply arrives as a dozen pieces, and
    ``sounddevice.play`` opens and closes a device for each one. Measured, that
    cost about 0.4 s per piece: nine pieces of a 3.6 s reply took 7.1 s to play,
    with a silence between every one. It was audible as speech breaking up
    mid-word. Writing into a stream that stays open takes 3.75 s for the same
    3.6 s of audio.
    """

    # One writer at a time, and no closing the stream while somebody is
    # writing to it. Two sources at different sample rates --- an audiobook at
    # 44.1 kHz and a spoken reply at 24 --- would otherwise each close the
    # other's stream to open its own, mid-write.
    #
    # Aborting has to take this too, which is not obvious and cost a crash to
    # learn: stopping a book cancels a play() that is already inside a blocking
    # write on a worker thread, then calls abort() and close() from another.
    # PortAudio closing a stream underneath a write in flight takes the process
    # down natively, with no Python traceback to explain it.
    #
    # Reentrant because _writeBlocking holds it while _ensureStream may close
    # and reopen the stream, and a plain Lock would deadlock on itself.
    #
    # Held on the class rather than the instance because it guards one sound
    # card, not one object, and because an instance built without __init__
    # still needs it.
    _lock: ClassVar[threading.RLock] = threading.RLock()

    def __init__(self, device: str | int | None = None) -> None:
        self._device = device
        self._soundDevice = _importSoundDevice()
        self._numpy = _importNumpy()
        self._stream: Any = None
        self._format: tuple[int, int] | None = None

    async def play(self, audio: AudioBuffer) -> None:
        if audio.sampleWidth != 2:
            raise AudioOutputError(
                f"SoundDeviceOutput supports 16-bit audio only, got {audio.sampleWidth * 8}-bit"
            )
        if not audio.data:
            return

        samples = self._numpy.frombuffer(audio.data, dtype="<i2")
        if audio.channels > 1:
            samples = samples.reshape(-1, audio.channels)

        try:
            await asyncio.to_thread(
                self._writeBlocking, samples, audio.sampleRate, audio.channels
            )
        except AudioOutputError:
            raise
        except Exception as error:  # backend errors vary by platform
            raise AudioOutputError(f"Playback failed: {error}") from error

    def _writeBlocking(self, samples: Any, sampleRate: int, channels: int) -> None:
        """Write into the open stream, blocking until the device accepts it.

        Blocking is what keeps consecutive pieces continuous, and it doubles as
        backpressure so synthesis cannot run away from playback.
        """
        with self._lock:
            self._ensureStream(sampleRate, channels).write(samples)

    def _ensureStream(self, sampleRate: int, channels: int) -> Any:
        if self._format is not None and self._format != (sampleRate, channels):
            self._closeStream()

        if self._stream is None:
            try:
                stream = self._soundDevice.OutputStream(
                    samplerate=sampleRate,
                    channels=channels,
                    dtype="int16",
                    device=self._device,
                )
                stream.start()
            except Exception as error:
                raise AudioOutputError(
                    f"Could not open the speaker: {error}\n"
                    "Run with --list-devices and set audio.outputDevice."
                ) from error
            self._stream = stream
            self._format = (sampleRate, channels)

        return self._stream

    def _closeStream(self) -> None:
        with self._lock:
            stream, self._stream, self._format = self._stream, None, None
            if stream is None:
                return
            try:
                stream.stop()
                stream.close()
            except Exception as error:  # closing must not raise  # noqa: BLE001
                logger.debug("Ignoring error while closing the audio stream: %s", error)

    async def stop(self) -> None:
        """Interrupt playback, discarding anything still buffered.

        Barge-in has to drop buffered audio rather than let it drain: the point
        of interrupting is to stop talking now.
        """
        await asyncio.to_thread(self._abort)

    def _abort(self) -> None:
        # Waits for any write in flight to return before touching the stream.
        # Without this the close below lands underneath it and the process dies.
        with self._lock:
            if self._stream is None:
                return
            try:
                self._stream.abort()
            except Exception as error:  # aborting must not raise  # noqa: BLE001
                logger.debug("Ignoring error while stopping playback: %s", error)
            self._closeStream()

    def close(self) -> None:
        self._closeStream()

    def describe(self) -> str:
        return f"system audio device ({self._device if self._device is not None else 'default'})"


def _join(pieces: list[AudioBuffer]) -> AudioBuffer:
    """Concatenate pieces of one reply into a single buffer."""
    first = pieces[0]
    return AudioBuffer(
        data=b"".join(piece.data for piece in pieces),
        sampleRate=first.sampleRate,
        channels=first.channels,
        sampleWidth=first.sampleWidth,
    )


def createAudioOutput(settings: Settings) -> AudioOutput:
    """Build the output backend named in configuration."""
    if settings.audio.outputBackend == "file":
        return WavFileOutput(settings.paths.outputs)
    return SoundDeviceOutput(settings.audio.outputDevice)


def listOutputDevices() -> list[str]:
    """Human-readable list of available output devices."""
    soundDevice = _importSoundDevice()
    descriptions = []
    for index, device in enumerate(soundDevice.query_devices()):
        if device.get("max_output_channels", 0) > 0:
            descriptions.append(f"{index}: {device['name']}")
    return descriptions


def _importSoundDevice():  # type: ignore[no-untyped-def]
    try:
        import sounddevice
    except (ImportError, OSError) as error:
        # sounddevice raises OSError when the PortAudio library is absent.
        raise AudioOutputError(
            "Audio playback requires the 'sounddevice' package and PortAudio.\n"
            'Install with: pip install -e ".[tts]"\n'
            'Or set audio.outputBackend to "file" to write WAV files instead.'
        ) from error
    return sounddevice


def _importNumpy():  # type: ignore[no-untyped-def]
    try:
        import numpy
    except ImportError as error:
        raise AudioOutputError(
            'Audio playback requires numpy. Install with: pip install -e ".[tts]"'
        ) from error
    return numpy
