"""Audio capture devices.

Capture sits behind an interface for the same reason playback does: the
assistant must run on a workstation with a microphone, on a headless server
fed by a remote client, and in tests with no hardware at all.

Recording is interruptible rather than fixed-length, because push-to-talk in
Stage 2 and voice activity detection in Stage 6 both need to stop on an
external signal.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

from app.audio.audioBuffer import AudioBuffer

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16000
BLOCK_FRAMES = 1024
# Small enough for voice activity detection to react promptly, large enough
# that the callback overhead stays negligible.
DEFAULT_FRAME_MILLISECONDS = 32
# About four seconds of audio. Beyond this the consumer is not keeping up, and
# responding to stale speech is worse than dropping it.
MAXIMUM_QUEUED_FRAMES = 128

# How long the microphone may deliver nothing at all before it is treated as
# gone. Silence still arrives as frames of zeroes, so this does not fire on a
# quiet room --- only when the callback itself stops, which is what happens
# when a device leaves the bus.
DEFAULT_STALL_SECONDS = 5.0


class AudioInputError(Exception):
    """Raised when audio cannot be captured."""


class AudioInput(ABC):
    """Somewhere audio can be captured from."""

    @property
    @abstractmethod
    def sampleRate(self) -> int:
        """Sample rate of returned audio."""

    @property
    def channels(self) -> int:
        return 1

    @abstractmethod
    def frames(
        self, frameMilliseconds: int = DEFAULT_FRAME_MILLISECONDS
    ) -> AsyncIterator[AudioBuffer]:
        """Yield fixed-size frames continuously.

        The primitive that continuous listening is built on: voice activity
        detection and wake-word models both consume small fixed frames, and
        neither can work with a single blocking recording.
        """

    async def record(
        self,
        *,
        stopWhen: asyncio.Event | None = None,
        maximumSeconds: float = 30.0,
    ) -> AudioBuffer:
        """Capture until ``stopWhen`` is set or ``maximumSeconds`` elapses."""
        pieces: list[bytes] = []
        captured = 0.0

        async for frame in self.frames():
            if stopWhen is not None and stopWhen.is_set():
                break
            pieces.append(frame.data)
            captured += frame.durationSeconds
            if captured >= maximumSeconds:
                break

        return AudioBuffer(
            data=b"".join(pieces), sampleRate=self.sampleRate, channels=self.channels
        )

    def drain(self) -> int:
        """Discard audio captured but not yet consumed, returning how much went.

        Capture never stops, so anything recorded while the assistant was
        speaking is still queued when it finishes. Feeding that to the
        segmenter is how an assistant ends up answering itself.
        """
        return 0

    def close(self) -> None:  # noqa: B027 - optional hook, not every input holds a device
        """Release any device held by this input."""

    def describe(self) -> str:
        return type(self).__name__


class MemoryAudioInput(AudioInput):
    """Returns buffers handed to it. Intended for tests."""

    def __init__(self, buffers: list[AudioBuffer] | None = None) -> None:
        self._buffers = list(buffers or [])
        self.recordCount = 0

    @property
    def sampleRate(self) -> int:
        return self._buffers[0].sampleRate if self._buffers else DEFAULT_SAMPLE_RATE

    async def record(
        self,
        *,
        stopWhen: asyncio.Event | None = None,
        maximumSeconds: float = 30.0,
    ) -> AudioBuffer:
        self.recordCount += 1
        if not self._buffers:
            return AudioBuffer.silence(0.5, DEFAULT_SAMPLE_RATE)
        return self._buffers.pop(0)

    async def frames(
        self, frameMilliseconds: int = DEFAULT_FRAME_MILLISECONDS
    ) -> AsyncIterator[AudioBuffer]:
        """Split the queued buffers into frames, then stop.

        Ending rather than blocking forever is what lets a test drive a
        listener to completion without arranging to cancel it.
        """
        for buffer in list(self._buffers):
            for frame in splitIntoFrames(buffer, frameMilliseconds):
                yield frame
        self._buffers.clear()


class WavFileInput(AudioInput):
    """Reads audio from a WAV file, resampling if necessary.

    Used by the integration tests, and by anything replaying captured audio.
    """

    def __init__(self, path: Path, targetSampleRate: int = DEFAULT_SAMPLE_RATE) -> None:
        self._path = path
        self._targetSampleRate = targetSampleRate

    @property
    def sampleRate(self) -> int:
        return self._targetSampleRate

    async def record(
        self,
        *,
        stopWhen: asyncio.Event | None = None,
        maximumSeconds: float = 30.0,
    ) -> AudioBuffer:
        if not self._path.is_file():
            raise AudioInputError(f"Audio file not found: {self._path}")

        buffer = await asyncio.to_thread(AudioBuffer.fromWavFile, self._path)
        return resampleBuffer(buffer, self._targetSampleRate)

    async def frames(
        self, frameMilliseconds: int = DEFAULT_FRAME_MILLISECONDS
    ) -> AsyncIterator[AudioBuffer]:
        """Replay the file as frames, then stop."""
        buffer = await self.record()
        for frame in splitIntoFrames(buffer, frameMilliseconds):
            yield frame

    def describe(self) -> str:
        return f"WAV file {self._path}"


class SoundDeviceInput(AudioInput):
    """Captures from the system microphone via PortAudio."""

    def __init__(
        self,
        device: str | int | None = None,
        sampleRate: int = DEFAULT_SAMPLE_RATE,
        channels: int = 1,
        stallSeconds: float = DEFAULT_STALL_SECONDS,
    ) -> None:
        self._device = device
        self._targetSampleRate = sampleRate
        self._channels = channels
        self._stallSeconds = stallSeconds
        self._soundDevice = _importSoundDevice()
        self._numpy = _importNumpy()
        # Resolved on first use: not every device offers 16 kHz natively.
        self._captureSampleRate: int | None = None
        # Held while frames() is running, so a caller can throw away audio
        # captured during a stretch it was not listening to.
        self._queue: asyncio.Queue[bytes] | None = None

    @property
    def sampleRate(self) -> int:
        return self._targetSampleRate

    @property
    def channels(self) -> int:
        return self._channels

    async def frames(
        self, frameMilliseconds: int = DEFAULT_FRAME_MILLISECONDS
    ) -> AsyncIterator[AudioBuffer]:
        """Capture continuously, yielding frames as they arrive.

        PortAudio delivers audio on its own thread, so frames are handed to the
        event loop through a queue rather than by polling. The queue is bounded:
        if the consumer stalls, dropping old audio is better than growing a
        backlog that makes the assistant respond to something said a minute ago.
        """
        rate = self._resolveCaptureRate()
        blockFrames = max(1, int(rate * frameMilliseconds / 1000))
        needsResampling = rate != self._targetSampleRate

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=MAXIMUM_QUEUED_FRAMES)
        self._queue = queue

        def onAudio(indata, frameCount, timeInfo, status) -> None:
            if status:
                logger.debug("Input status: %s", status)
            loop.call_soon_threadsafe(_offer, queue, bytes(indata))

        try:
            stream = self._soundDevice.InputStream(
                samplerate=rate,
                channels=self._channels,
                dtype="int16",
                device=self._device,
                blocksize=blockFrames,
                callback=onAudio,
            )
        except Exception as error:
            raise AudioInputError(f"Could not open the microphone: {error}") from error

        stream.start()
        try:
            while True:
                data = await self._nextFrame(queue)
                frame = AudioBuffer(data=data, sampleRate=rate, channels=self._channels)
                yield resampleBuffer(frame, self._targetSampleRate) if needsResampling else frame
        finally:
            self._queue = None
            stream.stop()
            stream.close()

    async def _nextFrame(self, queue: asyncio.Queue[bytes]) -> bytes:
        """Wait for the next frame, giving up if the device stops delivering.

        Without this a microphone that leaves the bus --- unplugged, or
        switched off at the connector rather than muted --- leaves the caller
        awaiting a queue nothing will ever fill again. The process stays alive
        and connected but deaf, which is the worst of both: no audio, and no
        crash for a supervisor to restart.

        A muted microphone is a different thing and is not affected. Muting
        silences the signal while the device keeps streaming, so frames of
        zeroes continue to arrive and this never fires.
        """
        if self._stallSeconds <= 0:
            return await queue.get()

        try:
            return await asyncio.wait_for(queue.get(), timeout=self._stallSeconds)
        except TimeoutError as error:
            raise AudioInputError(
                f"No audio from the microphone for {self._stallSeconds:.1f}s.\n"
                "The device has stopped delivering, which usually means it was "
                "unplugged or switched off at the connector.\n"
                "A mute switch that silences the signal keeps the device "
                "streaming and will not cause this; one that cuts the "
                "connection does.\n"
                "Set audio.captureStallSeconds to 0 to wait indefinitely instead."
            ) from error

    def drain(self) -> int:
        """Throw away whatever the microphone has queued but nobody has read."""
        queue = self._queue
        if queue is None:
            return 0

        dropped = 0
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return dropped
            dropped += 1

    def _resolveCaptureRate(self) -> int:
        """Use the requested rate if the device supports it, else its default.

        Anything captured at a different rate is resampled afterwards, so this
        only decides where the conversion happens.
        """
        if self._captureSampleRate is not None:
            return self._captureSampleRate

        try:
            self._soundDevice.check_input_settings(
                device=self._device,
                samplerate=self._targetSampleRate,
                channels=self._channels,
                dtype="int16",
            )
            self._captureSampleRate = self._targetSampleRate
        except Exception as error:
            fallback = self._deviceDefaultRate()
            if fallback is None:
                raise AudioInputError(
                    f"Microphone does not support {self._targetSampleRate} Hz "
                    f"and its default rate could not be determined: {error}\n"
                    "Run with --list-input-devices and set audio.inputDevice."
                ) from error
            logger.info(
                "Microphone does not offer %d Hz; capturing at %d Hz and resampling",
                self._targetSampleRate,
                fallback,
            )
            self._captureSampleRate = fallback

        return self._captureSampleRate

    def _deviceDefaultRate(self) -> int | None:
        try:
            info = self._soundDevice.query_devices(self._device, "input")
            return int(info["default_samplerate"])
        except Exception as error:  # noqa: BLE001
            logger.debug("Could not query input device: %s", error)
            return None

    def describe(self) -> str:
        device = self._device if self._device is not None else "default"
        return f"microphone ({device}) at {self._targetSampleRate} Hz"


def splitIntoFrames(buffer: AudioBuffer, frameMilliseconds: int) -> list[AudioBuffer]:
    """Cut a buffer into fixed-size frames, discarding any short remainder.

    Voice activity and wake-word models are fed uniform frames, so a partial
    frame at the end is dropped rather than passed on undersized.
    """
    frameCount = max(1, int(buffer.sampleRate * frameMilliseconds / 1000))
    frameBytes = frameCount * buffer.channels * buffer.sampleWidth

    return [
        AudioBuffer(
            data=buffer.data[start : start + frameBytes],
            sampleRate=buffer.sampleRate,
            channels=buffer.channels,
        )
        for start in range(0, len(buffer.data) - frameBytes + 1, frameBytes)
    ]


def _offer(queue: asyncio.Queue[bytes], data: bytes) -> None:
    """Add a frame, discarding the oldest if the consumer has fallen behind."""
    if queue.full():
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()
    queue.put_nowait(data)


def resampleBuffer(buffer: AudioBuffer, targetSampleRate: int) -> AudioBuffer:
    """Resample to ``targetSampleRate``, and downmix to mono.

    Whisper expects 16 kHz mono, and capture devices rarely offer it directly.
    """
    if buffer.channels > 1:
        buffer = _downmixToMono(buffer)

    if buffer.sampleRate == targetSampleRate or not buffer.data:
        return buffer

    numpy = _importNumpy()
    samples = numpy.frombuffer(buffer.data, dtype="<i2")

    try:
        import soxr

        resampled = soxr.resample(samples, buffer.sampleRate, targetSampleRate)
    except ImportError as error:
        raise AudioInputError(
            f"Audio captured at {buffer.sampleRate} Hz must be resampled to "
            f"{targetSampleRate} Hz, which needs the 'soxr' package.\n"
            'Install with: pip install -e ".[stt]"'
        ) from error

    return AudioBuffer(
        data=numpy.asarray(resampled).astype("<i2").tobytes(),
        sampleRate=targetSampleRate,
        channels=1,
    )


def _downmixToMono(buffer: AudioBuffer) -> AudioBuffer:
    numpy = _importNumpy()
    samples = numpy.frombuffer(buffer.data, dtype="<i2").reshape(-1, buffer.channels)
    mono = samples.mean(axis=1).round().astype("<i2")
    return AudioBuffer(data=mono.tobytes(), sampleRate=buffer.sampleRate, channels=1)


def createAudioInput(settings: Settings) -> AudioInput:
    """Build the capture backend named in configuration."""
    return SoundDeviceInput(
        device=settings.audio.inputDevice,
        sampleRate=settings.audio.sampleRate,
        stallSeconds=settings.audio.captureStallSeconds,
    )


def listInputDevices() -> list[str]:
    """Human-readable list of available input devices."""
    soundDevice = _importSoundDevice()
    descriptions = []
    for index, device in enumerate(soundDevice.query_devices()):
        if device.get("max_input_channels", 0) > 0:
            descriptions.append(f"{index}: {device['name']}")
    return descriptions


def _importSoundDevice():  # type: ignore[no-untyped-def]
    try:
        import sounddevice
    except (ImportError, OSError) as error:
        raise AudioInputError(
            "Audio capture requires the 'sounddevice' package and PortAudio.\n"
            'Install with: pip install -e ".[stt]"'
        ) from error
    return sounddevice


def _importNumpy():  # type: ignore[no-untyped-def]
    try:
        import numpy
    except ImportError as error:
        raise AudioInputError(
            'Audio capture requires numpy. Install with: pip install -e ".[stt]"'
        ) from error
    return numpy
