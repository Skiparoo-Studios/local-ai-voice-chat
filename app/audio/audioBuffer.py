"""Audio representation shared between providers and output devices.

Raw ``bytes`` alone cannot be played back --- sample rate and channel layout
matter --- so audio crosses interfaces as an :class:`AudioBuffer` carrying its
own format. Samples are interleaved little-endian PCM, which the standard
library can write as WAV without numpy, keeping the base package dependency
free.
"""

from __future__ import annotations

import io
import math
import sys
import wave
from array import array
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AudioBuffer:
    """A block of PCM audio."""

    data: bytes
    sampleRate: int
    channels: int = 1
    sampleWidth: int = 2

    def __post_init__(self) -> None:
        if self.sampleRate <= 0:
            raise ValueError(f"sampleRate must be positive, got {self.sampleRate}")
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")
        if self.sampleWidth not in (1, 2, 3, 4):
            raise ValueError(f"sampleWidth must be 1, 2, 3 or 4 bytes, got {self.sampleWidth}")

        frameSize = self.channels * self.sampleWidth
        if len(self.data) % frameSize:
            raise ValueError(
                f"data length {len(self.data)} is not a whole number of "
                f"{frameSize}-byte frames"
            )

    @property
    def frameCount(self) -> int:
        return len(self.data) // (self.channels * self.sampleWidth)

    @property
    def durationSeconds(self) -> float:
        return self.frameCount / self.sampleRate

    def toWavBytes(self) -> bytes:
        """Encode as a WAV file in memory."""
        stream = io.BytesIO()
        with wave.open(stream, "wb") as handle:
            handle.setnchannels(self.channels)
            handle.setsampwidth(self.sampleWidth)
            handle.setframerate(self.sampleRate)
            handle.writeframes(self.data)
        return stream.getvalue()

    def writeWav(self, path: Path) -> Path:
        """Write as a WAV file, creating parent directories as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.toWavBytes())
        return path

    @classmethod
    def fromWavBytes(cls, payload: bytes) -> AudioBuffer:
        with wave.open(io.BytesIO(payload), "rb") as handle:
            return cls(
                data=handle.readframes(handle.getnframes()),
                sampleRate=handle.getframerate(),
                channels=handle.getnchannels(),
                sampleWidth=handle.getsampwidth(),
            )

    @classmethod
    def fromWavFile(cls, path: Path) -> AudioBuffer:
        return cls.fromWavBytes(path.read_bytes())

    @classmethod
    def fromFloatSamples(
        cls,
        samples: Iterable[float],
        sampleRate: int,
        channels: int = 1,
    ) -> AudioBuffer:
        """Build a 16-bit buffer from floats in [-1.0, 1.0].

        Values outside the range are clipped rather than allowed to wrap, which
        would turn a loud sample into noise.
        """
        encoded = array("h")
        for sample in samples:
            clipped = -1.0 if sample < -1.0 else (1.0 if sample > 1.0 else sample)
            encoded.append(round(clipped * 32767.0))

        if sys.byteorder == "big":
            encoded.byteswap()

        return cls(data=encoded.tobytes(), sampleRate=sampleRate, channels=channels)

    def toFloatSamples(self) -> list[float]:
        """Decode to floats in [-1.0, 1.0]. 16-bit only."""
        if self.sampleWidth != 2:
            raise ValueError(f"toFloatSamples supports 16-bit audio only, got {self.sampleWidth}")

        decoded = array("h")
        decoded.frombytes(self.data)
        if sys.byteorder == "big":
            decoded.byteswap()
        return [sample / 32767.0 for sample in decoded]

    @classmethod
    def silence(cls, durationSeconds: float, sampleRate: int, channels: int = 1) -> AudioBuffer:
        frames = max(0, int(durationSeconds * sampleRate))
        return cls(
            data=bytes(frames * channels * 2),
            sampleRate=sampleRate,
            channels=channels,
        )

    @classmethod
    def tone(
        cls,
        frequency: float,
        durationSeconds: float,
        sampleRate: int,
        amplitude: float = 0.3,
        fadeSeconds: float = 0.01,
    ) -> AudioBuffer:
        """A sine tone with short fades, which prevent clicks at the edges."""
        frames = max(0, int(durationSeconds * sampleRate))
        fadeFrames = max(1, int(fadeSeconds * sampleRate))
        step = 2.0 * math.pi * frequency / sampleRate

        def generate() -> Iterable[float]:
            for index in range(frames):
                envelope = min(1.0, index / fadeFrames, (frames - index) / fadeFrames)
                yield amplitude * envelope * math.sin(step * index)

        return cls.fromFloatSamples(generate(), sampleRate)

    def concatenate(self, other: AudioBuffer) -> AudioBuffer:
        """Join two buffers of identical format."""
        if (self.sampleRate, self.channels, self.sampleWidth) != (
            other.sampleRate,
            other.channels,
            other.sampleWidth,
        ):
            raise ValueError("cannot concatenate buffers with different formats")
        return AudioBuffer(
            data=self.data + other.data,
            sampleRate=self.sampleRate,
            channels=self.channels,
            sampleWidth=self.sampleWidth,
        )
