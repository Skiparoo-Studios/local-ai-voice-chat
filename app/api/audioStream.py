"""The audio streaming protocol.

Audio crosses the WebSocket as **binary frames of raw 16-bit PCM**, bracketed
by JSON control messages on the text channel. Splitting the two means audio
avoids the third of its size that base64 would add, and the control channel
stays readable.

    client -> server   {"type": "audio.start", "sampleRate": 16000}
                       <binary frames>
                       {"type": "audio.end"}

    server -> client   {"type": "audio.reply", "sampleRate": 24000, ...}
                       <binary frames>
                       {"type": "audio.replyEnd"}

**Why raw PCM rather than Opus.** 16 kHz mono PCM is 32 KB/s, and because the
client runs its own wake word and voice activity detection, it only transmits
during an utterance --- a five second request is 160 KB. On a home network that
is not worth an encoder, a decoder, a dependency and the latency of both. Opus
becomes the right answer when this crosses the internet, which is the reason
the format is announced in the control message rather than assumed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.audio.audioBuffer import AudioBuffer

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16000

# A single utterance should never approach this. It exists so a client that
# never sends audio.end cannot exhaust the server's memory.
MAXIMUM_UTTERANCE_BYTES = 16 * 1024 * 1024

# Audio is sent back in pieces so a client can begin playing before the whole
# reply has arrived. 32 KB is about a second at 24 kHz.
REPLY_CHUNK_BYTES = 32 * 1024


class AudioStreamError(Exception):
    """Raised when a client's audio stream is malformed."""


@dataclass(slots=True)
class IncomingAudio:
    """Audio being received from one client."""

    sampleRate: int = DEFAULT_SAMPLE_RATE
    channels: int = 1
    chunks: list[bytes] = field(default_factory=list)
    byteCount: int = 0
    open: bool = False

    def begin(self, message: dict[str, Any]) -> None:
        """Start a new utterance, discarding anything left from the last."""
        self.sampleRate = _positiveInt(message.get("sampleRate"), DEFAULT_SAMPLE_RATE)
        self.channels = _positiveInt(message.get("channels"), 1)
        self.chunks = []
        self.byteCount = 0
        self.open = True

    def add(self, payload: bytes) -> None:
        """Add a binary frame."""
        if not self.open:
            raise AudioStreamError(
                "Audio arrived before audio.start. Send audio.start first."
            )

        self.byteCount += len(payload)
        if self.byteCount > MAXIMUM_UTTERANCE_BYTES:
            self.reset()
            raise AudioStreamError(
                f"An utterance may not exceed "
                f"{MAXIMUM_UTTERANCE_BYTES // (1024 * 1024)} MB."
            )

        self.chunks.append(payload)

    def finish(self) -> AudioBuffer:
        """Close the utterance and return what arrived."""
        if not self.open:
            raise AudioStreamError("audio.end arrived without a matching audio.start.")

        payload = b"".join(self.chunks)
        sampleRate, channels = self.sampleRate, self.channels
        self.reset()

        frameSize = channels * 2
        if len(payload) % frameSize:
            # A truncated final frame would misalign every sample after it.
            payload = payload[: len(payload) - (len(payload) % frameSize)]
            logger.debug("Trimmed a partial frame from the end of an utterance")

        return AudioBuffer(data=payload, sampleRate=sampleRate, channels=channels)

    def reset(self) -> None:
        self.chunks = []
        self.byteCount = 0
        self.open = False


def replyHeader(
    audio: AudioBuffer, *, text: str = "", serverSeconds: float = 0.0
) -> dict[str, Any]:
    """The control message that precedes reply audio.

    ``serverSeconds`` is how long the assistant itself took. A client can
    subtract it from its own round trip to see what the network cost, which is
    the only way to tell a slow model from a slow link.
    """
    return {
        "type": "audio.reply",
        "format": "pcm16",
        "sampleRate": audio.sampleRate,
        "channels": audio.channels,
        "durationSeconds": round(audio.durationSeconds, 3),
        "byteCount": len(audio.data),
        "serverSeconds": round(serverSeconds, 3),
        "text": text,
    }


def streamedReplyHeader(first: AudioBuffer) -> dict[str, Any]:
    """The control message that precedes streamed reply audio.

    Length is absent because it is not known: the audio is still being
    generated. Clients play until ``audio.replyEnd``, which carries the text
    and the timings once they exist.
    """
    return {
        "type": "audio.reply",
        "format": "pcm16",
        "sampleRate": first.sampleRate,
        "channels": first.channels,
        "streaming": True,
    }


def chunkAudio(audio: AudioBuffer, chunkBytes: int = REPLY_CHUNK_BYTES) -> list[bytes]:
    """Split audio into pieces small enough to start playing early."""
    frameSize = audio.channels * audio.sampleWidth
    aligned = max(frameSize, (chunkBytes // frameSize) * frameSize)
    return [audio.data[start : start + aligned] for start in range(0, len(audio.data), aligned)]


def _positiveInt(value: Any, fallback: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return fallback
    return number if number > 0 else fallback


__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "AudioStreamError",
    "IncomingAudio",
    "chunkAudio",
    "replyHeader",
    "streamedReplyHeader",
]
