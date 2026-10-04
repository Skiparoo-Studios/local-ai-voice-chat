"""Measure what a remote client costs.

Sends the same utterance to a running assistant over the network, and compares
the round trip with the server's own report of how long the turn took. The
difference is what the network added: transfer, framing and scheduling.

Usage:
    python scripts/benchmarkRemote.py --server http://192.168.1.10:8000 --token ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from pathlib import Path

import websockets

from app.audio.audioBuffer import AudioBuffer
from app.audio.input import resampleBuffer
from app.client.remoteClient import _websocketUrl

FIXTURE = Path("tests/data/utterance.wav")


def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure remote client latency")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default="")
    parser.add_argument("--audio", type=Path, default=FIXTURE)
    parser.add_argument("--runs", type=int, default=3)
    return parser.parse_args()


async def oneTurn(socket, utterance: AudioBuffer) -> dict[str, float | str]:
    """Send an utterance, wait for the reply audio, and time the whole thing."""
    started = time.perf_counter()

    await socket.send(
        json.dumps(
            {
                "type": "audio.start",
                "sampleRate": utterance.sampleRate,
                "channels": utterance.channels,
                "includeAudio": True,
            }
        )
    )
    await socket.send(utterance.data)
    await socket.send(json.dumps({"type": "audio.end"}))
    uploadedAt = time.perf_counter()

    transcript = ""
    reply = ""
    firstTextAt = 0.0
    firstAudioAt = 0.0
    received = 0
    serverSeconds = 0.0

    while True:
        message = await asyncio.wait_for(socket.recv(), timeout=180)

        if isinstance(message, bytes):
            received += len(message)
            # The first byte of audio: what the listener actually waits for.
            if not firstAudioAt:
                firstAudioAt = time.perf_counter()
            continue

        payload = json.loads(message)
        kind = payload.get("type")

        if kind == "assistant.transcription":
            transcript = payload.get("text", "")
        elif kind == "assistant.response":
            reply = payload.get("text", "")
            if not firstTextAt:
                firstTextAt = time.perf_counter()
        elif kind == "audio.replyEnd":
            # Timings arrive at the end, because when audio is streamed the
            # server does not know them when the first piece leaves.
            serverSeconds = float(payload.get("serverSeconds") or 0.0)
            reply = payload.get("text") or reply
            break

    finished = time.perf_counter()
    total = finished - started
    return {
        "upload": uploadedAt - started,
        "firstText": (firstTextAt or finished) - started,
        "firstAudio": (firstAudioAt or finished) - started,
        "total": total,
        "serverSeconds": serverSeconds,
        # What the network and framing cost, as distinct from the assistant.
        "overhead": max(0.0, total - serverSeconds),
        "bytesDown": received,
        "transcript": transcript,
        "reply": reply,
    }


async def main() -> int:
    arguments = parseArguments()
    logging.basicConfig(level="WARNING")

    if not arguments.audio.is_file():
        print(f"audio fixture not found: {arguments.audio}")
        return 2

    utterance = resampleBuffer(AudioBuffer.fromWavFile(arguments.audio), 16000)
    url = _websocketUrl(arguments.server, arguments.token or None)

    print(f"server={arguments.server} runs={arguments.runs}")
    print(
        f"  {'utterance':<22} {utterance.durationSeconds:.2f}s, "
        f"{len(utterance.data) / 1024:.0f} KB as pcm16"
    )

    results = []
    async with websockets.connect(url, max_size=None) as socket:
        await socket.recv()
        # The first turn pays for whatever the server has not warmed.
        results.append(await oneTurn(socket, utterance))
        for _ in range(arguments.runs):
            results.append(await oneTurn(socket, utterance))

    first, warm = results[0], results[1:]

    print(f"  {'first turn':<22} {first['total']:.2f}s")
    print(
        f"  {'upload':<22} {statistics.mean(r['upload'] for r in warm):.3f}s "
        f"({len(utterance.data) / 1024:.0f} KB)"
    )
    print(
        f"  {'to reply text':<22} "
        f"{statistics.mean(r['firstText'] for r in warm):.2f}s"
    )
    print(
        f"  {'to first audio':<22} "
        f"{statistics.mean(r['firstAudio'] for r in warm):.2f}s mean, "
        f"{min(r['firstAudio'] for r in warm):.2f}s best"
    )
    print(
        f"  {'full round trip':<22} "
        f"{statistics.mean(r['total'] for r in warm):.2f}s mean, "
        f"{min(r['total'] for r in warm):.2f}s best"
    )
    print(
        f"  {'of which the server':<22} "
        f"{statistics.mean(r['serverSeconds'] for r in warm):.2f}s"
    )
    print(
        f"  {'added by the network':<22} "
        f"{statistics.mean(r['overhead'] for r in warm):.3f}s"
    )
    print(f"  {'reply audio':<22} {warm[-1]['bytesDown'] / 1024:.0f} KB")
    print(f"  {'transcript':<22} {warm[-1]['transcript']!r}")
    print(f"  {'reply':<22} {str(warm[-1]['reply'])[:60]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
