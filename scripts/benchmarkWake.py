"""Measure the cost of listening, and the accuracy of waking.

Two things matter for a device that sits in a room all day. It must be cheap
enough to leave running, and it must wake when addressed without waking at
other times.

Usage:
    python scripts/benchmarkWake.py --idle-seconds 20
    python scripts/benchmarkWake.py --audio recording.wav --expect 3
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import statistics
import time
from pathlib import Path

from app.audio.audioBuffer import AudioBuffer
from app.audio.input import splitIntoFrames
from app.audio.vad import createVadProvider, rootMeanSquare
from app.audio.wakeWord import createWakeWordProvider
from app.config import Settings


def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark wake word and VAD cost")
    parser.add_argument("--word", default="hey_jarvis")
    parser.add_argument("--vad", default="silero", choices=["silero", "energy"])
    parser.add_argument(
        "--idle-seconds",
        dest="idleSeconds",
        type=float,
        default=20.0,
        help="how long to run the detectors over silence",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="a WAV file to run the wake word over, for accuracy",
    )
    parser.add_argument(
        "--expect",
        type=int,
        default=0,
        help="how many wake words the file contains",
    )
    return parser.parse_args()


def processCpuSeconds() -> float:
    times = os.times()
    return times.user + times.system


async def main() -> int:
    arguments = parseArguments()
    logging.basicConfig(level="WARNING")

    settings = Settings(
        wakeWord={"provider": "openwakeword", "word": arguments.word},
        vad={"provider": arguments.vad},
    )

    wakeWord = createWakeWordProvider(settings)
    vad = createVadProvider(settings)

    started = time.perf_counter()
    await wakeWord.load()
    await vad.load()
    print(f"  {'load':<22} {time.perf_counter() - started:.1f}s")
    print(f"  {'wake word':<22} {wakeWord.describe()}")
    print(f"  {'vad':<22} {vad.describe()}")

    await measureIdle(wakeWord, vad, arguments.idleSeconds)

    if arguments.audio is not None:
        await measureAccuracy(wakeWord, arguments.audio, arguments.expect)

    await wakeWord.unload()
    await vad.unload()
    return 0


async def measureIdle(wakeWord, vad, seconds: float) -> None:
    """Run both detectors over silence and report what it cost.

    Silence is the state a wake-word device spends almost all its time in, so
    this is the figure that decides whether it can be left running.
    """
    vadFrame = AudioBuffer.silence(vad.frameMilliseconds / 1000, 16000)
    wakeFrame = AudioBuffer.silence(wakeWord.frameMilliseconds / 1000, 16000)

    vadFrames = int(seconds * 1000 / vad.frameMilliseconds)
    wakeFrames = int(seconds * 1000 / wakeWord.frameMilliseconds)

    cpuBefore = processCpuSeconds()
    wallBefore = time.perf_counter()

    vadTimes: list[float] = []
    for _ in range(vadFrames):
        started = time.perf_counter()
        vad.isSpeech(vadFrame)
        vadTimes.append(time.perf_counter() - started)

    wakeTimes: list[float] = []
    for _ in range(wakeFrames):
        started = time.perf_counter()
        wakeWord.detect(wakeFrame)
        wakeTimes.append(time.perf_counter() - started)

    cpuUsed = processCpuSeconds() - cpuBefore
    wallUsed = time.perf_counter() - wallBefore

    print(f"  {'simulated idle':<22} {seconds:.0f}s of audio in {wallUsed:.1f}s")
    print(
        f"  {'vad per frame':<22} {statistics.mean(vadTimes) * 1000:.2f}ms "
        f"({vad.frameMilliseconds}ms of audio)"
    )
    print(
        f"  {'wake word per frame':<22} {statistics.mean(wakeTimes) * 1000:.2f}ms "
        f"({wakeWord.frameMilliseconds}ms of audio)"
    )
    label = f"cpu for {seconds:.0f}s audio"
    print(f"  {label:<22} {cpuUsed:.2f}s")
    print(f"  {'estimated idle load':<22} {cpuUsed / seconds * 100:.1f}% of one core")


async def measureAccuracy(wakeWord, path: Path, expected: int) -> None:
    """Count detections in a recording, and false accepts in its quiet parts."""
    if not path.is_file():
        print(f"  audio file not found: {path}")
        return

    buffer = AudioBuffer.fromWavFile(path)
    frames = splitIntoFrames(buffer, wakeWord.frameMilliseconds)

    wakeWord.reset()
    detections = [
        detection for frame in frames if (detection := wakeWord.detect(frame)) is not None
    ]

    print(f"  {'file':<22} {path.name} ({buffer.durationSeconds:.1f}s)")
    print(f"  {'detections':<22} {len(detections)}")
    if expected:
        missed = max(0, expected - len(detections))
        extra = max(0, len(detections) - expected)
        print(f"  {'expected':<22} {expected}")
        print(f"  {'missed / spurious':<22} {missed} / {extra}")

    quiet = [frame for frame in frames if rootMeanSquare(frame) < 0.01]
    print(f"  {'quiet frames':<22} {len(quiet)} of {len(frames)}")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
