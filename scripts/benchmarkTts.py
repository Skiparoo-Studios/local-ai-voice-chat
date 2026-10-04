"""Measure text-to-speech latency.

The first synthesis after loading pays for CUDA kernel compilation and cuDNN
autotuning, so it is treated as a warm-up and excluded from the reported
figures. Both are reported separately, because cold latency is what a user
feels on the first interaction after startup.

Usage:
    python scripts/benchmarkTts.py --provider xtts --device cuda
    python scripts/benchmarkTts.py --provider xtts --device cpu --runs 2
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import time
from pathlib import Path

from app.config import Settings
from app.speech.textToSpeech import createTextToSpeechProvider
from app.speech.voices import VoiceLibrary

SHORT_TEXT = "Turn the kitchen light off."
LONG_TEXT = (
    "The assistant runs entirely on local hardware, so speech recognition, "
    "language understanding and speech synthesis all happen on this machine. "
    "Nothing is sent to a cloud service unless you configure one deliberately."
)


def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a text-to-speech provider")
    parser.add_argument("--provider", default="xtts")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--voice", default="default")
    parser.add_argument("--runs", type=int, default=3, help="timed runs per utterance")
    return parser.parse_args()


def describeVram(stage: str) -> str | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"{stage}: {allocated:.2f} GiB allocated, {reserved:.2f} GiB reserved"
    except ImportError:
        return None


async def timeSynthesis(provider, text: str, voice: str, runs: int) -> dict[str, float]:
    durations: list[float] = []
    audioSeconds = 0.0

    for _ in range(runs):
        started = time.perf_counter()
        audio = await provider.synthesise(text, voice=voice)
        durations.append(time.perf_counter() - started)
        audioSeconds = audio.durationSeconds

    mean = statistics.mean(durations)
    return {
        "mean": mean,
        "best": min(durations),
        "audioSeconds": audioSeconds,
        "realTimeFactor": mean / audioSeconds if audioSeconds else 0.0,
    }


def report(label: str, result: dict[str, float]) -> None:
    print(
        f"  {label:<22} {result['mean']:.2f}s mean, {result['best']:.2f}s best, "
        f"for {result['audioSeconds']:.2f}s audio (RTF {result['realTimeFactor']:.2f})"
    )


async def main() -> int:
    arguments = parseArguments()
    logging.basicConfig(level="WARNING")

    settings = Settings(
        textToSpeech={
            "provider": arguments.provider,
            "device": arguments.device,
            "voice": arguments.voice,
        },
        paths={"voices": Path("voices"), "models": Path("models")},
    )
    provider = createTextToSpeechProvider(settings, VoiceLibrary(Path("voices")))

    print(f"provider={arguments.provider} device={arguments.device} voice={arguments.voice}")

    started = time.perf_counter()
    await provider.load()
    loadSeconds = time.perf_counter() - started
    print(f"  {'load':<22} {loadSeconds:.1f}s  ({provider.describe()})")

    vram = describeVram("after load")
    if vram:
        print(f"  {vram}")

    # First call pays for kernel compilation; report it, then discard it.
    started = time.perf_counter()
    firstAudio = await provider.synthesise(SHORT_TEXT, voice=arguments.voice)
    coldSeconds = time.perf_counter() - started
    print(
        f"  {'first utterance':<22} {coldSeconds:.2f}s "
        f"for {firstAudio.durationSeconds:.2f}s audio "
        f"(RTF {coldSeconds / firstAudio.durationSeconds:.2f})"
    )

    voice, runs = arguments.voice, arguments.runs
    report("short, warm", await timeSynthesis(provider, SHORT_TEXT, voice, runs))
    report("long, warm", await timeSynthesis(provider, LONG_TEXT, voice, runs))

    vram = describeVram("after synthesis")
    if vram:
        print(f"  {vram}")

    await provider.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
