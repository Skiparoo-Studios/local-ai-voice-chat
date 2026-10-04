"""Measure speech-to-text latency.

The first transcription after loading pays for CUDA kernel compilation, so it
is reported separately from the warm figures.

Usage:
    python scripts/benchmarkStt.py --device cuda
    python scripts/benchmarkStt.py --device cpu --model tiny
    python scripts/benchmarkStt.py --check-microphone
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import time
from pathlib import Path

from app.audio.audioBuffer import AudioBuffer
from app.audio.input import SoundDeviceInput, resampleBuffer
from app.config import Settings
from app.speech.speechToText import createSpeechToTextProvider

FIXTURE = Path("tests/data/utterance.wav")


def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a speech-to-text provider")
    parser.add_argument("--provider", default="faster-whisper")
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--audio", type=Path, default=FIXTURE)
    parser.add_argument("--runs", type=int, default=3, help="timed runs")
    parser.add_argument(
        "--check-microphone",
        dest="checkMicrophone",
        action="store_true",
        help="record from the microphone and report what arrived",
    )
    parser.add_argument(
        "--input-device",
        dest="inputDevice",
        default=None,
        help="which input to check, by index or name. Defaults to audio.inputDevice",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="how long to listen while checking the microphone",
    )
    return parser.parse_args()


async def checkMicrophone(device: str | int | None, seconds: float) -> int:
    """Listen, and report level second by second.

    Reported as it goes rather than as one number at the end, because the
    useful test is speaking into it and watching the meter move. A single
    figure cannot distinguish a dead microphone from a quiet room.
    """
    source = SoundDeviceInput(device=device)
    print(f"  input: {source.describe()}")
    print(f"  say something for the next {seconds:.0f} seconds...\n")

    loudest = 0.0
    elapsed = 0.0
    window: list[float] = []

    async for frame in source.frames(32):
        samples = frame.toFloatSamples()
        peak = max((abs(sample) for sample in samples), default=0.0)
        loudest = max(loudest, peak)
        window.append(peak)
        elapsed += frame.durationSeconds

        # One line a second, so the meter is readable rather than a blur.
        if len(window) >= int(1000 / 32):
            level = max(window)
            window.clear()
            bars = min(40, int(level * 40))
            print(f"  {'#' * bars:<40} {level:.3f}")

        if elapsed >= seconds:
            break

    source.close()
    print(f"\n  loudest: {loudest:.4f}")

    if loudest < 0.02:
        print(
            "  NOTHING HEARD. Either this is the wrong device or it is muted.\n"
            "  Run --list-input-devices and try another with --input-device."
        )
        return 1
    if loudest > 0.99:
        print(
            "  CLIPPING. The input gain is too high, which will distort speech.\n"
            "  Turn it down in Windows sound settings."
        )
        return 1

    print("  Good level.")
    return 0


def deviceMemoryUsedGiB() -> float | None:
    """GPU memory in use, from the driver's point of view.

    CTranslate2 allocates outside torch's allocator, so torch.cuda.memory_allocated
    would report zero for Whisper. mem_get_info reflects everything on the device.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1024**3
    except Exception:  # noqa: BLE001 - diagnostics must not break the benchmark
        return None


async def timeTranscription(provider, audio: AudioBuffer, runs: int) -> dict[str, float]:
    durations: list[float] = []
    text = ""

    for _ in range(runs):
        started = time.perf_counter()
        transcript = await provider.transcribe(audio)
        durations.append(time.perf_counter() - started)
        text = transcript.text

    mean = statistics.mean(durations)
    return {
        "mean": mean,
        "best": min(durations),
        "realTimeFactor": mean / audio.durationSeconds if audio.durationSeconds else 0.0,
        "text": text,
    }


async def main() -> int:
    arguments = parseArguments()
    logging.basicConfig(level="WARNING")

    if arguments.checkMicrophone:
        device = arguments.inputDevice
        if device is None:
            device = Settings().audio.inputDevice
        elif device.isdigit():
            device = int(device)
        return await checkMicrophone(device, arguments.seconds)

    if not arguments.audio.is_file():
        print(f"audio fixture not found: {arguments.audio}")
        return 2

    settings = Settings(
        speechToText={
            "provider": arguments.provider,
            "model": arguments.model,
            "device": arguments.device,
        },
        paths={"models": Path("models")},
    )
    provider = createSpeechToTextProvider(settings)

    print(f"provider={arguments.provider} model={arguments.model} device={arguments.device}")

    baseline = deviceMemoryUsedGiB()

    started = time.perf_counter()
    await provider.load()
    print(f"  {'load':<22} {time.perf_counter() - started:.1f}s  ({provider.describe()})")

    afterLoad = deviceMemoryUsedGiB()
    if baseline is not None and afterLoad is not None:
        print(f"  {'device memory':<22} {afterLoad:.2f} GiB used ({afterLoad - baseline:+.2f})")

    audio = resampleBuffer(
        AudioBuffer.fromWavFile(arguments.audio), provider.requiredSampleRate
    )

    started = time.perf_counter()
    first = await provider.transcribe(audio)
    coldSeconds = time.perf_counter() - started
    print(
        f"  {'first transcription':<22} {coldSeconds:.2f}s "
        f"for {audio.durationSeconds:.2f}s audio "
        f"(RTF {coldSeconds / audio.durationSeconds:.2f})"
    )
    print(f"  {'transcript':<22} {first.text!r}")

    result = await timeTranscription(provider, audio, arguments.runs)
    print(
        f"  {'warm':<22} {result['mean']:.2f}s mean, {result['best']:.2f}s best, "
        f"for {audio.durationSeconds:.2f}s audio (RTF {result['realTimeFactor']:.2f})"
    )

    await provider.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
