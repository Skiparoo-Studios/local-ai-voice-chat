"""Find out where synthesis time actually goes.

Sentence-level streaming has existed since Stage 4, but a one-sentence reply
has nothing to pipeline, so a short confirmation still waits for the whole
thing. This measures what that costs and what could be done about it:

    scaling     whether time is fixed per call or proportional to length
    streaming   whether XTTS can emit audio before a sentence is finished
    repetition  how often the assistant says exactly the same thing

Usage:
    python scripts/diagnoseTts.py
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

# Representative of what the assistant actually says, shortest first.
SAMPLES = [
    ("confirmation", "I've turned the kitchen light off."),
    ("short answer", "The capital of Australia is Canberra."),
    ("two sentences", "The kitchen light is now off. Would you like the lamp on instead?"),
    (
        "conversational",
        "It's fairly mild outside at the moment, around fifteen degrees. "
        "There is a good chance of rain later this evening, so an umbrella "
        "would be sensible.",
    ),
]


def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose synthesis latency")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--runs", type=int, default=3)
    return parser.parse_args()


async def timeSynthesis(provider, text: str, runs: int) -> dict[str, float]:
    durations: list[float] = []
    audioSeconds = 0.0

    for _ in range(runs):
        started = time.perf_counter()
        audio = await provider.synthesise(text)
        durations.append(time.perf_counter() - started)
        audioSeconds = audio.durationSeconds

    return {
        "mean": statistics.mean(durations),
        "best": min(durations),
        "audioSeconds": audioSeconds,
        "characters": len(text),
    }


def reportScaling(label: str, result: dict[str, float]) -> None:
    perCharacter = result["mean"] / result["characters"] * 1000
    print(
        f"  {label:<16} {result['mean']:.2f}s for {result['audioSeconds']:.2f}s audio  "
        f"({int(result['characters'])} chars, {perCharacter:.1f}ms/char, "
        f"RTF {result['mean'] / result['audioSeconds']:.2f})"
    )


def inspectStreaming(provider) -> None:
    """Whether the underlying model can emit audio mid-sentence.

    Sentence-level streaming cannot help a one-sentence reply. Chunk-level
    streaming can, and XTTS exposes it if the low-level model is reachable.
    """
    model = getattr(provider, "_model", None)
    if model is None:
        print("  streaming        low-level model unavailable")
        return

    hasStream = hasattr(model, "inference_stream")
    print(f"  streaming        inference_stream present: {hasStream}")
    if hasStream:
        import inspect

        signature = inspect.signature(model.inference_stream)
        names = [name for name in signature.parameters if name != "self"]
        print(f"  {'':<16} parameters: {', '.join(names[:8])}")


async def measureStreamingFirstChunk(provider, text: str) -> None:
    """Time to the first audio chunk, against the whole utterance."""
    model = getattr(provider, "_model", None)
    if model is None or not hasattr(model, "inference_stream"):
        return

    source = provider.resolveSpeaker("default")
    latent, embedding = provider._conditioningFor(source)

    started = time.perf_counter()
    firstAt = 0.0
    chunks = 0

    for _ in model.inference_stream(text, "en", latent, embedding):
        if not firstAt:
            firstAt = time.perf_counter() - started
        chunks += 1

    total = time.perf_counter() - started
    print(
        f"  {'streamed':<16} first chunk {firstAt:.2f}s, "
        f"{chunks} chunks, {total:.2f}s total"
    )


async def main() -> int:
    arguments = parseArguments()
    logging.basicConfig(level="WARNING")

    settings = Settings(
        textToSpeech={"provider": "xtts", "device": arguments.device},
        paths={"models": Path("models"), "voices": Path("voices")},
    )
    provider = createTextToSpeechProvider(settings, VoiceLibrary(Path("voices")))

    started = time.perf_counter()
    await provider.load()
    print(f"  {'load':<16} {time.perf_counter() - started:.1f}s")

    # Warm before timing, as every other benchmark here does.
    await provider.synthesise("Ready.")

    print("\n  how synthesis time scales with what is said")
    results = []
    for label, text in SAMPLES:
        result = await timeSynthesis(provider, text, arguments.runs)
        reportScaling(label, result)
        results.append(result)

    shortest, longest = results[0], results[-1]
    perCharacter = (longest["mean"] - shortest["mean"]) / (
        longest["characters"] - shortest["characters"]
    )
    fixed = shortest["mean"] - perCharacter * shortest["characters"]
    print(
        f"\n  {'fitted':<16} {fixed * 1000:.0f}ms fixed per call, "
        f"{perCharacter * 1000:.1f}ms per character"
    )

    print("\n  what the model can do about it")
    inspectStreaming(provider)
    await measureStreamingFirstChunk(provider, SAMPLES[0][1])

    await provider.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
