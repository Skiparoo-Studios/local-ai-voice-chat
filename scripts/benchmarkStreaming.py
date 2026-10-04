"""Compare streaming and non-streaming replies.

Runs the same reply through the runtime both ways and reports time to first
audio. Everything except the streaming flag is held constant, so the difference
is attributable.

By default the language model is simulated, emitting words at a fixed rate.
That isolates the effect of overlapping synthesis with generation, and lets the
comparison run without a model service. Point it at a real one with
--llm-provider ollama.

Usage:
    python scripts/benchmarkStreaming.py
    python scripts/benchmarkStreaming.py --tts-provider tone
    python scripts/benchmarkStreaming.py --llm-provider ollama --model qwen2.5:3b
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import time
from pathlib import Path

from app.assistant.llmHandler import LlmHandler
from app.assistant.runtime import AssistantRuntime
from app.assistant.session import Session
from app.audio.input import MemoryAudioInput
from app.audio.output import MemoryAudioOutput
from app.config import Settings
from app.events import EventBus
from app.intelligence.llmProvider import createLlmProvider
from app.intelligence.providers.scriptedLlmProvider import ScriptedLlmProvider
from app.speech.models.scriptedProvider import ScriptedProvider
from app.speech.textToSpeech import createTextToSpeechProvider
from app.speech.voices import VoiceLibrary

QUESTION = "What is the weather like today?"

# Three sentences: long enough that overlapping matters, short enough to be a
# realistic spoken reply.
SIMULATED_REPLY = (
    "It's fairly mild outside at the moment, around fifteen degrees. "
    "There is a good chance of rain later this evening, so an umbrella "
    "would be sensible. Tomorrow looks brighter and a little warmer."
)

# 25 ms per word is roughly 40 tokens per second, a plausible rate for a small
# quantised model on a consumer GPU.
DEFAULT_FRAGMENT_DELAY = 0.025


def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare streaming with non-streaming")
    parser.add_argument("--tts-provider", dest="ttsProvider", default="xtts")
    parser.add_argument("--tts-device", dest="ttsDevice", default="auto")
    parser.add_argument(
        "--llm-provider",
        dest="llmProvider",
        default="simulated",
        help="'simulated', or a configured provider such as 'ollama'",
    )
    parser.add_argument("--model", default="", help="model name for a real provider")
    parser.add_argument("--base-url", dest="baseUrl", default="")
    parser.add_argument(
        "--fragment-delay",
        dest="fragmentDelay",
        type=float,
        default=DEFAULT_FRAGMENT_DELAY,
        help="seconds per word when simulating a model",
    )
    parser.add_argument("--runs", type=int, default=3)
    return parser.parse_args()


def buildSettings(arguments: argparse.Namespace) -> Settings:
    llm: dict[str, object] = {"provider": arguments.llmProvider, "model": arguments.model}
    if arguments.baseUrl:
        llm["baseUrl"] = arguments.baseUrl

    return Settings(
        textToSpeech={"provider": arguments.ttsProvider, "device": arguments.ttsDevice},
        llm=llm if arguments.llmProvider != "simulated" else {},
        paths={"models": Path("models"), "voices": Path("voices")},
    )


def buildRuntime(settings: Settings, arguments: argparse.Namespace, streaming: bool):
    """One runtime per mode, sharing nothing so neither warms the other."""
    if arguments.llmProvider == "simulated":
        llmProvider = ScriptedLlmProvider(
            replies=(SIMULATED_REPLY,), fragmentDelaySeconds=arguments.fragmentDelay
        )
    else:
        llmProvider = createLlmProvider(settings)

    output = MemoryAudioOutput()
    runtime = AssistantRuntime(
        speechToText=ScriptedProvider(phrases=(QUESTION,)),
        textToSpeech=createTextToSpeechProvider(settings, VoiceLibrary(Path("voices"))),
        handler=LlmHandler(llmProvider, streaming=streaming),
        audioInput=MemoryAudioInput(),
        audioOutput=output,
        bus=EventBus(),
        session=Session.create(),
        language="en",
        voice=settings.textToSpeech.voice,
        warmUpOnStart=True,
        streaming=streaming,
    )
    return runtime, output


async def measure(
    runtime: AssistantRuntime, output: MemoryAudioOutput, runs: int
) -> dict[str, float | str]:
    await runtime.start()

    firstAudio: list[float] = []
    pieces = 0
    reply = ""

    for _ in range(runs):
        result = await runtime.handleText(QUESTION)
        firstAudio.append(result.timing.thinkingSeconds)
        pieces = len(output.played)
        output.played.clear()
        reply = result.assistantText
        runtime.conversation.clear()

    await runtime.stop()

    return {
        "mean": statistics.mean(firstAudio),
        "best": min(firstAudio),
        "pieces": pieces,
        "reply": reply,
    }


async def main() -> int:
    arguments = parseArguments()
    logging.basicConfig(level="WARNING")

    settings = buildSettings(arguments)
    print(
        f"tts={arguments.ttsProvider} on {arguments.ttsDevice}, "
        f"llm={arguments.llmProvider}, runs={arguments.runs}"
    )

    started = time.perf_counter()
    complete = await measure(*buildRuntime(settings, arguments, streaming=False), arguments.runs)
    streamed = await measure(*buildRuntime(settings, arguments, streaming=True), arguments.runs)
    print(f"  measured in {time.perf_counter() - started:.0f}s")

    print(f"  {'without streaming':<22} {complete['mean']:.2f}s mean, {complete['best']:.2f}s best")
    print(
        f"  {'with streaming':<22} {streamed['mean']:.2f}s mean, {streamed['best']:.2f}s best "
        f"({streamed['pieces']} pieces)"
    )

    saved = float(complete["mean"]) - float(streamed["mean"])
    share = saved / float(complete["mean"]) * 100 if complete["mean"] else 0.0
    print(f"  {'improvement':<22} {saved:.2f}s faster to first audio ({share:.0f}%)")
    print(f"  {'reply':<22} {str(streamed['reply'])[:70]!r}...")

    return 0 if saved > 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
