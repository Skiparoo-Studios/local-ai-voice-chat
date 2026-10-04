"""Find out why the language model is slow inside the assistant.

Stage 5 measured a tool-using turn at 0.49 s with nothing else on the GPU.
Stage 7 measured 7.4 s through the API, with speech models resident. This
separates the candidate explanations:

    prompt size     the router sends tool schemas, which are verbose JSON
    residency       whether Ollama keeps the model loaded between turns
    placement       whether it runs on the GPU or spills layers to the CPU

Usage:
    python scripts/diagnoseLlm.py
    python scripts/diagnoseLlm.py --load-speech     # with XTTS and Whisper
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import subprocess
import time
from pathlib import Path

from app.assistant.conversation import Conversation
from app.assistant.llmHandler import LlmHandler
from app.assistant.toolRegistry import ToolRegistry
from app.automation.simulatedProvider import SimulatedAutomationProvider
from app.config import Settings
from app.intelligence.llmProvider import createLlmProvider
from app.tools.homeAutomation import buildAutomationTools

OLLAMA = Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"

QUESTION = "what is the capital of Australia"


def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose language model latency")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--load-speech",
        dest="loadSpeech",
        action="store_true",
        help="load XTTS and Whisper first, as the running assistant does",
    )
    return parser.parse_args()


def ollamaPlacement() -> str:
    """What Ollama says about where the model is running."""
    try:
        result = subprocess.run(
            [str(OLLAMA), "ps"], capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"could not run ollama ps: {error}"

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return lines[1] if len(lines) > 1 else "no model resident"


def gpuMemoryUsedMiB() -> int:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return int(result.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 - diagnostics must not fail the run
        return -1


async def loadSpeechModels(settings: Settings):  # type: ignore[no-untyped-def]
    """Occupy the GPU exactly as the running assistant does."""
    from app.speech.speechToText import createSpeechToTextProvider
    from app.speech.textToSpeech import createTextToSpeechProvider
    from app.speech.voices import VoiceLibrary

    speechToText = createSpeechToTextProvider(settings)
    textToSpeech = createTextToSpeechProvider(settings, VoiceLibrary(Path("voices")))
    await asyncio.gather(speechToText.load(), textToSpeech.load())
    return speechToText, textToSpeech


def buildRegistry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in buildAutomationTools(SimulatedAutomationProvider()):
        registry.register(tool)
    return registry


async def timeGeneration(provider, messages, tools, runs: int) -> dict[str, float]:
    durations: list[float] = []
    tokens = 0

    for _ in range(runs):
        started = time.perf_counter()
        result = await provider.generate(messages, tools=tools)
        durations.append(time.perf_counter() - started)
        tokens = result.completionTokens

    return {
        "mean": statistics.mean(durations),
        "best": min(durations),
        "tokens": tokens,
    }


def report(label: str, result: dict[str, float], promptChars: int) -> None:
    print(
        f"  {label:<28} {result['mean']:.2f}s mean, {result['best']:.2f}s best  "
        f"({int(result['tokens'])} tokens out, {promptChars} chars in)"
    )


async def main() -> int:
    arguments = parseArguments()
    logging.basicConfig(level="WARNING")

    settings = Settings(
        llm={"provider": "ollama", "model": arguments.model, "keepAlive": "10m"},
        textToSpeech={"provider": "xtts", "device": "auto"},
        speechToText={"provider": "faster-whisper", "device": "auto"},
        paths={"models": Path("models"), "voices": Path("voices")},
    )

    print(f"model={arguments.model} speechModelsLoaded={arguments.loadSpeech}")
    print(f"  {'gpu before':<28} {gpuMemoryUsedMiB()} MiB")

    speech = None
    if arguments.loadSpeech:
        speech = await loadSpeechModels(settings)
        print(f"  {'gpu after speech models':<28} {gpuMemoryUsedMiB()} MiB")

    provider = createLlmProvider(settings)
    handler = LlmHandler(provider, streaming=False)
    registry = buildRegistry()

    plain = handler.buildMessages(QUESTION, Conversation())
    tools = registry.schemas()
    toolChars = len(json.dumps(tools))

    # Warm the model before timing, so the first figure is not a load.
    await provider.generate(plain)
    print(f"  {'gpu after model warm':<28} {gpuMemoryUsedMiB()} MiB")
    print(f"  {'placement':<28} {ollamaPlacement()}")

    withoutTools = await timeGeneration(provider, plain, None, arguments.runs)
    report("no tools", withoutTools, len(json.dumps(plain)))

    withTools = await timeGeneration(provider, plain, tools, arguments.runs)
    report("with tool schemas", withTools, len(json.dumps(plain)) + toolChars)

    print(f"  {'tool schemas add':<28} {toolChars} chars to every prompt")
    print(
        f"  {'cost of offering tools':<28} "
        f"{withTools['mean'] - withoutTools['mean']:+.2f}s"
    )
    print(f"  {'placement after':<28} {ollamaPlacement()}")

    await provider.unload()
    if speech is not None:
        await asyncio.gather(speech[0].unload(), speech[1].unload())
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
