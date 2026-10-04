"""Compare the routing layers.

§12 of the brief argues that obvious commands should not need a language
model. This measures what that is worth: the same command routed
deterministically, and phrasing that has to reach the model instead.

Usage:
    python scripts/benchmarkRouting.py
    python scripts/benchmarkRouting.py --model qwen2.5:3b --runs 5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import time

from app.assistant.conversation import Conversation
from app.assistant.intentRouter import IntentRouter
from app.assistant.llmHandler import LlmHandler
from app.assistant.toolRegistry import ToolRegistry
from app.automation.simulatedProvider import SimulatedAutomationProvider
from app.config import Settings
from app.intelligence.llmProvider import createLlmProvider
from app.tools.homeAutomation import HomeAutomationTool, buildAutomationTools

# Phrasing the deterministic patterns cover.
LAYER_ONE = "turn off the bedroom light"
# Phrasing they do not, which must reach the model. Whether the model calls a
# tool or asks a clarifying question varies between runs, so the reported layer
# is not always 2 --- that variability is itself the point of Layer 1.
LAYER_TWO = "it is dark in the bedroom"
# Not a command at all.
LAYER_THREE = "what is the capital of Australia"


def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the routing layers")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--runs", type=int, default=3)
    return parser.parse_args()


def buildRouter(arguments: argparse.Namespace) -> tuple[IntentRouter, SimulatedAutomationProvider]:
    settings = Settings(
        llm={"provider": arguments.provider, "model": arguments.model, "keepAlive": "10m"}
    )
    provider = SimulatedAutomationProvider()
    registry = ToolRegistry()

    automationTool = None
    for tool in buildAutomationTools(provider):
        registry.register(tool)
        if isinstance(tool, HomeAutomationTool):
            automationTool = tool

    router = IntentRouter(
        registry,
        llmHandler=LlmHandler(createLlmProvider(settings), streaming=False),
        automationTool=automationTool,
    )
    return router, provider


async def measure(
    router: IntentRouter, provider: SimulatedAutomationProvider, phrase: str, runs: int
) -> dict[str, object]:
    durations: list[float] = []
    layers: list[int] = []
    reply = ""

    for _ in range(runs):
        provider.calls.clear()
        started = time.perf_counter()
        response = await router.respond(phrase, Conversation())
        durations.append(time.perf_counter() - started)
        layers.append(router.lastLayer)
        reply = response.text

    return {
        "mean": statistics.mean(durations),
        "best": min(durations),
        "layers": sorted(set(layers)),
        "acted": bool(provider.calls),
        "reply": reply,
    }


def report(label: str, result: dict[str, object]) -> None:
    layers = ",".join(str(layer) for layer in result["layers"])  # type: ignore[union-attr]
    acted = "acted" if result["acted"] else "no action"
    print(
        f"  {label:<26} {result['mean']:.3f}s mean, {result['best']:.3f}s best  "
        f"[layer {layers}, {acted}]"
    )
    print(f"  {'':<26} {str(result['reply'])[:66]!r}")


async def main() -> int:
    arguments = parseArguments()
    logging.basicConfig(level="WARNING")

    router, provider = buildRouter(arguments)
    print(f"llm={arguments.provider}/{arguments.model}, runs={arguments.runs}")

    # Load the model before timing, or the first measurement is a load time.
    await router.respond("hello", Conversation())

    deterministic = await measure(router, provider, LAYER_ONE, arguments.runs)
    viaModel = await measure(router, provider, LAYER_TWO, arguments.runs)
    conversation = await measure(router, provider, LAYER_THREE, arguments.runs)

    report("deterministic command", deterministic)
    report("same intent, via model", viaModel)
    report("conversation", conversation)

    ratio = float(viaModel["mean"]) / float(deterministic["mean"])  # type: ignore[arg-type]
    print(f"  {'':<26} deterministic is {ratio:.0f}x faster")

    await router.unload()
    return 0 if deterministic["mean"] < viaModel["mean"] else 1  # type: ignore[operator]


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
