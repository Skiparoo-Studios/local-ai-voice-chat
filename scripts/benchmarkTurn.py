"""Measure a full conversation turn.

Feeds a recorded utterance through the real pipeline --- capture, Whisper, the
handler, XTTS --- and reports where the time goes. Playback is excluded from
the headline figure: what a user feels is the gap between finishing speaking
and hearing the first audio, not how long the reply takes to play.

Usage:
    python scripts/benchmarkTurn.py
    python scripts/benchmarkTurn.py --stt-device cpu --runs 5
    python scripts/benchmarkTurn.py --no-warm-up
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import time
from pathlib import Path

from app.assistant.handlers import createResponseHandler
from app.assistant.runtime import AssistantRuntime, TurnResult
from app.assistant.session import Session
from app.audio.input import WavFileInput
from app.audio.output import MemoryAudioOutput
from app.config import Settings
from app.events import AssistantState, AssistantStateChanged, EventBus
from app.speech.speechToText import createSpeechToTextProvider
from app.speech.textToSpeech import createTextToSpeechProvider
from app.speech.voices import VoiceLibrary

FIXTURE = Path("tests/data/utterance.wav")

EXPECTED_STATES = [
    AssistantState.listening,
    AssistantState.thinking,
    AssistantState.speaking,
    AssistantState.idle,
]


def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a full conversation turn")
    parser.add_argument("--audio", type=Path, default=FIXTURE)
    parser.add_argument("--stt-device", dest="sttDevice", default="auto")
    parser.add_argument("--tts-device", dest="ttsDevice", default="auto")
    parser.add_argument("--stt-model", dest="sttModel", default="small")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--no-warm-up",
        dest="warmUp",
        action="store_false",
        help="skip the startup warm-up, to measure what it saves",
    )
    return parser.parse_args()


async def main() -> int:
    arguments = parseArguments()
    logging.basicConfig(level="WARNING")

    if not arguments.audio.is_file():
        print(f"audio fixture not found: {arguments.audio}")
        return 2

    settings = Settings(
        speechToText={"device": arguments.sttDevice, "model": arguments.sttModel},
        textToSpeech={"device": arguments.ttsDevice},
        assistant={"warmUpOnStart": arguments.warmUp},
        paths={"models": Path("models"), "voices": Path("voices")},
    )

    bus = EventBus()
    states: list[AssistantState] = []
    bus.subscribe(AssistantStateChanged, lambda event: states.append(event.state))

    speechToText = createSpeechToTextProvider(settings)
    textToSpeech = createTextToSpeechProvider(settings, VoiceLibrary(Path("voices")))
    output = MemoryAudioOutput()

    runtime = AssistantRuntime(
        speechToText=speechToText,
        textToSpeech=textToSpeech,
        handler=createResponseHandler(settings),
        audioInput=WavFileInput(arguments.audio, speechToText.requiredSampleRate),
        audioOutput=output,
        bus=bus,
        session=Session.create(),
        language="en",
        voice=settings.textToSpeech.voice,
        warmUpOnStart=arguments.warmUp,
    )

    print(f"stt={arguments.sttDevice} tts={arguments.ttsDevice} warmUp={arguments.warmUp}")

    started = time.perf_counter()
    await runtime.start()
    print(f"  {'startup':<20} {time.perf_counter() - started:.1f}s")

    states.clear()
    first = await runtime.runTurn()
    reportTurn("first turn", first)

    observed = [state for state in states if state in EXPECTED_STATES]
    matches = observed == EXPECTED_STATES
    print(f"  {'state sequence':<20} {' -> '.join(s.value for s in observed)}")
    print(f"  {'':<20} {'as expected' if matches else 'UNEXPECTED'}")

    results: list[TurnResult] = []
    for _ in range(arguments.runs):
        results.append(await runtime.runTurn())

    thinking = [result.timing.thinkingSeconds for result in results]
    print(
        f"  {'warm turns':<20} {statistics.mean(thinking):.2f}s mean, "
        f"{min(thinking):.2f}s best to first audio"
    )
    reportTurn("warm breakdown", results[-1])

    lastTiming = results[-1].timing
    print(f"  {'reply audio':<20} {lastTiming.audioSeconds:.2f}s")
    print(f"  {'transcript':<20} {results[-1].userText!r}")
    print(f"  {'reply':<20} {results[-1].assistantText!r}")

    await runtime.stop()
    return 0 if matches else 1


def reportTurn(label: str, result: TurnResult) -> None:
    timing = result.timing
    print(
        f"  {label:<20} capture {timing.captureSeconds:.2f}s  "
        f"stt {timing.transcriptionSeconds:.2f}s  "
        f"think {timing.responseSeconds:.3f}s  "
        f"tts {timing.synthesisSeconds:.2f}s  "
        f"= {timing.thinkingSeconds:.2f}s"
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
