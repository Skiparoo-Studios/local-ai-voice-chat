"""Application entry point.

Three modes so far:

    speak     an interactive text-to-speech prompt (Stage 1)
    listen    push-to-talk transcription (Stage 2)
    converse  speak, get a spoken reply (Stage 3)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from app.assistant.conversationRepl import ConversationRepl
from app.assistant.handlers import UnknownHandlerError, createResponseHandler
from app.assistant.listener import ContinuousListener
from app.assistant.questionBank import QuestionBankError
from app.assistant.runtime import createRuntime
from app.audio.input import (
    AudioInputError,
    MemoryAudioInput,
    createAudioInput,
    listInputDevices,
)
from app.audio.output import AudioOutputError, createAudioOutput, listOutputDevices
from app.audio.vad import SegmenterSettings, VadError, createVadProvider
from app.audio.wakeWord import WakeWordError, createWakeWordProvider
from app.automation.automationProvider import AutomationError
from app.config import ConfigurationError, Settings, loadSettings
from app.events import AssistantState, AssistantStateChanged, Event, EventBus
from app.intelligence.llmProvider import LlmError
from app.speech.device import DeviceUnavailableError
from app.speech.listenRepl import ListenRepl
from app.speech.models.scriptedProvider import ScriptedProvider
from app.speech.repl import SpeechRepl
from app.speech.speechToText import TranscriptionError, createSpeechToTextProvider
from app.speech.textToSpeech import SpeechSynthesisError, createTextToSpeechProvider
from app.speech.voices import VoiceLibrary, VoiceNotFoundError

logger = logging.getLogger("app")

# Failures that mean something is missing or misconfigured, rather than that
# the code is wrong. Each of these already carries a message saying what to do,
# so it is printed on its own; a traceback would only bury it. Anything not
# listed here keeps its traceback, because it is a bug worth seeing in full.
RUNTIME_ERRORS = (
    SpeechSynthesisError,
    TranscriptionError,
    AudioOutputError,
    AudioInputError,
    AutomationError,
    DeviceUnavailableError,
    LlmError,
    QuestionBankError,
    UnknownHandlerError,
    VadError,
    VoiceNotFoundError,
    WakeWordError,
)


def parseArguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="app", description="Local voice assistant")
    parser.add_argument(
        "--mode",
        default="speak",
        choices=["speak", "listen", "converse", "wake", "serve", "client"],
        help=(
            "speak: text-to-speech prompt. listen: push-to-talk transcription. "
            "converse: full conversation loop. wake: hands-free. serve: HTTP API. "
            "client: microphone and speaker for a remote assistant"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="path to a JSON configuration file (default: config/settings.json if present)",
    )
    parser.add_argument(
        "--log-level",
        dest="logLevel",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="override the configured log level",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="override the provider for the selected mode (e.g. xtts, tone, scripted)",
    )
    parser.add_argument("--voice", default=None, help="override the configured voice (speak mode)")
    parser.add_argument(
        "--say",
        default=None,
        help="speak this text and exit, instead of starting the prompt",
    )
    parser.add_argument(
        "--transcribe",
        type=Path,
        default=None,
        help="transcribe this WAV file and exit (implies --mode listen)",
    )
    parser.add_argument(
        "--list-devices",
        dest="listDevices",
        action="store_true",
        help="list audio output devices and exit",
    )
    parser.add_argument(
        "--list-input-devices",
        dest="listInputDevices",
        action="store_true",
        help="list audio input devices and exit",
    )
    parser.add_argument(
        "--no-microphone",
        dest="noMicrophone",
        action="store_true",
        help="converse mode: type messages instead of speaking them",
    )
    parser.add_argument(
        "--server",
        default=None,
        help="client mode: the assistant to connect to, e.g. http://192.168.1.10:8000",
    )
    parser.add_argument(
        "--send",
        default=None,
        help="client mode: send this text and exit, instead of listening",
    )
    parser.add_argument(
        "--handler",
        default=None,
        choices=["rules", "llm", "echo"],
        help="converse mode: override the configured response handler",
    )
    parser.add_argument(
        "--no-streaming",
        dest="noStreaming",
        action="store_true",
        help="converse mode: wait for the whole reply before speaking",
    )
    parser.add_argument(
        "--toddler",
        action="store_true",
        help=(
            "start in toddler mode: short, simple replies and questions from "
            "config/toddler.json. Say 'end toddler mode' to leave it"
        ),
    )
    return parser.parse_args(argv)


def configureLogging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def applyOverrides(settings: Settings, arguments: argparse.Namespace) -> Settings:
    """Apply command-line overrides on top of loaded configuration."""
    updates: dict[str, object] = {}

    if arguments.server:
        updates["client"] = settings.client.model_copy(
            update={"serverUrl": arguments.server}
        )

    assistant: dict[str, object] = {}
    if arguments.handler:
        assistant["handler"] = arguments.handler
    if arguments.noStreaming:
        assistant["streaming"] = False
    if assistant:
        updates["assistant"] = settings.assistant.model_copy(update=assistant)

    if arguments.toddler:
        updates["toddler"] = settings.toddler.model_copy(update={"enabled": True})

    if arguments.mode == "listen":
        if arguments.provider:
            updates["speechToText"] = settings.speechToText.model_copy(
                update={"provider": arguments.provider}
            )
    else:
        textToSpeech: dict[str, object] = {}
        if arguments.provider:
            textToSpeech["provider"] = arguments.provider
        if arguments.voice:
            textToSpeech["voice"] = arguments.voice
        if textToSpeech:
            updates["textToSpeech"] = settings.textToSpeech.model_copy(update=textToSpeech)

    return settings.model_copy(update=updates) if updates else settings


def connectBookReader(runtime) -> None:  # type: ignore[no-untyped-def]
    """Tell the book reader how to know the assistant is still talking.

    The reader must not start a chapter over the top of the reply announcing
    it, and only the runtime knows when that reply has finished. The runtime is
    built after the handler, so the connection is made here rather than passed
    in.
    """
    reader = getattr(runtime.handler, "reader", None)
    if reader is None:
        return
    reader.isBusy = lambda: runtime.state is not AssistantState.idle


def subscribeLogging(bus: EventBus) -> None:
    def onEvent(event: Event) -> None:
        logger.debug("event %s", event.toDict())

    bus.subscribeAll(onEvent)


async def runSpeak(settings: Settings, sayText: str | None) -> int:
    """Load the text-to-speech provider, then speak one line or start the prompt."""
    bus = EventBus()
    subscribeLogging(bus)

    voices = VoiceLibrary(settings.paths.voices)
    provider = createTextToSpeechProvider(settings, voices)
    output = createAudioOutput(settings)

    logger.info("%s starting", settings.assistant.name)
    logger.info("Text-to-speech provider: %s", settings.textToSpeech.provider)

    started = time.perf_counter()
    await provider.load()
    logger.info("Ready in %.1fs (%s)", time.perf_counter() - started, provider.describe())

    await bus.publish(AssistantStateChanged(state=AssistantState.idle))

    repl = SpeechRepl(
        provider,
        output,
        voices,
        bus,
        voice=settings.textToSpeech.voice,
        language=settings.textToSpeech.language,
        outputsDirectory=settings.paths.outputs,
    )

    try:
        if sayText:
            await repl.speakOnce(sayText)
            return 0
        return await repl.run()
    finally:
        await bus.drain()
        await provider.unload()
        output.close()
        logger.info("Stopped")


async def runListen(settings: Settings, transcribePath: Path | None) -> int:
    """Load the speech-to-text provider, then transcribe a file or start the prompt."""
    bus = EventBus()
    subscribeLogging(bus)

    provider = createSpeechToTextProvider(settings)

    logger.info("%s starting", settings.assistant.name)
    logger.info("Speech-to-text provider: %s", settings.speechToText.provider)

    started = time.perf_counter()
    await provider.load()
    logger.info("Ready in %.1fs (%s)", time.perf_counter() - started, provider.describe())

    # A file transcription needs no microphone, so do not open one.
    audioInput = (
        createAudioInput(settings)
        if transcribePath is None
        else _fileInput(transcribePath, provider.requiredSampleRate)
    )

    await bus.publish(AssistantStateChanged(state=AssistantState.idle))

    repl = ListenRepl(
        provider,
        audioInput,
        bus,
        language=settings.speechToText.language or None,
        maximumSeconds=settings.speechToText.maximumRecordingSeconds,
        outputsDirectory=settings.paths.outputs,
    )

    try:
        if transcribePath is not None:
            await repl.transcribeFile(transcribePath)
            return 0
        return await repl.run()
    finally:
        await bus.drain()
        await provider.unload()
        audioInput.close()
        logger.info("Stopped")


def _fileInput(path: Path, sampleRate: int):  # type: ignore[no-untyped-def]
    from app.audio.input import WavFileInput

    return WavFileInput(path, sampleRate)


async def runConverse(settings: Settings, useMicrophone: bool) -> int:
    """Load everything and run the conversation loop."""
    bus = EventBus()
    subscribeLogging(bus)

    voices = VoiceLibrary(settings.paths.voices)
    textToSpeech = createTextToSpeechProvider(settings, voices)
    audioOutput = createAudioOutput(settings)
    handler = createResponseHandler(
        settings, audioOutput=audioOutput, textToSpeech=textToSpeech
    )

    # Typed input never reaches the recogniser, so neither the capture device
    # nor the Whisper model is opened. Loading a model that cannot be used
    # would cost several seconds and half a gigabyte of VRAM for nothing.
    if useMicrophone:
        speechToText = createSpeechToTextProvider(settings)
        audioInput = createAudioInput(settings)
    else:
        speechToText = ScriptedProvider(phrases=("",))
        audioInput = MemoryAudioInput()

    logger.info("%s starting", settings.assistant.name)
    logger.info(
        "Providers: stt=%s tts=%s",
        settings.speechToText.provider if useMicrophone else "none (typed input)",
        settings.textToSpeech.provider,
    )

    runtime = createRuntime(
        settings,
        speechToText=speechToText,
        textToSpeech=textToSpeech,
        handler=handler,
        audioInput=audioInput,
        audioOutput=audioOutput,
        bus=bus,
    )

    connectBookReader(runtime)

    started = time.perf_counter()
    await runtime.start()
    logger.info("Ready in %.1fs", time.perf_counter() - started)

    repl = ConversationRepl(
        runtime,
        outputsDirectory=settings.paths.outputs,
        useMicrophone=useMicrophone,
    )

    try:
        return await repl.run()
    finally:
        await bus.drain()
        await runtime.stop()
        audioInput.close()
        audioOutput.close()
        logger.info("Stopped")


async def runWake(settings: Settings) -> int:
    """Load everything and listen hands-free until interrupted."""
    bus = EventBus()
    subscribeLogging(bus)

    voices = VoiceLibrary(settings.paths.voices)
    textToSpeech = createTextToSpeechProvider(settings, voices)
    audioOutput = createAudioOutput(settings)
    runtime = createRuntime(
        settings,
        speechToText=createSpeechToTextProvider(settings),
        textToSpeech=textToSpeech,
        handler=createResponseHandler(
            settings, audioOutput=audioOutput, textToSpeech=textToSpeech
        ),
        audioInput=createAudioInput(settings),
        audioOutput=audioOutput,
        bus=bus,
    )
    connectBookReader(runtime)

    listener = ContinuousListener(
        runtime,
        createAudioInput(settings),
        createWakeWordProvider(settings),
        createVadProvider(settings),
        bus,
        segmenter=SegmenterSettings(
            startFrames=settings.vad.startFrames,
            silenceFrames=settings.vad.silenceFrames,
            prerollFrames=settings.vad.prerollFrames,
            maximumSeconds=settings.vad.maximumSeconds,
            minimumSeconds=settings.vad.minimumSeconds,
        ),
        allowBargeIn=settings.wakeWord.allowBargeIn,
        acknowledgeWake=settings.wakeWord.acknowledge,
        echoGuardSeconds=settings.audio.echoGuardSeconds,
    )

    logger.info("%s starting", settings.assistant.name)

    started = time.perf_counter()
    await runtime.start()
    await listener.start()
    logger.info("Ready in %.1fs", time.perf_counter() - started)

    print(f"  {listener.describe()}")
    if settings.wakeWord.provider == "always-awake":
        print("  Listening continuously. Ctrl-C to stop.")
    else:
        print(f"  Say '{settings.wakeWord.word}' to wake me. Ctrl-C to stop.")

    try:
        statistics = await listener.run()
        print(f"  {statistics.describe()}")
        return 0
    except KeyboardInterrupt:
        print()
        return 0
    finally:
        await listener.stop()
        await bus.drain()
        await runtime.stop()
        logger.info("Stopped")


async def runClient(settings: Settings, sendText: str | None) -> int:
    """Act as a microphone and speaker for an assistant running elsewhere."""
    from app.client.remoteClient import ClientError, RemoteClient

    client = RemoteClient.fromSettings(settings)

    try:
        if sendText:
            reply = await client.sendText(sendText)
            print(f"  {reply.get('text') or reply.get('message') or '(no reply)'}")
            return 0

        await client.start()
        print(f"  {client.describe()}")
        if settings.wakeWord.provider == "always-awake":
            print("  Listening continuously. Ctrl-C to stop.")
        else:
            print(f"  Say '{settings.wakeWord.word}' to wake me. Ctrl-C to stop.")

        statistics = await client.run()
        print(f"  {statistics.describe()}")
        return 0
    except ClientError as error:
        logger.error("%s", error)
        return 1
    except KeyboardInterrupt:
        print()
        return 0
    finally:
        client.close()


def describePortConflict(host: str, port: int) -> str | None:
    """Report whether an address is already in use, before anything loads."""
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host if host != "localhost" else "127.0.0.1", port))
    except OSError as error:
        return (
            f"Cannot listen on {host}:{port}: {error.strerror or error}\n"
            "Something else is already using that port. "
            "Set api.port to a free one, or stop the other service."
        )
    finally:
        probe.close()
    return None


def runServe(settings: Settings) -> int:
    """Serve the assistant over HTTP and WebSockets.

    Not an async function: uvicorn owns the event loop, and starting one inside
    another is what the other modes do only because they own theirs.
    """
    try:
        import uvicorn
    except ImportError:
        logger.error(
            'The API requires uvicorn and FastAPI.\nInstall with: pip install -e ".[api]"'
        )
        return 2

    from app.api.app import createApp

    # Checked before anything is loaded. uvicorn binds after startup, so
    # without this a busy port is only discovered once both models are
    # resident --- around eighty seconds of waiting for an error available
    # immediately.
    conflict = describePortConflict(settings.api.host, settings.api.port)
    if conflict is not None:
        logger.error("%s", conflict)
        return 2

    if settings.api.authToken is None:
        logger.info(
            "Binding to %s with no token; only this machine can connect",
            settings.api.host,
        )

    logger.info(
        "%s serving on http://%s:%d", settings.assistant.name, settings.api.host, settings.api.port
    )

    uvicorn.run(
        createApp(settings),
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.logLevel.lower(),
        access_log=False,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = parseArguments(argv)

    if arguments.listDevices or arguments.listInputDevices:
        configureLogging("INFO")
        try:
            lister = listInputDevices if arguments.listInputDevices else listOutputDevices
            for description in lister():
                print(description)
        except (AudioOutputError, AudioInputError) as error:
            logger.error("%s", error)
            return 2
        return 0

    if arguments.transcribe is not None:
        arguments.mode = "listen"

    try:
        settings = applyOverrides(loadSettings(arguments.config), arguments)
    except ConfigurationError as error:
        configureLogging("INFO")
        logger.error("%s", error)
        return 2

    configureLogging(arguments.logLevel or settings.logLevel)

    try:
        if arguments.mode == "client":
            return asyncio.run(runClient(settings, arguments.send))
        if arguments.mode == "serve":
            return runServe(settings)
        if arguments.mode == "wake":
            return asyncio.run(runWake(settings))
        if arguments.mode == "converse":
            return asyncio.run(runConverse(settings, not arguments.noMicrophone))
        if arguments.mode == "listen":
            return asyncio.run(runListen(settings, arguments.transcribe))
        return asyncio.run(runSpeak(settings, arguments.say))
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down")
        return 0
    except RUNTIME_ERRORS as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
