"""Configuration schema.

Every section forbids unknown keys so that a typo in ``settings.json`` fails
loudly at startup rather than silently falling back to a default.
"""

from __future__ import annotations

import ipaddress
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

DeviceChoice = Literal["auto", "cpu", "cuda"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Section(BaseModel):
    """Base for configuration sections."""

    model_config = ConfigDict(extra="forbid")


class AssistantSection(Section):
    name: str = "Assistant"
    # rules: deterministic replies, no model. llm: the configured language
    # model. echo: repeats what was said.
    handler: Literal["rules", "llm", "echo", "router"] = "rules"
    # Whether the router may offer tools to the language model (Layer 2 of
    # brief section 12). Deterministic commands work regardless.
    useLlmForTools: bool = True
    # Conversation history is bounded because it is sent to a model with a
    # finite context window.
    maxHistoryTurns: int = Field(default=20, gt=0)
    systemPrompt: str = ""
    # Speak each sentence as it is generated rather than waiting for the whole
    # reply. Synthesis is the largest part of round-trip latency, so this is
    # the main lever on how responsive the assistant feels.
    streaming: bool = True
    # The first synthesis after loading costs about three times the warm rate,
    # so pay it at startup rather than on the user's first turn.
    warmUpOnStart: bool = True


class ToddlerSection(Section):
    """Conversation with a small child, and the questions it asks."""

    # Start in toddler mode. --toddler sets this; the spoken phrases below
    # switch it either way once running.
    enabled: bool = False
    questionsFile: Path = Path("config/toddler.json")
    # Whether "start toddler mode" and "end toddler mode" switch modes. Left on
    # by default because the mode is no use if only the person at the keyboard
    # can reach it.
    allowModePhrases: bool = True
    # Ask the child their name on entering, when the questions file does not
    # already name one. Every question mentioning {name} waits until it knows.
    askForName: bool = True
    # Seconds of silence before the assistant speaks unprompted: revealing the
    # answer to a question nobody attempted, or offering a different one. A
    # small child needs longer than an adult to gather a reply. Zero turns it
    # off. Hands-free listening only --- there is no silence to measure when
    # someone has to press Enter.
    promptAfterSeconds: float = Field(default=7.0, ge=0.0)
    # Consecutive prompts spoken into a silence before the assistant gives up
    # and goes quiet. A child who has wandered off has genuinely finished, and
    # an assistant still asking questions of an empty room is worse company
    # than one that stops. Anything the child says starts the count again.
    giveUpAfterPrompts: int = Field(default=3, gt=0)


class BooksSection(Section):
    """Reading books aloud, from a library of recordings and text files."""

    enabled: bool = True
    # An audiobook is a folder of numbered files; a
    # text book is a .txt. Both are read, audiobooks first, because a
    # professional narrator beats the assistant reading aloud.
    audioDirectory: Path = Path("books/audio")
    textDirectory: Path = Path("books/text")
    # Seconds of audio decoded and written to the speaker at a time. Small
    # enough that "stop" is acted on almost at once, since a block already
    # given to the device has to finish.
    blockSeconds: float = Field(default=5.0, gt=0)
    # How loudly a book plays, as a multiplier on the samples. Below 1 makes
    # room for the microphone to hear "stop audiobook" over it, which is the
    # practical answer to one speaker and one microphone in a room --- and
    # better than reaching for the system mixer every time.
    volume: float = Field(default=1.0, gt=0.0, le=1.0)
    # Where each book was left off, so that "carry on" survives a restart.
    # State the assistant maintains rather than configuration, which is why it
    # is not in this file.
    bookmarksFile: Path = Path("outputs/bookmarks.json")


class SentrySection(Section):
    """Standing down on "goodbye" until greeted with "hello"."""

    # With no wake word, every conversation in the room reaches the recogniser.
    # This is the off switch that does not involve closing anything.
    enabled: bool = True
    # Start dismissed, so nothing is answered until somebody says hello.
    dormant: bool = False
    # Say something on waking. Off makes it wake silently.
    greetOnWaking: bool = True


class SpeechToTextSection(Section):
    provider: str = "faster-whisper"
    model: str = "small"
    device: DeviceChoice = "auto"
    computeType: str = "auto"
    language: str = "en"
    beamSize: int = Field(default=5, gt=0)
    # Whisper invents plausible text during silence. This is the model's own
    # filter, not the VAD abstraction that arrives in Stage 6.
    vadFilter: bool = True
    # How long a single push-to-talk capture may run before stopping itself.
    maximumRecordingSeconds: float = Field(default=30.0, gt=0)


class TextToSpeechSection(Section):
    provider: str = "xtts"
    # Provider-specific, and not every provider has one. XTTS names a coqui
    # model here; Piper does not use this field at all, because a Piper voice
    # IS its model and is named in 'voice' below. Leaving an XTTS model name in
    # place while running Piper is harmless but reads like a mistake --- empty
    # is clearer, and XTTS falls back to its own default when it is.
    model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    # XTTS: a folder or file under paths.voices holding your own recordings.
    # Piper: a published voice name such as "en_GB-cori-medium", optionally
    # with a speaker index for multi-speaker voices ("...high#42").
    voice: str = "default"
    language: str = "en"
    device: DeviceChoice = "auto"
    useEmbeddingCache: bool = True
    # Decoder steps per streamed piece of audio. Smaller emits sound sooner
    # but costs more in total; 0 disables streaming and waits for the whole
    # utterance. Below about 5, generation approaches real time and a long
    # reply risks stuttering.
    streamChunkSize: int = Field(default=10, ge=0)
    # Piper only: fetch a named voice from its published repository if it is
    # not already in paths.models. Turn off on a machine that should never
    # reach the network, and download voices by hand instead.
    allowVoiceDownload: bool = True
    # Piper only: how fast it speaks, as a multiplier on the duration of the
    # audio. Above 1 is slower, which is worth having for a small child. 0
    # leaves the voice at its trained rate.
    speechRate: float = Field(default=0.0, ge=0.0, le=3.0)


class LlmSection(Section):
    provider: str = "ollama"
    model: str = ""
    baseUrl: str = "http://localhost:11434"
    apiKey: SecretStr | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    # None lets the model decide. The system prompt asks for brevity, which
    # works better than a hard cut mid-sentence.
    maxTokens: int | None = Field(default=None, gt=0)
    timeoutSeconds: float = Field(default=120.0, gt=0)
    # Ollama only: how long to keep the model resident between requests.
    # Ollama's own default is five minutes, which means an assistant spoken to
    # every ten minutes reloads the model every single time --- 3.3 seconds
    # against 0.2 for a warm generation. Thirty minutes keeps ordinary use
    # warm while still releasing the memory on a shared machine. Use "-1" to
    # keep it resident indefinitely, which is right for a dedicated one.
    keepAlive: str = "30m"


class AutomationSection(Section):
    provider: str = "home-assistant"
    baseUrl: str = ""
    accessToken: SecretStr | None = None
    timeoutSeconds: float = Field(default=15.0, gt=0)
    # The allow-list required by brief section 17. Wildcards are permitted, so
    # "light.*" enables every light action. Locks, covers and alarms are
    # deliberately absent: enabling those should be a decision, not a default.
    allowedActions: tuple[str, ...] = (
        "light.turnOn",
        "light.turnOff",
        "light.toggle",
        "switch.turnOn",
        "switch.turnOff",
        "switch.toggle",
        "fan.turnOn",
        "fan.turnOff",
    )
    # A global stop on anything that changes the physical environment.
    # Read-only tools continue to work.
    allowStateChanges: bool = True


class AudioSection(Section):
    # A number selects a device by index; a name selects by substring match.
    # Environment variables and JSON both deliver "2" as a string, which
    # PortAudio would otherwise read as a name and match the wrong device, so
    # digit-only values are converted below.
    inputDevice: int | str | None = None
    outputDevice: int | str | None = None

    @field_validator("inputDevice", "outputDevice", mode="before")
    @classmethod
    def numericStringsAreIndices(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value.strip())
        return value
    sampleRate: int = Field(default=16000, gt=0)
    # "file" writes utterances to paths.outputs instead of playing them, which
    # is what headless machines and CI need.
    outputBackend: Literal["sounddevice", "file"] = "sounddevice"
    # How long to ignore the microphone after the assistant stops speaking.
    # Without it a hands-free assistant hears the end of its own reply and
    # answers itself: PortAudio's write() returns with about 180 ms still in
    # the device buffer, and a room adds reverberation on top. Raise it if the
    # speakers are loud or close to the microphone; headphones remove the
    # problem altogether. Zero disables the guard.
    echoGuardSeconds: float = Field(default=0.4, ge=0.0)
    # How long the microphone may deliver nothing before it is treated as gone
    # and the process stops, so that a supervisor can restart it. Silence still
    # arrives as frames, so a quiet room never triggers this and neither does a
    # mute switch --- only a device that has actually left the bus. Zero waits
    # indefinitely, which is the old behaviour.
    captureStallSeconds: float = Field(default=5.0, ge=0.0)


class ApiSection(Section):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, gt=0, lt=65536)
    authToken: SecretStr | None = None

    @model_validator(mode="after")
    def requireAuthWhenNotLoopback(self) -> ApiSection:
        """Refuse to expose the API beyond loopback without a token.

        The assistant can act on the physical environment, so this invariant is
        enforced by the configuration itself rather than left to the API layer.
        """
        if self.authToken is not None:
            return self

        host = self.host.strip()
        if host in {"localhost", ""}:
            return self

        try:
            address = ipaddress.ip_address(host)
        except ValueError as error:
            raise ValueError(
                f"api.host {host!r} is not a recognised address; set api.authToken "
                "(or VOICE_API__AUTHTOKEN) if you intend to accept remote clients"
            ) from error

        if not address.is_loopback:
            raise ValueError(
                f"api.host {host!r} accepts non-local connections, so api.authToken "
                "must be set. Provide it via the VOICE_API__AUTHTOKEN environment "
                "variable rather than in settings.json."
            )
        return self


class VadSection(Section):
    # silero distinguishes speech from other sound; energy is a dependency-free
    # loudness threshold, predictable but unable to tell speech from a door.
    provider: Literal["silero", "energy"] = "silero"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    energyThreshold: float = Field(default=0.02, ge=0.0, le=1.0)
    # Consecutive speech frames before capture starts, so one noisy frame does
    # not begin an utterance.
    startFrames: int = Field(default=2, gt=0)
    # Silence before an utterance is considered finished. 25 frames is about
    # 0.8 seconds: long enough to pause mid-sentence without being cut off.
    silenceFrames: int = Field(default=25, gt=0)
    # Audio kept from before speech was detected, so the first word survives.
    prerollFrames: int = Field(default=10, ge=0)
    maximumSeconds: float = Field(default=15.0, gt=0)
    minimumSeconds: float = Field(default=0.3, ge=0.0)


class WakeWordSection(Section):
    provider: Literal["openwakeword", "always-awake", "manual"] = "openwakeword"
    # Pretrained openWakeWord models: alexa, hey_jarvis, hey_mycroft,
    # hey_rhasspy, timer, weather. A path to a .onnx file also works.
    word: str = "hey_jarvis"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    # One spoken wake word spans many frames; without a refractory period it
    # fires on each of them.
    refractorySeconds: float = Field(default=2.0, ge=0.0)
    # Interrupt the assistant when the wake word is heard while it is speaking.
    allowBargeIn: bool = True
    # Say a short word on waking, before the real reply.
    acknowledge: bool = False


class ClientSection(Section):
    """Settings for a machine that uses a remote assistant."""

    serverUrl: str = "http://127.0.0.1:8000"
    token: SecretStr | None = None


class PathsSection(Section):
    voices: Path = Path("voices")
    models: Path = Path("models")
    outputs: Path = Path("outputs")
