"""Assistant events.

These correspond to §8 of the architecture brief. Each event carries an
``eventType`` string used when the event is serialised to WebSocket clients,
matching the wire format described in §9 (``assistant.state`` and friends).

Events describe things that have happened. They carry no behaviour and no
references to providers, so any part of the system can observe them without
depending on the part that raised them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any, ClassVar


class AssistantState(StrEnum):
    """States a client may display."""

    idle = "idle"
    listening = "listening"
    thinking = "thinking"
    controllingDevice = "controlling-device"
    speaking = "speaking"


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """Base event.

    Frozen so that a handler cannot mutate an event other handlers will see.
    """

    eventType: ClassVar[str] = "assistant.event"

    sessionId: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def toDict(self) -> dict[str, Any]:
        """Serialise for transport to clients."""
        payload: dict[str, Any] = {"type": self.eventType}
        for key, value in asdict(self).items():
            payload[key] = _toJsonValue(value)
        return payload


def _toJsonValue(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _toJsonValue(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_toJsonValue(item) for item in value]
    return value


# --- Audio capture -------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioDetected(Event):
    eventType: ClassVar[str] = "assistant.audioDetected"

    durationSeconds: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class WakeWordDetected(Event):
    eventType: ClassVar[str] = "assistant.wakeWordDetected"

    word: str = ""
    confidence: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class SpeechStarted(Event):
    eventType: ClassVar[str] = "assistant.speechStarted"


@dataclass(frozen=True, slots=True, kw_only=True)
class SpeechEnded(Event):
    eventType: ClassVar[str] = "assistant.speechEnded"

    durationSeconds: float = 0.0


# --- Transcription -------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptionStarted(Event):
    eventType: ClassVar[str] = "assistant.transcriptionStarted"


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptionCompleted(Event):
    eventType: ClassVar[str] = "assistant.transcription"

    text: str = ""
    language: str | None = None
    durationSeconds: float = 0.0


# --- Intent and tools ----------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class IntentDetected(Event):
    eventType: ClassVar[str] = "assistant.intentDetected"

    intent: str = ""
    confidence: float = 0.0
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolExecutionStarted(Event):
    eventType: ClassVar[str] = "assistant.toolExecutionStarted"

    tool: str = ""
    action: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolExecutionCompleted(Event):
    eventType: ClassVar[str] = "assistant.toolExecutionCompleted"

    tool: str = ""
    action: str = ""
    succeeded: bool = True
    result: dict[str, Any] = field(default_factory=dict)
    durationSeconds: float = 0.0


# --- Response and speech generation --------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ResponseGenerated(Event):
    eventType: ClassVar[str] = "assistant.response"

    text: str = ""
    usedLlm: bool = False
    durationSeconds: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class SpeechGenerationStarted(Event):
    eventType: ClassVar[str] = "assistant.speechGenerationStarted"

    text: str = ""
    voice: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class SpeechGenerationCompleted(Event):
    eventType: ClassVar[str] = "assistant.speechGenerationCompleted"

    voice: str = ""
    audioSeconds: float = 0.0
    durationSeconds: float = 0.0


# --- State and errors ----------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class AssistantStateChanged(Event):
    eventType: ClassVar[str] = "assistant.state"

    state: AssistantState = AssistantState.idle
    previousState: AssistantState | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class AssistantError(Event):
    eventType: ClassVar[str] = "assistant.error"

    message: str = ""
    component: str = ""
    recoverable: bool = True
