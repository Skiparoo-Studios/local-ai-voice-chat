"""Internal event bus and event definitions."""

from app.events.eventBus import EventBus, Subscription
from app.events.events import (
    AssistantError,
    AssistantState,
    AssistantStateChanged,
    AudioDetected,
    Event,
    IntentDetected,
    ResponseGenerated,
    SpeechEnded,
    SpeechGenerationCompleted,
    SpeechGenerationStarted,
    SpeechStarted,
    ToolExecutionCompleted,
    ToolExecutionStarted,
    TranscriptionCompleted,
    TranscriptionStarted,
    WakeWordDetected,
)

__all__ = [
    "AssistantError",
    "AssistantState",
    "AssistantStateChanged",
    "AudioDetected",
    "Event",
    "EventBus",
    "IntentDetected",
    "ResponseGenerated",
    "SpeechEnded",
    "SpeechGenerationCompleted",
    "SpeechGenerationStarted",
    "SpeechStarted",
    "Subscription",
    "ToolExecutionCompleted",
    "ToolExecutionStarted",
    "TranscriptionCompleted",
    "TranscriptionStarted",
    "WakeWordDetected",
]
