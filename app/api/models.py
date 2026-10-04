"""Request and response shapes for the API.

Every request body is validated here before reaching the assistant, which is
the same rule §17 applies to tool arguments: nothing crossing a trust boundary
is taken on faith.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    """Base for API payloads, rejecting unknown fields.

    A request with a misspelled field is a client bug, and failing it is more
    useful than silently ignoring the part that was meant to matter.
    """

    model_config = ConfigDict(extra="forbid")


# --- Requests ------------------------------------------------------------


class MessageRequest(ApiModel):
    """POST /assistant/message"""

    text: str = Field(min_length=1, max_length=4000)
    # Whether the assistant should also say the reply on its own speaker.
    speak: bool = False
    # Whether to return the generated audio with the reply.
    includeAudio: bool = False


class SynthesiseRequest(ApiModel):
    """POST /speech/synthesise"""

    text: str = Field(min_length=1, max_length=4000)
    voice: str | None = None
    language: str | None = None


class AutomationRequest(ApiModel):
    """POST /automation/execute"""

    action: str = Field(min_length=1, max_length=100)
    target: str = Field(min_length=1, max_length=200)
    all: bool = False


# --- Responses -----------------------------------------------------------


class HealthResponse(ApiModel):
    status: str
    version: str


class TimingResponse(ApiModel):
    transcriptionSeconds: float
    responseSeconds: float
    synthesisSeconds: float
    firstAudioSeconds: float
    streamed: bool


class MessageResponse(ApiModel):
    reply: str
    sessionId: str
    usedLlm: bool
    handledBy: str
    timings: TimingResponse
    # Base64 WAV, present only when the client asked for it.
    audio: str | None = None


class TranscriptionResponse(ApiModel):
    text: str
    language: str | None
    audioSeconds: float
    transcriptionSeconds: float


class DeviceResponse(ApiModel):
    deviceId: str
    name: str
    # How a person would refer to it. Derived on the server so that every
    # client does not have to reimplement the rule that a name already
    # carrying its area should not be given it twice.
    describedName: str
    domain: str
    area: str | None
    state: str | None


class AutomationResponse(ApiModel):
    succeeded: bool
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(ApiModel):
    error: str
    detail: str = ""


__all__ = [
    "AutomationRequest",
    "AutomationResponse",
    "DeviceResponse",
    "ErrorResponse",
    "HealthResponse",
    "MessageRequest",
    "MessageResponse",
    "SynthesiseRequest",
    "TimingResponse",
    "TranscriptionResponse",
]
