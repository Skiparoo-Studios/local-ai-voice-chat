"""REST endpoints.

The set described in §9 of the brief. Handlers stay thin: they validate, call
:class:`~app.api.service.AssistantService`, and shape the result. Anything that
looks like assistant logic in here would be logic the command-line client
could not reach.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request, Response, UploadFile, status

from app import __version__
from app.api.models import (
    AutomationRequest,
    AutomationResponse,
    DeviceResponse,
    HealthResponse,
    MessageRequest,
    MessageResponse,
    SynthesiseRequest,
    TimingResponse,
    TranscriptionResponse,
)
from app.assistant.runtime import TurnResult
from app.audio.audioBuffer import AudioBuffer
from app.automation.automationProvider import AutomationError
from app.speech.speechToText import TranscriptionError
from app.speech.textToSpeech import SpeechSynthesisError
from app.tools.base import ToolError

if TYPE_CHECKING:
    from app.api.service import AssistantService

logger = logging.getLogger(__name__)

router = APIRouter()

# Enough for a long utterance, small enough that a malformed request cannot
# exhaust memory.
MAXIMUM_UPLOAD_BYTES = 25 * 1024 * 1024


def getService(request: Request) -> AssistantService:
    """The service built at startup and shared by every request."""
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The assistant is not ready yet.",
        )
    return service


WEB_PAGE = Path(__file__).parent / "web" / "index.html"


@router.get("/", include_in_schema=False)
async def webClient() -> Response:
    """The browser client.

    Unauthenticated because it is a static page holding no data and no
    credentials: the token is typed in by whoever opens it, and every request
    the page makes carries it.
    """
    try:
        page = WEB_PAGE.read_text(encoding="utf-8")
    except OSError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The web client is not installed alongside the API.",
        ) from error

    return Response(
        content=page,
        media_type="text/html; charset=utf-8",
        # A local tool that changes with the code it ships beside.
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness, deliberately unauthenticated so monitoring needs no secret."""
    return HealthResponse(status="ok", version=__version__)


@router.get("/status")
async def readStatus(request: Request) -> dict:
    """What the assistant is doing, and what it is running."""
    return getService(request).status().toDict()


@router.post("/assistant/message", response_model=MessageResponse)
async def sendMessage(request: Request, body: MessageRequest) -> MessageResponse:
    """Send text and get the reply."""
    service = getService(request)

    try:
        result = await service.handleText(body.text, speak=body.speak)
    except (SpeechSynthesisError, TranscriptionError, ToolError) as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)
        ) from error

    return _messageResponse(service, result, includeAudio=body.includeAudio)


@router.post("/speech/transcribe", response_model=TranscriptionResponse)
async def transcribe(request: Request, audio: UploadFile) -> TranscriptionResponse:
    """Transcribe an uploaded WAV file, without replying to it."""
    service = getService(request)
    payload = await _readUpload(audio)

    try:
        buffer = AudioBuffer.fromWavBytes(payload)
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="That did not parse as a WAV file.",
        ) from error

    try:
        transcript = await service.transcribe(buffer)
    except TranscriptionError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)
        ) from error

    return TranscriptionResponse(
        text=transcript.text,
        language=transcript.language,
        audioSeconds=round(transcript.audioSeconds, 3),
        transcriptionSeconds=round(transcript.transcriptionSeconds, 3),
    )


@router.post("/speech/synthesise")
async def synthesise(request: Request, body: SynthesiseRequest) -> Response:
    """Generate speech and return it as a WAV file."""
    service = getService(request)

    try:
        audio = await service.synthesise(
            body.text, voice=body.voice, language=body.language
        )
    except SpeechSynthesisError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)
        ) from error

    return Response(
        content=audio.toWavBytes(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )


@router.get("/automation/devices", response_model=list[DeviceResponse])
async def listDevices(request: Request) -> list[DeviceResponse]:
    """Devices the assistant can act on."""
    tool = _automationTool(getService(request))

    try:
        devices = await tool.devices(refresh=True)
    except AutomationError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)
        ) from error

    return [
        DeviceResponse(
            deviceId=device.deviceId,
            name=device.name,
            describedName=device.describedName,
            domain=device.domain,
            area=device.area,
            state=device.state,
        )
        for device in devices
    ]


@router.post("/automation/execute", response_model=AutomationResponse)
async def executeAutomation(
    request: Request, body: AutomationRequest
) -> AutomationResponse:
    """Perform an automation action.

    The same allow-list applies as when the assistant decides for itself. An
    API client is not more trusted than the language model.
    """
    service = getService(request)
    registry = _toolRegistry(service)

    try:
        result = await registry.invoke(
            "homeAutomation",
            {"action": body.action, "target": body.target, "all": body.all},
        )
    except ToolError as error:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=str(error)
        ) from error

    return AutomationResponse(
        succeeded=result.succeeded, message=result.message, details=result.details
    )


# --- Helpers -------------------------------------------------------------


async def _readUpload(upload: UploadFile) -> bytes:
    payload = await upload.read()
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No audio was uploaded."
        )
    if len(payload) > MAXIMUM_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio must be under {MAXIMUM_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    return payload


def _messageResponse(
    service: AssistantService, result: TurnResult, *, includeAudio: bool
) -> MessageResponse:
    timing = result.timing
    audio = None
    if includeAudio and result.audio is not None:
        audio = base64.b64encode(result.audio.toWavBytes()).decode("ascii")

    return MessageResponse(
        reply=result.assistantText,
        sessionId=service.session.sessionId,
        usedLlm=bool(result.response and result.response.usedLlm),
        handledBy=result.response.handledBy if result.response else "",
        timings=TimingResponse(
            transcriptionSeconds=round(timing.transcriptionSeconds, 3),
            responseSeconds=round(timing.responseSeconds, 3),
            synthesisSeconds=round(timing.synthesisSeconds, 3),
            firstAudioSeconds=round(timing.firstAudioSeconds, 3),
            streamed=timing.streamed,
        ),
        audio=audio,
    )


def _toolRegistry(service: AssistantService):  # type: ignore[no-untyped-def]
    registry = getattr(service.runtime.handler, "registry", None)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Home automation is not enabled. "
                "Set assistant.handler to 'router' to turn it on."
            ),
        )
    return registry


def _automationTool(service: AssistantService):  # type: ignore[no-untyped-def]
    registry = _toolRegistry(service)
    if not registry.has("homeAutomation"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No home automation tool is registered.",
        )
    return registry.get("homeAutomation")


__all__ = ["getService", "router"]
