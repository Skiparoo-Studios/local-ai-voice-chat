"""The FastAPI application.

Models load once, at startup, and are shared by every request. The service is
built in the lifespan handler rather than at import time, so that importing
this module does not load two gigabytes of models --- which is what makes the
API testable.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse

from app import __version__
from app.api.auth import ApiAuthenticator
from app.api.routes import router
from app.api.service import AssistantService
from app.api.websocket import ConnectionManager, handleConnection

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

DESCRIPTION = (
    "A local-first voice assistant. Speech recognition, language understanding "
    "and speech synthesis all run on this machine."
)


def createApp(settings: Settings, service: AssistantService | None = None) -> FastAPI:
    """Build the application.

    ``service`` may be supplied already built, which is how tests avoid loading
    real models.
    """
    authenticator = ApiAuthenticator(
        settings.api.authToken.get_secret_value() if settings.api.authToken else None
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        assistant = service or AssistantService(settings)
        application.state.service = assistant
        application.state.connections = ConnectionManager(assistant.bus)

        await assistant.start()
        application.state.connections.start()

        if authenticator.required:
            logger.info("API authentication is enabled")
        else:
            logger.info("No API token configured; only loopback clients can connect")

        try:
            yield
        finally:
            application.state.connections.stop()
            await assistant.stop()

    application = FastAPI(
        title="Local Voice Assistant",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def authenticate(request: Request, callNext):
        """Reject unauthenticated requests, and log the outcome of every one.

        Logging records method, path, status and duration. Bodies are never
        logged: they carry transcripts, replies and, on one endpoint, audio.
        """
        started = time.perf_counter()

        try:
            authenticator.checkRequest(request)
        except Exception as error:  # HTTPException, raised for unauthorised requests
            statusCode = getattr(error, "status_code", 500)
            if statusCode != 401:
                raise
            return JSONResponse(
                status_code=401,
                content={"error": "unauthorised", "detail": getattr(error, "detail", "")},
                headers={"WWW-Authenticate": "Bearer"},
            )

        response = await callNext(request)
        logger.info(
            "%s %s -> %d in %.0fms",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response

    application.include_router(router)

    @application.websocket("/ws/assistant")
    async def assistantSocket(socket: WebSocket) -> None:
        """The event stream, and a way to drive the assistant."""
        if not authenticator.isAuthorised(_socketToken(socket)):
            # 1008 is "policy violation", the closest thing a WebSocket has to
            # a 401. The handshake must be accepted before a close frame can
            # carry a reason the client will see.
            await socket.accept()
            await socket.close(code=1008, reason="A valid bearer token is required.")
            logger.warning("Rejected an unauthenticated WebSocket connection")
            return

        await handleConnection(
            socket, socket.app.state.connections, socket.app.state.service
        )

    return application


def _socketToken(socket: WebSocket) -> str | None:
    """Read the token from a header, or from the query string.

    Browsers cannot set headers on a WebSocket handshake, so the query string
    is the only route available to a web client.
    """
    header = socket.headers.get("authorization")
    if header:
        return header
    token = socket.query_params.get("token")
    return f"Bearer {token}" if token else None


__all__ = ["createApp"]
