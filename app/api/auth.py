"""Authentication for the API.

§17 requires authentication before remote clients are allowed. The
configuration already refuses to bind to a non-loopback address without a
token, so this enforces the other half: when a token is configured, everything
except the health check must present it.

Comparison is constant-time. Timing differences on a token check are a real
leak, and the cost of avoiding them is nothing.
"""

from __future__ import annotations

import hmac
import logging

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)

# Checked before authentication so that monitoring does not need a credential,
# and so the browser client can be opened before a token has been entered. The
# page is static and holds nothing; every request it makes carries the token.
UNAUTHENTICATED_PATHS = frozenset({"/health", "/"})

BEARER_PREFIX = "bearer "


class ApiAuthenticator:
    """Checks the bearer token on incoming requests."""

    def __init__(self, token: str | None) -> None:
        self._token = token or None

    @property
    def required(self) -> bool:
        return self._token is not None

    def isAuthorised(self, header: str | None) -> bool:
        """Whether an Authorization header carries the configured token."""
        if self._token is None:
            return True
        if not header:
            return False

        value = header.strip()
        if value.lower().startswith(BEARER_PREFIX):
            value = value[len(BEARER_PREFIX) :].strip()

        return hmac.compare_digest(value, self._token)

    def checkRequest(self, request: Request) -> None:
        """Raise 401 unless the request is authorised."""
        if request.url.path in UNAUTHENTICATED_PATHS:
            return
        if self.isAuthorised(request.headers.get("authorization")):
            return

        logger.warning(
            "Rejected unauthenticated %s %s from %s",
            request.method,
            request.url.path,
            request.client.host if request.client else "unknown",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid bearer token is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )


__all__ = ["UNAUTHENTICATED_PATHS", "ApiAuthenticator"]
