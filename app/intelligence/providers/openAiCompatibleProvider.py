"""Any service speaking the OpenAI chat-completions API.

That includes llama.cpp's server, LM Studio, vLLM, text-generation-webui and
OpenAI itself. The brief permits cloud providers as optional plugins, and this
is the one interface that reaches most of them --- but nothing here is required
for the assistant to work offline.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, ClassVar

from app.intelligence.llmProvider import (
    LlmError,
    LlmProvider,
    LlmResponse,
    LlmUnavailableError,
    Message,
    ToolCall,
)

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8080/v1"
STREAM_DONE = "[DONE]"


class OpenAiCompatibleProvider(LlmProvider):
    """Talks to an OpenAI-compatible chat-completions endpoint."""

    name: ClassVar[str] = "openai-compatible"

    def __init__(
        self,
        *,
        model: str,
        baseUrl: str = DEFAULT_BASE_URL,
        apiKey: str | None = None,
        temperature: float = 0.7,
        maxTokens: int | None = None,
        timeoutSeconds: float = 120.0,
        transport: Any = None,
    ) -> None:
        self._model = model
        self._baseUrl = baseUrl.rstrip("/")
        self._apiKey = apiKey
        self._temperature = temperature
        self._maxTokens = maxTokens
        self._timeoutSeconds = timeoutSeconds
        self._transport = transport
        self._client: Any = None

    @classmethod
    def fromSettings(cls, settings: Settings) -> OpenAiCompatibleProvider:
        section = settings.llm
        return cls(
            model=section.model,
            baseUrl=section.baseUrl or DEFAULT_BASE_URL,
            apiKey=section.apiKey.get_secret_value() if section.apiKey else None,
            temperature=section.temperature,
            maxTokens=section.maxTokens,
            timeoutSeconds=section.timeoutSeconds,
        )

    # --- Lifecycle -------------------------------------------------------

    async def load(self) -> None:
        self._ensureClient()

    def _ensureClient(self) -> Any:
        if self._client is None:
            httpx = _importHttpx()
            headers = {"Content-Type": "application/json"}
            if self._apiKey:
                headers["Authorization"] = f"Bearer {self._apiKey}"
            self._client = httpx.AsyncClient(
                base_url=self._baseUrl,
                timeout=self._timeoutSeconds,
                headers=headers,
                transport=self._transport,
            )
        return self._client

    async def unload(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # --- Requests --------------------------------------------------------

    def _payload(
        self,
        messages: list[Message],
        stream: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
            "temperature": self._temperature,
        }
        if self._maxTokens:
            payload["max_tokens"] = self._maxTokens
        if tools:
            payload["tools"] = tools
        return payload

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> LlmResponse:
        client = self._ensureClient()
        started = time.perf_counter()

        try:
            response = await client.post(
                "/chat/completions", json=self._payload(messages, False, tools)
            )
            response.raise_for_status()
            body = response.json()
        except Exception as error:
            raise _describeFailure(error, self._baseUrl, self._model) from error

        choices = body.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        usage = body.get("usage") or {}

        return LlmResponse(
            text=message.get("content") or "",
            model=body.get("model", self._model),
            promptTokens=int(usage.get("prompt_tokens") or 0),
            completionTokens=int(usage.get("completion_tokens") or 0),
            durationSeconds=time.perf_counter() - started,
            toolCalls=_parseToolCalls(message.get("tool_calls")),
        )

    async def generateStream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Yield fragments from a server-sent-events stream."""
        client = self._ensureClient()

        try:
            async with client.stream(
                "POST", "/chat/completions", json=self._payload(messages, True)
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    fragment, finished = _fragmentFromEvent(line)
                    if finished:
                        break
                    if fragment:
                        yield fragment
        except Exception as error:
            raise _describeFailure(error, self._baseUrl, self._model) from error

    # --- Introspection ---------------------------------------------------

    async def isAvailable(self) -> bool:
        try:
            client = self._ensureClient()
            response = await client.get("/models", timeout=5.0)
            return response.status_code < 500
        except Exception as error:  # noqa: BLE001 - availability is a question, not a failure
            logger.debug("OpenAI-compatible endpoint not reachable: %s", error)
            return False

    async def listModels(self) -> list[str]:
        try:
            client = self._ensureClient()
            response = await client.get("/models", timeout=10.0)
            response.raise_for_status()
            entries = response.json().get("data") or []
        except Exception as error:
            raise _describeFailure(error, self._baseUrl, self._model) from error
        return sorted(entry.get("id", "") for entry in entries if entry.get("id"))

    def describe(self) -> str:
        return f"openai-compatible [{self._model}] at {self._baseUrl}"


def _parseToolCalls(raw: Any) -> tuple[ToolCall, ...]:
    """Read tool calls from a response message.

    The OpenAI shape sends arguments as a JSON string. Anything unreadable is
    dropped rather than guessed at.
    """
    if not raw:
        return ()

    calls: list[ToolCall] = []
    for entry in raw:
        function = (entry or {}).get("function") or {}
        name = function.get("name")
        if not name:
            continue

        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                logger.warning("Discarding tool call %r with unparseable arguments", name)
                continue
        if not isinstance(arguments, dict):
            arguments = {}

        calls.append(ToolCall(name=name, arguments=arguments))
    return tuple(calls)


def _fragmentFromEvent(line: str) -> tuple[str, bool]:
    """Extract text from one SSE line. Returns (fragment, streamFinished)."""
    line = line.strip()
    if not line or not line.startswith("data:"):
        return "", False

    data = line[len("data:") :].strip()
    if data == STREAM_DONE:
        return "", True

    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        logger.debug("Ignoring unparseable stream line: %.80s", line)
        return "", False

    if payload.get("error"):
        raise LlmError(f"The endpoint reported an error: {payload['error']}")

    choices = payload.get("choices") or []
    if not choices:
        return "", False

    delta = choices[0].get("delta") or {}
    return delta.get("content") or "", False


def _importHttpx() -> Any:
    try:
        import httpx
    except ImportError as error:
        raise LlmUnavailableError(
            "Language model providers require httpx.\n"
            'Install with: pip install -e ".[llm]"'
        ) from error
    return httpx


def _describeFailure(error: Exception, baseUrl: str, model: str) -> Exception:
    """Translate transport failures into messages that say what to do."""
    text = str(error).lower()
    status = getattr(getattr(error, "response", None), "status_code", None)

    if status in (401, 403):
        return LlmUnavailableError(
            f"The endpoint at {baseUrl} rejected the credentials. "
            "Set llm.apiKey through the VOICE_LLM__APIKEY environment variable."
        )

    if status == 404:
        return LlmUnavailableError(
            f"The endpoint at {baseUrl} has no model {model!r}, or the path is wrong. "
            "llm.baseUrl usually needs to end in /v1."
        )

    if isinstance(error, LlmError):
        return error

    if "connect" in text or "refused" in text or "timed out" in text:
        return LlmUnavailableError(
            f"Could not reach the endpoint at {baseUrl}.\n"
            "Check the service is running, or set llm.baseUrl.\n\n"
            f"Original error: {error}"
        )

    return LlmError(f"Request failed: {error}")


__all__ = ["OpenAiCompatibleProvider"]
