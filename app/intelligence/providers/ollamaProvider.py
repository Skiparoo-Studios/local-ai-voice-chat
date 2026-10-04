"""Ollama, over its HTTP API.

No Ollama SDK is used. The API is small and stable, and a direct HTTP client
keeps the dependency surface to httpx, which the OpenAI-compatible provider
needs anyway.
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

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(LlmProvider):
    """Talks to a local Ollama service."""

    name: ClassVar[str] = "ollama"

    def __init__(
        self,
        *,
        model: str,
        baseUrl: str = DEFAULT_BASE_URL,
        temperature: float = 0.7,
        maxTokens: int | None = None,
        timeoutSeconds: float = 120.0,
        keepAlive: str | None = None,
        transport: Any = None,
    ) -> None:
        if not model:
            raise LlmError(
                "No language model configured.\n"
                "Set llm.model in config/settings.json, or:\n"
                '    $env:VOICE_LLM__MODEL = "qwen2.5:3b"\n'
                "Run `ollama list` to see what is installed, or `ollama pull "
                "qwen2.5:3b` to fetch one.\n"
                "Alternatively use --handler rules, which needs no model."
            )
        self._model = model
        self._baseUrl = baseUrl.rstrip("/")
        self._temperature = temperature
        self._maxTokens = maxTokens
        self._timeoutSeconds = timeoutSeconds
        self._keepAlive = keepAlive
        self._transport = transport
        self._client: Any = None

    @classmethod
    def fromSettings(cls, settings: Settings) -> OllamaProvider:
        section = settings.llm
        return cls(
            model=section.model,
            baseUrl=section.baseUrl or DEFAULT_BASE_URL,
            temperature=section.temperature,
            maxTokens=section.maxTokens,
            timeoutSeconds=section.timeoutSeconds,
            keepAlive=section.keepAlive or None,
        )

    # --- Lifecycle -------------------------------------------------------

    async def load(self) -> None:
        self._ensureClient()

    def _ensureClient(self) -> Any:
        if self._client is None:
            httpx = _importHttpx()
            self._client = httpx.AsyncClient(
                base_url=self._baseUrl,
                timeout=self._timeoutSeconds,
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
        options: dict[str, Any] = {"temperature": self._temperature}
        if self._maxTokens:
            options["num_predict"] = self._maxTokens

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
            "options": options,
        }
        if self._keepAlive:
            payload["keep_alive"] = self._keepAlive
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
                "/api/chat", json=self._payload(messages, False, tools)
            )
            response.raise_for_status()
            body = response.json()
        except Exception as error:
            raise _describeFailure(error, self._baseUrl, self._model) from error

        message = body.get("message") or {}
        return LlmResponse(
            text=message.get("content", ""),
            model=body.get("model", self._model),
            promptTokens=int(body.get("prompt_eval_count") or 0),
            completionTokens=int(body.get("eval_count") or 0),
            durationSeconds=time.perf_counter() - started,
            toolCalls=_parseToolCalls(message.get("tool_calls")),
        )

    async def generateStream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Yield fragments as Ollama produces them.

        The service returns newline-delimited JSON, one object per fragment.
        """
        client = self._ensureClient()

        try:
            async with client.stream(
                "POST", "/api/chat", json=self._payload(messages, True)
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    fragment = _fragmentFromLine(line)
                    if fragment:
                        yield fragment
        except Exception as error:
            raise _describeFailure(error, self._baseUrl, self._model) from error

    # --- Introspection ---------------------------------------------------

    async def isAvailable(self) -> bool:
        try:
            client = self._ensureClient()
            response = await client.get("/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception as error:  # noqa: BLE001 - availability is a question, not a failure
            logger.debug("Ollama not reachable: %s", error)
            return False

    async def listModels(self) -> list[str]:
        try:
            client = self._ensureClient()
            response = await client.get("/api/tags", timeout=10.0)
            response.raise_for_status()
            models = response.json().get("models") or []
        except Exception as error:
            raise _describeFailure(error, self._baseUrl, self._model) from error
        return sorted(entry.get("name", "") for entry in models if entry.get("name"))

    def describe(self) -> str:
        return f"ollama [{self._model}] at {self._baseUrl}"


def _parseToolCalls(raw: Any) -> tuple[ToolCall, ...]:
    """Read tool calls from a response message.

    Ollama sends arguments as an object; some builds send a JSON string. Both
    are accepted, and anything unreadable is dropped rather than guessed at.
    """
    if not raw:
        return ()

    calls: list[ToolCall] = []
    for entry in raw:
        function = (entry or {}).get("function") or {}
        name = function.get("name")
        if not name:
            continue

        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning("Discarding tool call %r with unparseable arguments", name)
                continue
        if not isinstance(arguments, dict):
            arguments = {}

        calls.append(ToolCall(name=name, arguments=arguments))
    return tuple(calls)


def _fragmentFromLine(line: str) -> str:
    """Extract the text fragment from one NDJSON line, if it carries one."""
    line = line.strip()
    if not line:
        return ""

    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        logger.debug("Ignoring unparseable stream line: %.80s", line)
        return ""

    if payload.get("error"):
        raise LlmError(f"Ollama reported an error: {payload['error']}")

    return (payload.get("message") or {}).get("content", "") or ""


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

    if status == 404:
        return LlmUnavailableError(
            f"Ollama does not have the model {model!r}.\n"
            f"    ollama pull {model}\n"
            "Or set llm.model to one that is installed."
        )

    if isinstance(error, LlmError):
        return error

    if "connect" in text or "refused" in text or "timed out" in text:
        return LlmUnavailableError(
            f"Could not reach Ollama at {baseUrl}.\n"
            "Check the service is running (`ollama serve`), or set llm.baseUrl.\n\n"
            f"Original error: {error}"
        )

    return LlmError(f"Ollama request failed: {error}")


__all__ = ["OllamaProvider"]
