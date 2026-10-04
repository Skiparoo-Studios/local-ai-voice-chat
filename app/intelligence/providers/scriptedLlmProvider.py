"""A language model provider that returns prepared text.

Completes the set alongside the tone and scripted-speech providers: the whole
conversation loop, including the streaming path, can run with no model service
and no network. It also yields word by word with an optional delay, which is
what makes streaming behaviour observable in tests.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, ClassVar

from app.intelligence.llmProvider import LlmProvider, LlmResponse, Message, ToolCall

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_REPLIES = (
    "I've turned the kitchen light off. Is there anything else you need?",
    "It's currently quite mild outside. The forecast suggests rain this evening.",
    "That timer is set for ten minutes. I'll let you know when it finishes.",
)


class ScriptedLlmProvider(LlmProvider):
    """Returns configured replies instead of generating."""

    name: ClassVar[str] = "scripted"

    def __init__(
        self,
        replies: tuple[str, ...] = DEFAULT_REPLIES,
        *,
        fragmentDelaySeconds: float = 0.0,
        toolCalls: tuple[ToolCall, ...] = (),
    ) -> None:
        self._replies = replies or DEFAULT_REPLIES
        self._fragmentDelaySeconds = fragmentDelaySeconds
        # Returned in place of text when the caller offers tools, so the
        # tool-calling path can be tested without a model that supports it.
        self._toolCalls = toolCalls
        self._index = 0
        self._toolIndex = 0

    @classmethod
    def fromSettings(cls, settings: Settings) -> ScriptedLlmProvider:
        return cls()

    def _nextReply(self) -> str:
        reply = self._replies[self._index % len(self._replies)]
        self._index += 1
        return reply

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> LlmResponse:
        if tools and self._toolCalls:
            call = self._toolCalls[self._toolIndex % len(self._toolCalls)]
            self._toolIndex += 1
            return LlmResponse(text="", model="scripted", toolCalls=(call,))

        started = time.perf_counter()
        text = self._nextReply()
        words = text.split()

        # A real model takes as long to produce a complete reply as it does to
        # stream one. Returning instantly here would make any streaming
        # comparison measure against a baseline that does not exist.
        if self._fragmentDelaySeconds:
            await asyncio.sleep(self._fragmentDelaySeconds * len(words))

        return LlmResponse(
            text=text,
            model="scripted",
            completionTokens=len(words),
            durationSeconds=time.perf_counter() - started,
        )

    async def generateStream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Yield the reply word by word, as a real model would."""
        words = self._nextReply().split(" ")
        for index, word in enumerate(words):
            if self._fragmentDelaySeconds:
                await asyncio.sleep(self._fragmentDelaySeconds)
            yield word if index == 0 else f" {word}"

    async def isAvailable(self) -> bool:
        return True

    async def listModels(self) -> list[str]:
        return ["scripted"]

    def describe(self) -> str:
        return f"scripted ({len(self._replies)} replies)"
