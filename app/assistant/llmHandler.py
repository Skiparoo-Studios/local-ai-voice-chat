"""Replying with a language model.

Two things matter here beyond calling the model.

The system prompt is not decoration. A model left to its own habits answers a
spoken question with headings, bullet points and three paragraphs, all of which
sound terrible read aloud. Constraining length and formatting is what makes the
difference between a usable voice assistant and an unusable one.

Second, §17 of the brief requires model output to be treated as untrusted. In
this stage the output is only spoken, so the exposure is small, but the
sanitising happens here rather than being deferred to Stage 5, when the same
output starts selecting tools.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, ClassVar

from app.assistant.conversation import Conversation
from app.assistant.handlers import Response, ResponseHandler
from app.intelligence.llmProvider import LlmProvider

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

# Short enough that warming costs a token or two, long enough to be a real
# request rather than something a service might shortcut.
WARM_UP_PROMPT = "Reply with the single word: ready"

DEFAULT_SYSTEM_PROMPT = (
    "You are {name}, a voice assistant running locally in someone's home. "
    "Your replies are spoken aloud, so keep them short: one or two sentences "
    "unless more is genuinely needed. Write plain prose. Never use markdown, "
    "bullet points, headings, emoji or code blocks, because they cannot be "
    "spoken. Expand abbreviations and symbols into words. If you do not know "
    "something, say so briefly rather than guessing."
)

# Markdown decoration that would otherwise be read out character by character.
MARKDOWN_PATTERNS = (
    (re.compile(r"```[\s\S]*?```"), " "),
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE), ""),
    (re.compile(r"^\s{0,3}[-*+]\s+", re.MULTILINE), ""),
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),
    (re.compile(r"\*([^*]+)\*"), r"\1"),
    (re.compile(r"__([^_]+)__"), r"\1"),
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),
)


class LlmHandler(ResponseHandler):
    """Sends the conversation to a language model and speaks the reply."""

    name: ClassVar[str] = "llm"

    def __init__(
        self,
        provider: LlmProvider,
        *,
        assistantName: str = "Assistant",
        systemPrompt: str | None = None,
        streaming: bool = True,
    ) -> None:
        self._provider = provider
        self._systemPrompt = systemPrompt or DEFAULT_SYSTEM_PROMPT.format(name=assistantName)
        self._streaming = streaming

    @classmethod
    def fromSettings(cls, settings: Settings, provider: LlmProvider) -> LlmHandler:
        return cls(
            provider,
            assistantName=settings.assistant.name,
            systemPrompt=settings.assistant.systemPrompt or None,
            streaming=settings.assistant.streaming,
        )

    @property
    def provider(self) -> LlmProvider:
        return self._provider

    @property
    def supportsStreaming(self) -> bool:
        return self._streaming

    @property
    def usesLlm(self) -> bool:
        return True

    @property
    def languageModel(self) -> LlmProvider:
        return self._provider

    async def load(self) -> None:
        await self._provider.load()

    async def warmUp(self) -> None:
        """Make one tiny request so the model is resident before it is needed.

        Ollama loads a model on first use, not when the service starts, and
        that load measured 3.3 seconds against 0.2 for a warm generation. Left
        unwarmed it lands on whoever speaks first.
        """
        started = time.perf_counter()
        try:
            await self._provider.generate(
                [{"role": "user", "content": WARM_UP_PROMPT}]
            )
        except Exception as error:  # noqa: BLE001 - never block startup
            logger.warning("Could not warm the language model: %s", error)
            return
        logger.info("Language model warm in %.1fs", time.perf_counter() - started)

    async def unload(self) -> None:
        await self._provider.unload()

    def buildMessages(self, text: str, conversation: Conversation) -> list[dict[str, str]]:
        """Prior turns, then the current utterance.

        The runtime records the utterance only after the handler has run, so it
        is added here rather than read from the conversation.
        """
        messages = [{"role": "system", "content": self._systemPrompt}]
        messages.extend(
            {"role": turn.role.value, "content": turn.text} for turn in conversation.turns
        )
        messages.append({"role": "user", "content": text})
        return messages

    async def respond(self, text: str, conversation: Conversation) -> Response:
        result = await self._provider.generate(self.buildMessages(text, conversation))
        logger.debug("Model returned %s", result.describe())
        return Response(
            text=sanitiseForSpeech(result.text),
            usedLlm=True,
            handledBy=self.name,
        )

    async def respondStream(
        self, text: str, conversation: Conversation
    ) -> AsyncIterator[str]:
        """Yield fragments as the model produces them.

        Sanitising happens per fragment, which is why the patterns are line- and
        span-based rather than requiring the whole reply. A stray asterisk that
        arrives split across two fragments will survive, which is a reasonable
        trade for being able to speak before generation finishes.
        """
        async for fragment in self._provider.generateStream(
            self.buildMessages(text, conversation)
        ):
            cleaned = sanitiseFragment(fragment)
            if cleaned:
                yield cleaned

    def describe(self) -> str:
        return f"llm via {self._provider.describe()}"


def sanitiseForSpeech(text: str) -> str:
    """Strip formatting that cannot be spoken."""
    cleaned = text.strip()
    for pattern, replacement in MARKDOWN_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    # Collapse the whitespace left behind by removed markup.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned.strip()


def sanitiseFragment(fragment: str) -> str:
    """Remove characters that would be read aloud, without joining fragments.

    Only single characters are safe to drop here, because anything spanning a
    boundary is not visible from inside one fragment.
    """
    return fragment.replace("*", "").replace("#", "").replace("`", "")
