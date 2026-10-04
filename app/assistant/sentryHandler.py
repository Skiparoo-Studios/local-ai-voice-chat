"""Standing down until somebody says hello.

"Goodbye" puts the assistant into sentry mode: it keeps listening but answers
nothing at all until it is greeted. "Hello" brings it back.

This is the outermost wrapper, in front of toddler mode and books alike,
because standing down means standing down --- not standing down except when a
child is talking to it.

Why it is worth having on a hands-free assistant: with no wake word, every
conversation in the room reaches the recogniser and some of it gets answered.
Sentry mode is the off switch that does not involve closing anything, and
because the greeting is checked against the *whole* utterance, a conversation
that happens to contain the word hello does not wake it.

It is not a security control. Anyone who can be heard can say hello.
"""

from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, ClassVar

from app.assistant.conversation import Conversation
from app.assistant.handlers import Response, ResponseHandler

if TYPE_CHECKING:
    from app.intelligence.llmProvider import LlmProvider

logger = logging.getLogger(__name__)

# Standing down. Anchored to the whole utterance so that "goodbye" inside a
# sentence --- or read aloud from a book --- does not trigger it.
#
# "quit" and "exit" are deliberately absent: those close the program, which is
# a different thing, and the rules handler already answers them.
FAREWELL_PATTERN = re.compile(
    r"^\W*(?:ok(?:ay)?\s+|right\s+|well\s+)?"
    r"(goodbye|good bye|bye|bye bye|good night|goodnight|night night|"
    r"see you|see you later|that.s all|sleep now|go to sleep|stand down|"
    r"go away|leave me alone|thanks that.s all)"
    r"(?:\s+(?:for now|then|assistant))?\W*$",
    re.I,
)

# Waking up again.
GREETING_PATTERN = re.compile(
    r"^\W*(hello|hi|hiya|hey|good morning|good afternoon|good evening|"
    r"morning|wake up|are you (?:there|awake)|you there|come back)"
    r"(?:\s+(?:there|again|assistant))?\W*$",
    re.I,
)

FAREWELL_REPLY = "Goodbye. Say hello when you want me."
GREETING_REPLY = "Hello. I'm listening."


class SentryHandler(ResponseHandler):
    """Answers nothing until greeted, once it has been dismissed."""

    name: ClassVar[str] = "sentry"

    def __init__(
        self,
        base: ResponseHandler,
        *,
        dormant: bool = False,
        greetOnWaking: bool = True,
    ) -> None:
        self._base = base
        self._dormant = dormant
        self._greetOnWaking = greetOnWaking
        self.dismissals = 0
        self.greetings = 0
        self.ignored = 0

    # --- State ------------------------------------------------------------

    @property
    def isDormant(self) -> bool:
        return self._dormant

    @property
    def base(self) -> ResponseHandler:
        return self._base

    @property
    def supportsStreaming(self) -> bool:
        # A dormant assistant says one fixed line or nothing, so there is
        # nothing to stream. Awake, it is the wrapped handler's decision.
        return False if self._dormant else self._base.supportsStreaming

    @property
    def usesLlm(self) -> bool:
        return self._base.usesLlm

    @property
    def languageModel(self) -> LlmProvider | None:
        return self._base.languageModel

    @property
    def promptAfterSeconds(self) -> float | None:
        """Never speaks first while dormant.

        Toddler mode fills silences, and a dismissed assistant filling them
        would be the opposite of standing down.
        """
        return None if self._dormant else self._base.promptAfterSeconds

    async def promptWhenSilent(self) -> Response | None:
        if self._dormant:
            return None
        return await self._base.promptWhenSilent()

    def __getattr__(self, name: str) -> object:
        """Forward anything unrecognised to the handler being wrapped."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._base, name)

    # --- Lifecycle --------------------------------------------------------

    async def load(self) -> None:
        await self._base.load()

    async def warmUp(self) -> None:
        await self._base.warmUp()

    async def unload(self) -> None:
        await self._base.unload()

    # --- Turns ------------------------------------------------------------

    async def respond(self, text: str, conversation: Conversation) -> Response:
        cleaned = text.strip()
        if not cleaned:
            return Response(text="", handledBy=self.name)

        if self._dormant:
            if GREETING_PATTERN.match(cleaned):
                return await self._wake()
            # Silence, not a refusal. A dismissed assistant that answers "I am
            # asleep" every time somebody speaks in the room has not been
            # dismissed at all.
            self.ignored += 1
            logger.debug("Dormant, ignoring %r", cleaned[:60])
            return Response(text="", handledBy=f"{self.name}:dormant")

        if FAREWELL_PATTERN.match(cleaned):
            return await self._standDown()

        return await self._base.respond(cleaned, conversation)

    async def respondStream(
        self, text: str, conversation: Conversation
    ) -> AsyncIterator[str]:
        cleaned = text.strip()
        if self._dormant or FAREWELL_PATTERN.match(cleaned):
            response = await self.respond(cleaned, conversation)
            if response.text:
                yield response.text
            return

        async for fragment in self._base.respondStream(cleaned, conversation):
            yield fragment

    # --- Standing down and waking up --------------------------------------

    async def _standDown(self) -> Response:
        """Go dormant, stopping anything the wrapped handlers are doing.

        A book still playing would carry on happily after "goodbye", which is
        not what anyone means by it, so whatever is wrapped is asked to stop
        if it knows how.
        """
        self._dormant = True
        self.dismissals += 1
        logger.info("Standing down; say hello to wake me")

        reader = getattr(self._base, "reader", None)
        if reader is not None:
            # Standing down must not fail because stopping did.
            with contextlib.suppress(Exception):
                await reader.stop()

        return Response(text=FAREWELL_REPLY, handledBy=f"{self.name}:dormant")

    async def _wake(self) -> Response:
        self._dormant = False
        self.greetings += 1
        logger.info("Awake after %d ignored utterance(s)", self.ignored)
        self.ignored = 0

        if not self._greetOnWaking:
            return Response(text="", handledBy=f"{self.name}:awake")
        return Response(text=GREETING_REPLY, handledBy=f"{self.name}:awake")

    def describe(self) -> str:
        where = "dormant, waiting to be greeted" if self._dormant else "awake"
        return f"{where}; {self._base.describe()}"


__all__ = ["FAREWELL_PATTERN", "GREETING_PATTERN", "SentryHandler"]
