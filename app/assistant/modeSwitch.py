"""Switching between the ordinary assistant and toddler mode by voice.

A wrapper rather than something the runtime knows about, so the switch works
identically in ``--mode converse``, ``--mode wake`` and over the API: they all
go through a response handler, and this is one.

The phrases are checked before anything is delegated, so "start toddler mode"
never reaches a model or, worse, the intent router --- which would otherwise
be free to decide "start" meant a device.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, ClassVar

from app.assistant.conversation import Conversation
from app.assistant.handlers import Response, ResponseHandler
from app.assistant.toddlerHandler import ToddlerHandler

if TYPE_CHECKING:
    from app.intelligence.llmProvider import LlmProvider

logger = logging.getLogger(__name__)

# Whisper hears an adult saying "toddler" reliably enough, but not perfectly,
# so the near misses it actually produces are accepted too.
_TODDLER = r"(?:toddler|toddlers|todler|toddle|toddla)"

ENTER_PATTERNS = (
    re.compile(rf"^\W*(?:start|begin|enter|go into|switch to)\s+{_TODDLER}(?:\s+mode)?\W*$", re.I),
    re.compile(rf"^\W*{_TODDLER}\s+mode\s+(?:on|please)\W*$", re.I),
)

EXIT_PATTERNS = (
    re.compile(rf"^\W*(?:end|stop|exit|leave|quit)\s+{_TODDLER}(?:\s+mode)?\W*$", re.I),
    re.compile(rf"^\W*{_TODDLER}\s+mode\s+off\W*$", re.I),
    # A grown-up reclaiming the assistant, which is how this actually gets
    # used: the phrase is said over the top of a child's conversation.
    re.compile(r"^\W*(?:back to )?normal\s+mode\W*$", re.I),
)


class ModeSwitchingHandler(ResponseHandler):
    """Delegates to the ordinary handler, or to toddler mode."""

    name: ClassVar[str] = "modeSwitch"

    def __init__(
        self,
        base: ResponseHandler,
        toddler: ToddlerHandler,
        *,
        startInToddlerMode: bool = False,
        allowPhrases: bool = True,
    ) -> None:
        self._base = base
        self._toddler = toddler
        self._allowPhrases = allowPhrases
        self._inToddlerMode = False

        # Starting with --toddler has nobody to say hello to yet, so the
        # greeting is held until the child speaks. Without this, entering by
        # flag would skip the greeting that entering by phrase gives --- and
        # with it the request for a name, so every question about the child
        # would stay silently out of reach.
        self._pendingOpening: str | None = None

        if startInToddlerMode:
            self._inToddlerMode = True
            self._toddler.reset()
            self._pendingOpening = self._toddler.openingLine()

    # --- State ------------------------------------------------------------

    @property
    def inToddlerMode(self) -> bool:
        return self._inToddlerMode

    @property
    def active(self) -> ResponseHandler:
        return self._toddler if self._inToddlerMode else self._base

    @property
    def base(self) -> ResponseHandler:
        return self._base

    @property
    def toddler(self) -> ToddlerHandler:
        return self._toddler

    # Both are read once per turn by the runtime, so reporting the active
    # handler's answer rather than a fixed one is enough to make switching
    # take effect immediately.
    @property
    def supportsStreaming(self) -> bool:
        return self.active.supportsStreaming

    @property
    def usesLlm(self) -> bool:
        return self.active.usesLlm

    @property
    def languageModel(self) -> LlmProvider | None:
        return self._base.languageModel

    # Follows the active handler, so an assistant in normal mode stays silent
    # while one in toddler mode fills a gap. A held greeting counts as
    # something to say: nobody has heard it yet.
    @property
    def promptAfterSeconds(self) -> float | None:
        if self._pendingOpening is not None and self._inToddlerMode:
            return self._toddler.promptAfterSeconds
        return self.active.promptAfterSeconds

    async def promptWhenSilent(self) -> Response | None:
        opening = self._takeOpening()
        if opening is not None:
            return opening
        return await self.active.promptWhenSilent()

    def __getattr__(self, name: str) -> object:
        """Forward anything unrecognised to the handler being wrapped.

        The API and the prompt both ask a handler what it has --- ``registry``
        for the device list, ``provider`` for ``:models`` --- with getattr.
        Inserting a wrapper in front would otherwise silently remove those
        features, and the device panel would vanish from the web client the
        moment toddler mode became available.

        Deliberately the base handler rather than the active one: a parent
        listing devices over HTTP should not be told there are none because a
        child is mid-conversation in the next room.
        """
        # __getattr__ runs only when normal lookup fails, so nothing real is
        # shadowed. The guard is for dunder lookups during copying and pickling.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._base, name)

    # --- Lifecycle --------------------------------------------------------

    async def load(self) -> None:
        await self._base.load()
        await self._toddler.load()

    async def warmUp(self) -> None:
        await self._base.warmUp()

    async def unload(self) -> None:
        await self._base.unload()
        await self._toddler.unload()

    # --- Turns ------------------------------------------------------------

    async def respond(self, text: str, conversation: Conversation) -> Response:
        switch = self._switchFor(text)
        if switch is not None:
            return switch

        opening = self._takeOpening()
        if opening is not None:
            return opening

        return await self.active.respond(text, conversation)

    def _takeOpening(self) -> Response | None:
        """The held greeting, once, on the first thing said in toddler mode."""
        if self._pendingOpening is None or not self._inToddlerMode:
            return None

        spoken, self._pendingOpening = self._pendingOpening, None
        return Response(text=spoken, handledBy=f"{self.name}:toddler")

    async def respondStream(
        self, text: str, conversation: Conversation
    ) -> AsyncIterator[str]:
        immediate = self._switchFor(text) or self._takeOpening()
        if immediate is not None:
            if immediate.text:
                yield immediate.text
            return

        async for fragment in self.active.respondStream(text, conversation):
            yield fragment

    def _switchFor(self, text: str) -> Response | None:
        """The reply to a mode phrase, or None if this was not one."""
        if not self._allowPhrases:
            return None

        cleaned = text.strip()
        if not cleaned:
            return None

        if any(pattern.match(cleaned) for pattern in ENTER_PATTERNS):
            return self.enterToddlerMode()

        if any(pattern.match(cleaned) for pattern in EXIT_PATTERNS):
            return self.leaveToddlerMode()

        return None

    def enterToddlerMode(self) -> Response:
        """Switch on, forgetting any previous session."""
        alreadyThere = self._inToddlerMode
        self._inToddlerMode = True

        if alreadyThere:
            return Response(text="We're already playing!", handledBy=f"{self.name}:toddler")

        # Said now rather than held, so a greeting from --toddler that nobody
        # heard does not surface a turn later.
        self._pendingOpening = None
        self._toddler.reset()
        logger.info("Entering toddler mode")
        return Response(text=self._toddler.openingLine(), handledBy=f"{self.name}:toddler")

    def leaveToddlerMode(self) -> Response:
        """Switch off, returning to the configured handler."""
        if not self._inToddlerMode:
            return Response(text="We're not in toddler mode.", handledBy=f"{self.name}:normal")

        spoken = self._toddler.closingLine()
        self._inToddlerMode = False
        logger.info("Leaving toddler mode")
        return Response(text=spoken, handledBy=f"{self.name}:normal")

    def describe(self) -> str:
        where = "toddler mode" if self._inToddlerMode else "normal mode"
        return f"{where} (switchable); {self.active.describe()}"
