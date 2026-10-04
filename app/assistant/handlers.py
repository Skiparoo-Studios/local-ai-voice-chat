"""Turning what the user said into what the assistant should say.

This is the seam the rest of the system is built around. Stage 4 adds a
language-model handler and Stage 5 an intent router; both implement the same
interface, and the runtime does not change.

The rule handler here is deliberately small. It is not an attempt at natural
language understanding --- it exists so that Stage 3's round trip can be
measured and heard without a language model in the way.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar

from app.assistant.conversation import Conversation

if TYPE_CHECKING:
    from app.config import Settings
    from app.intelligence.llmProvider import LlmProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Response:
    """What the assistant will say, and how it decided."""

    text: str
    usedLlm: bool = False
    handledBy: str = ""
    shouldStop: bool = False

    @property
    def isEmpty(self) -> bool:
        return not self.text.strip()


class ResponseHandler(ABC):
    """Produces a reply to an utterance."""

    name: ClassVar[str] = "unnamed"

    @abstractmethod
    async def respond(self, text: str, conversation: Conversation) -> Response:
        """Reply to ``text``. ``conversation`` excludes the current utterance."""

    @property
    def supportsStreaming(self) -> bool:
        """Whether ``respondStream`` produces fragments before the reply is complete.

        A handler that answers instantly gains nothing from streaming, so the
        runtime only takes the streaming path when this is true.
        """
        return False

    @property
    def usesLlm(self) -> bool:
        """Whether replies come from a language model. Reported in events."""
        return False

    @property
    def promptAfterSeconds(self) -> float | None:
        """Seconds of silence after which this handler has something to say.

        None, for almost every handler: an assistant that speaks into an empty
        room unasked is a worse assistant. Toddler mode is the exception, where
        a child who has wandered off mid-question is better served by hearing
        the answer than by silence.
        """
        return None

    async def promptWhenSilent(self) -> Response | None:
        """What to say when nobody has spoken. None to stay quiet."""
        return None

    @property
    def languageModel(self) -> LlmProvider | None:
        """The model backing this handler, if there is one.

        Exposed so that a second handler --- toddler mode --- can talk to the
        same service with its own system prompt, rather than opening a second
        connection to a model already resident in memory.
        """
        return None

    async def respondStream(
        self, text: str, conversation: Conversation
    ) -> AsyncIterator[str]:
        """Yield the reply in fragments.

        The default produces the whole reply as one fragment, so that any
        handler can be driven through the streaming path.
        """
        response = await self.respond(text, conversation)
        if response.text:
            yield response.text

    async def load(self) -> None:  # noqa: B027
        """Prepare anything expensive. Called once at startup."""

    async def warmUp(self) -> None:  # noqa: B027
        """Pay any first-use cost before a user is waiting on it.

        Separate from ``load`` because a handler backed by a service does not
        control when that service loads its model: Ollama loads on first
        request, so the only way to pay for it early is to make one.
        """

    async def unload(self) -> None:  # noqa: B027
        """Release anything held."""

    def describe(self) -> str:
        return self.name


class EchoHandler(ResponseHandler):
    """Repeats what was said. The simplest thing that closes the loop."""

    name: ClassVar[str] = "echo"

    async def respond(self, text: str, conversation: Conversation) -> Response:
        return Response(text=f"You said: {text}", handledBy=self.name)


@dataclass(frozen=True, slots=True)
class Rule:
    """A pattern and the reply it produces."""

    pattern: re.Pattern[str]
    reply: str | None = None
    stops: bool = False


def compileRules() -> tuple[Rule, ...]:
    """Patterns matched against the whole utterance, in order.

    Anchored on word boundaries so that "stop" matches "stop" and "please stop"
    but not "stopwatch".
    """

    def compile(expression: str) -> re.Pattern[str]:
        return re.compile(expression, re.IGNORECASE)

    return (
        Rule(compile(r"^\W*(goodbye|good bye|bye|exit|quit|shut ?down)\W*$"),
             "Goodbye.", stops=True),
        Rule(compile(r"\b(stop|cancel|never ?mind)\b"), "Stopped."),
        Rule(compile(r"^\W*(hello|hi|hey|good morning|good afternoon|good evening)\b"),
             "Hello. How can I help?"),
        Rule(compile(r"\b(how are you|how's it going)\b"),
             "I'm running locally and working fine, thank you."),
        Rule(compile(r"\b(thank you|thanks|cheers)\b"), "You're welcome."),
        Rule(compile(r"\bwhat(?:'s| is) your name\b"), None),
        Rule(compile(r"\bwhat(?:'s| is) the time\b|\bwhat time is it\b"), None),
        Rule(compile(r"\bwhat can you do\b|\bhelp\b"), None),
    )


class RuleHandler(ResponseHandler):
    """Deterministic replies for a handful of phrases, echoing otherwise.

    Nothing here reaches a model, which is the point: it demonstrates that the
    §12 Layer 1 path can answer without one, and gives Stage 5 a latency
    baseline to compare deterministic tool calls against.
    """

    name: ClassVar[str] = "rules"

    def __init__(self, assistantName: str = "Assistant", capabilities: str | None = None) -> None:
        self._assistantName = assistantName
        self._capabilities = capabilities or (
            "At the moment I can hear you, repeat what you say, and answer a few "
            "fixed questions. Home automation and conversation come later."
        )
        self._rules = compileRules()

    async def respond(self, text: str, conversation: Conversation) -> Response:
        cleaned = text.strip()
        if not cleaned:
            return Response(text="", handledBy=self.name)

        matched = self.match(cleaned)
        if matched is not None:
            return matched

        return Response(text=f"You said: {cleaned}", handledBy=f"{self.name}:echo")

    def match(self, text: str) -> Response | None:
        """The deterministic answer for this phrase, if there is one.

        Separate from ``respond`` because the echo is a last resort for this
        handler and a wrong answer for anything else. The intent router needs
        to know whether a rule actually fired so it can pass anything else to
        the model rather than echoing at the user.
        """
        cleaned = text.strip()
        if not cleaned:
            return None

        for rule in self._rules:
            if not rule.pattern.search(cleaned):
                continue
            reply = rule.reply if rule.reply is not None else self._dynamicReply(cleaned)
            return Response(text=reply, handledBy=self.name, shouldStop=rule.stops)

        return None

    def _dynamicReply(self, text: str) -> str:
        """Replies that depend on something other than the pattern itself."""
        lowered = text.lower()

        if "your name" in lowered:
            return f"My name is {self._assistantName}."
        if "time" in lowered:
            # Spoken, so a 24-hour clock reads awkwardly.
            return f"It's {datetime.now().strftime('%I:%M %p').lstrip('0')}."
        return self._capabilities


class UnknownHandlerError(Exception):
    """Raised when configuration names a handler that does not exist."""


def createResponseHandler(
    settings: Settings,
    *,
    audioOutput: object | None = None,
    textToSpeech: object | None = None,
) -> ResponseHandler:
    """Build the handler named in configuration.

    The language-model handler is constructed here rather than registered
    lazily like the speech providers, because it needs a provider built from
    the same settings and the wiring is clearer in one place than split across
    a registry entry and a factory.

    Toddler mode wraps whatever was chosen rather than replacing it, so that
    the spoken phrases can switch back to it.
    """
    base = _createBaseHandler(settings)

    # Reading a book needs somewhere to play it, so this is only available to
    # a caller that has built the audio devices. A caller that has not gets
    # the assistant without it rather than an error.
    if settings.books.enabled and audioOutput is not None:
        base = _wrapWithBooks(settings, base, audioOutput, textToSpeech)

    if settings.toddler.enabled or settings.toddler.allowModePhrases:
        base = _wrapWithToddlerMode(settings, base)

    if settings.sentry.enabled:
        base = _wrapWithSentry(settings, base)

    return base


def _wrapWithSentry(settings: Settings, base: ResponseHandler) -> ResponseHandler:
    """Put the stand-down gate outermost.

    Outside toddler mode and books both, because "goodbye" has to mean
    everything stops answering, not everything except whichever mode happens
    to be active.
    """
    from app.assistant.sentryHandler import SentryHandler

    return SentryHandler(
        base,
        dormant=settings.sentry.dormant,
        greetOnWaking=settings.sentry.greetOnWaking,
    )


def _wrapWithBooks(
    settings: Settings,
    base: ResponseHandler,
    audioOutput: object,
    textToSpeech: object | None,
) -> ResponseHandler:
    """Put the book handler in front of the configured handler.

    Wrapped rather than substituted: "read me a story" is a request made in
    the middle of ordinary conversation, not a mode to be entered.
    """
    from app.assistant.bookHandler import BookHandler
    from app.books.bookmarks import Bookmarks
    from app.books.library import BookLibrary
    from app.books.reader import BookReader

    library = BookLibrary(settings.books.audioDirectory, settings.books.textDirectory)
    bookmarks = Bookmarks(settings.books.bookmarksFile)
    reader = BookReader(
        audioOutput,  # type: ignore[arg-type]
        textToSpeech=textToSpeech,  # type: ignore[arg-type]
        voice=settings.textToSpeech.voice,
        language=settings.textToSpeech.language or None,
        blockSeconds=settings.books.blockSeconds,
        volume=settings.books.volume,
    )
    logger.info("Book reading available: %s", library.describe())
    return BookHandler(base, library, reader, bookmarks)


def _createBaseHandler(settings: Settings) -> ResponseHandler:
    choice = settings.assistant.handler

    if choice == "rules":
        return RuleHandler(assistantName=settings.assistant.name)
    if choice == "echo":
        return EchoHandler()
    if choice == "llm":
        return _buildLlmHandler(settings)
    if choice == "router":
        return _buildRouter(settings)

    raise UnknownHandlerError(
        f"Unknown response handler {choice!r}. Available: echo, llm, router, rules"
    )


def _wrapWithToddlerMode(settings: Settings, base: ResponseHandler) -> ResponseHandler:
    """Put a mode switch in front of the configured handler.

    A failure to read the questions file must not stop the assistant starting:
    the ordinary handler still works, and the person who broke the JSON needs
    to be told rather than left with a machine that will not boot.
    """
    from app.assistant.modeSwitch import ModeSwitchingHandler
    from app.assistant.questionBank import QuestionBank, QuestionBankError
    from app.assistant.toddlerHandler import ToddlerHandler

    try:
        bank = QuestionBank.load(settings.toddler.questionsFile)
    except QuestionBankError as error:
        logger.error("Toddler mode disabled: %s", error)
        return base

    toddler = ToddlerHandler(
        bank,
        llm=base.languageModel,
        assistantName=settings.assistant.name,
        askForName=settings.toddler.askForName,
        promptAfterSeconds=settings.toddler.promptAfterSeconds,
        giveUpAfterPrompts=settings.toddler.giveUpAfterPrompts,
    )

    if toddler.languageModel is None:
        logger.info("Toddler mode has no language model; replies will be fixed phrases")

    return ModeSwitchingHandler(
        base,
        toddler,
        startInToddlerMode=settings.toddler.enabled,
        allowPhrases=settings.toddler.allowModePhrases,
    )


def _buildLlmHandler(settings: Settings):  # type: ignore[no-untyped-def]
    from app.assistant.llmHandler import LlmHandler
    from app.intelligence.llmProvider import createLlmProvider

    return LlmHandler.fromSettings(settings, createLlmProvider(settings))


def _buildRouter(settings: Settings):  # type: ignore[no-untyped-def]
    """Assemble the router with its tools and, optionally, a language model.

    The model is optional on purpose: deterministic commands are the point of
    Layer 1, and they should work on a machine with no model service running.
    """
    from app.assistant.intentRouter import buildIntentRouter
    from app.assistant.toolRegistry import ToolRegistry
    from app.automation.automationProvider import createAutomationProvider
    from app.tools.homeAutomation import HomeAutomationTool, buildAutomationTools

    provider = createAutomationProvider(settings)
    registry = ToolRegistry(allowStateChanges=settings.automation.allowStateChanges)

    automationTool: HomeAutomationTool | None = None
    for tool in buildAutomationTools(provider, tuple(settings.automation.allowedActions)):
        registry.register(tool)
        if isinstance(tool, HomeAutomationTool):
            automationTool = tool

    llmHandler = None
    if settings.llm.model:
        llmHandler = _buildLlmHandler(settings)
    else:
        logger.info("No llm.model configured; the router will use Layer 1 only")

    return buildIntentRouter(
        settings, registry, llmHandler=llmHandler, automationTool=automationTool
    )
