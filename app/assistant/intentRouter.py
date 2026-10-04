"""Routing an utterance to the cheapest thing that can answer it.

§12 of the brief describes three layers, and this implements them in order:

    Layer 1  deterministic patterns, no model. "Turn the kitchen light off"
             becomes a tool call directly.
    Layer 2  the language model, offered the tool schemas, for phrasing the
             patterns do not cover.
    Layer 3  ordinary conversation, when no tool applies.

The ordering is the point. Most real commands are Layer 1, which costs
microseconds rather than a model round trip, and never puts model output in
the path of something that changes the physical environment.

A note on streaming: Layers 2 and 3 share one non-streaming request, because
tool selection and token streaming do not combine cleanly across services. The
cost is that conversational replies through the router do not stream. Layer 1,
which handles the commands people actually repeat, is unaffected and remains
the fastest path in the system.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from app.assistant.conversation import Conversation
from app.assistant.handlers import Response, ResponseHandler, RuleHandler
from app.assistant.llmHandler import LlmHandler, sanitiseForSpeech
from app.assistant.toolRegistry import ToolRegistry
from app.automation.deviceMatcher import DOMAIN_WORDS, domainFor, matchDevice, tokenise
from app.tools.base import ToolError, ToolNotPermittedError, ToolResult, ToolValidationError
from app.tools.homeAutomation import HomeAutomationTool

if TYPE_CHECKING:
    from app.config import Settings
    from app.intelligence.llmProvider import LlmProvider

logger = logging.getLogger(__name__)

DEFAULT_DOMAIN = "light"

# Added when tools are offered. A small model will otherwise sometimes reply
# "I've turned the lamp on" without calling anything, which is worse than
# refusing: the user believes the house changed when it did not.
TOOL_SYSTEM_NOTE = (
    "You can control devices in the home, but only by calling the tools "
    "provided. Never state that you have turned something on or off, or "
    "changed anything, unless you called a tool to do it. If you cannot do "
    "something, say so plainly."
)

# Utterances that map straight to an action. Anchored so that a passing
# mention of "off" in conversation does not switch anything off.
COMMAND_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^\W*(?:please\s+)?turn\s+(?:on|up)\s+(?P<target>.+?)\s*$", re.I), "turnOn"),
    (re.compile(r"^\W*(?:please\s+)?turn\s+(?:off|down)\s+(?P<target>.+?)\s*$", re.I), "turnOff"),
    (re.compile(r"^\W*(?:please\s+)?turn\s+(?P<target>.+?)\s+on\s*$", re.I), "turnOn"),
    (re.compile(r"^\W*(?:please\s+)?turn\s+(?P<target>.+?)\s+off\s*$", re.I), "turnOff"),
    (re.compile(r"^\W*(?:please\s+)?switch\s+on\s+(?P<target>.+?)\s*$", re.I), "turnOn"),
    (re.compile(r"^\W*(?:please\s+)?switch\s+off\s+(?P<target>.+?)\s*$", re.I), "turnOff"),
    (re.compile(r"^\W*(?:please\s+)?switch\s+(?P<target>.+?)\s+on\s*$", re.I), "turnOn"),
    (re.compile(r"^\W*(?:please\s+)?switch\s+(?P<target>.+?)\s+off\s*$", re.I), "turnOff"),
    (re.compile(r"^\W*(?:please\s+)?toggle\s+(?P<target>.+?)\s*$", re.I), "toggle"),
    (re.compile(r"^\W*(?P<target>.+?)\s+(?:on|off)\s*,?\s*please\s*$", re.I), ""),
)

# "all the lights", "every light" address a group rather than one device.
PLURAL_PATTERN = re.compile(r"\b(all|every|both)\b", re.I)


@dataclass(frozen=True, slots=True)
class Intent:
    """A command recognised without a model."""

    action: str
    target: str
    all: bool = False

    def asArguments(self) -> dict[str, object]:
        return {"action": self.action, "target": self.target, "all": self.all}


class IntentRouter(ResponseHandler):
    """Answers using the cheapest layer that can."""

    name: ClassVar[str] = "router"

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        llmHandler: LlmHandler | None = None,
        fallback: ResponseHandler | None = None,
        automationTool: HomeAutomationTool | None = None,
        useLlmForTools: bool = True,
    ) -> None:
        self._registry = registry
        self._llm = llmHandler
        self._fallback = fallback or RuleHandler()
        self._automationTool = automationTool
        self._useLlmForTools = useLlmForTools
        self.lastLayer = 0

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def usesLlm(self) -> bool:
        return self._llm is not None

    @property
    def languageModel(self) -> LlmProvider | None:
        return self._llm.provider if self._llm is not None else None

    @property
    def supportsStreaming(self) -> bool:
        # Layer 1 answers instantly and Layers 2 and 3 share a non-streaming
        # request, so there is nothing for the runtime to stream.
        return False

    async def load(self) -> None:
        if self._llm is not None:
            await self._llm.load()

    async def warmUp(self) -> None:
        if self._llm is not None:
            await self._llm.warmUp()

    async def unload(self) -> None:
        if self._llm is not None:
            await self._llm.unload()

    # --- Routing ---------------------------------------------------------

    async def respond(self, text: str, conversation: Conversation) -> Response:
        cleaned = text.strip()
        if not cleaned:
            return Response(text="", handledBy=self.name)

        intent = await self.recogniseIntent(cleaned)
        if intent is not None:
            self.lastLayer = 1
            logger.debug("Layer 1 matched %s on %r", intent.action, intent.target)
            return await self._runTool("homeAutomation", intent.asArguments(), layer=1)

        # Things the assistant can answer itself. The clock is the clearest
        # case: a language model does not know the time, and asking one is both
        # slower and wrong. Only a rule that actually fires is used --- the
        # fallback echo belongs to the rules handler alone, and echoing at
        # someone here would be worse than passing them to the model.
        deterministic = self._deterministicAnswer(cleaned)
        if deterministic is not None:
            self.lastLayer = 1
            return deterministic

        if self._llm is not None and self._useLlmForTools and len(self._registry):
            self.lastLayer = 2
            return await self._respondWithTools(cleaned, conversation)

        if self._llm is not None:
            self.lastLayer = 3
            return await self._llm.respond(cleaned, conversation)

        self.lastLayer = 3
        return await self._fallback.respond(cleaned, conversation)

    async def respondStream(
        self, text: str, conversation: Conversation
    ) -> AsyncIterator[str]:
        response = await self.respond(text, conversation)
        if response.text:
            yield response.text

    # --- Layer 1 ---------------------------------------------------------

    def _deterministicAnswer(self, text: str) -> Response | None:
        """An answer the assistant can give without a model, if it has one."""
        matcher = getattr(self._fallback, "match", None)
        return matcher(text) if matcher is not None else None

    async def recogniseIntent(self, text: str) -> Intent | None:
        """Match an utterance against the deterministic command patterns.

        Returns None when nothing matches, or when the phrase matches but does
        not name anything controllable --- "turn off the news" should reach the
        model rather than hunt for a device called "the news".
        """
        for pattern, verb in COMMAND_PATTERNS:
            match = pattern.match(text)
            if not match:
                continue

            target = match.group("target").strip()
            if not target:
                continue

            resolvedVerb = verb or _verbFromSuffix(text)
            if not resolvedVerb:
                continue

            domain = await self._domainFor(target)
            if domain is None:
                logger.debug("Pattern matched %r but no device resembles it", target)
                return None

            return Intent(
                action=f"{domain}.{resolvedVerb}",
                target=target,
                all=bool(PLURAL_PATTERN.search(target)),
            )
        return None

    async def _domainFor(self, target: str) -> str | None:
        """Which device domain a target phrase refers to.

        A spoken domain word settles it. Otherwise the device list does, which
        is what makes "turn off the coffee machine" reach a switch without the
        user naming its type.
        """
        tokens = tokenise(target)
        spoken = domainFor(tokens)
        if spoken:
            return spoken

        if self._automationTool is None:
            return DEFAULT_DOMAIN

        try:
            devices = await self._automationTool.devices()
        except Exception as error:  # noqa: BLE001 - fall through to the model
            logger.warning("Could not list devices for intent matching: %s", error)
            return None

        match = matchDevice(target, devices)
        if match.found and match.best is not None:
            return match.best.domain
        if match.ambiguous and match.best is not None:
            return match.best.domain
        return None

    # --- Layer 2 ---------------------------------------------------------

    async def _respondWithTools(self, text: str, conversation: Conversation) -> Response:
        """Offer the tools to the model and act on what it proposes."""
        assert self._llm is not None

        messages = self._llm.buildMessages(text, conversation)
        # Inserted after the assistant's own system prompt so it is the last
        # instruction the model reads before the conversation.
        messages.insert(1, {"role": "system", "content": TOOL_SYSTEM_NOTE})

        result = await self._llm.provider.generate(messages, tools=self._registry.schemas())

        if not result.hasToolCalls:
            self.lastLayer = 3
            spoken = sanitiseForSpeech(result.text)
            if not spoken:
                # Neither an action nor an answer. Saying nothing is the worst
                # option: the user cannot tell whether it worked, failed, or
                # was heard at all.
                logger.warning("Model returned no tool call and no text")
                spoken = "Sorry, I'm not sure what you'd like me to do."
            return Response(
                text=spoken,
                usedLlm=True,
                handledBy=f"{self.name}:conversation",
            )

        call = result.toolCalls[0]
        logger.info("Model proposed %s with %s", call.name, call.arguments)
        return await self._runTool(call.name, call.arguments, layer=2, fromModel=True)

    # --- Execution -------------------------------------------------------

    async def _runTool(
        self,
        name: str,
        arguments: dict[str, object],
        *,
        layer: int,
        fromModel: bool = False,
    ) -> Response:
        """Run a tool and turn the outcome into something speakable.

        Every refusal produces a spoken explanation rather than silence, and is
        logged. A user who is told "that isn't allowed" can act; one who hears
        nothing assumes it worked.
        """
        try:
            result = await self._registry.invoke(name, dict(arguments))
        except ToolNotPermittedError as error:
            logger.warning("Refused tool %r from layer %d: %s", name, layer, error)
            return Response(
                text=str(error), usedLlm=fromModel, handledBy=f"{self.name}:refused"
            )
        except ToolValidationError as error:
            logger.warning("Invalid arguments for %r from layer %d: %s", name, layer, error)
            return Response(
                text="I didn't understand what to do with that.",
                usedLlm=fromModel,
                handledBy=f"{self.name}:invalid",
            )
        except ToolError as error:
            logger.error("Tool %r failed: %s", name, error)
            return Response(
                text="Something went wrong doing that.",
                usedLlm=fromModel,
                handledBy=f"{self.name}:error",
            )

        return Response(
            text=_speak(result),
            usedLlm=fromModel,
            handledBy=f"{self.name}:layer{layer}",
        )

    def describe(self) -> str:
        parts = [f"router over {len(self._registry)} tool(s)"]
        if self._llm is not None:
            parts.append(self._llm.describe())
        return "; ".join(parts)


def _verbFromSuffix(text: str) -> str:
    """The verb implied by a trailing 'on please' or 'off please'."""
    lowered = text.lower()
    if re.search(r"\boff\b", lowered):
        return "turnOff"
    if re.search(r"\bon\b", lowered):
        return "turnOn"
    return ""


def _speak(result: ToolResult) -> str:
    return result.message or ("Done." if result.succeeded else "That didn't work.")


def buildIntentRouter(
    settings: Settings,
    registry: ToolRegistry,
    *,
    llmHandler: LlmHandler | None = None,
    automationTool: HomeAutomationTool | None = None,
) -> IntentRouter:
    """Assemble the router from configuration and built components."""
    return IntentRouter(
        registry,
        llmHandler=llmHandler,
        fallback=RuleHandler(assistantName=settings.assistant.name),
        automationTool=automationTool,
        useLlmForTools=settings.assistant.useLlmForTools,
    )


__all__ = ["DOMAIN_WORDS", "Intent", "IntentRouter", "buildIntentRouter"]
