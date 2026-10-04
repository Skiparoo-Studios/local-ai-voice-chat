"""Talking with a small child.

Two things are happening in every turn, and keeping them apart is what makes
this work.

The *questions* are deterministic. What a sheep says, how old someone is and
whether they brushed their teeth are all judged against a list of expected
answers held in :mod:`app.assistant.questionBank`. A three-year-old who says
"baa" should hear "that's right" in a fraction of a second, and a small local
model is both slower and less consistent at deciding whether "baa" means "baa"
than a list of strings is.

The *conversation between* the questions is the model's job. It is given a
great deal of context --- the child's name and age, what has already been
asked, what was got right and wrong, and whether a question is about to be
appended --- and a hard instruction to reply in one short sentence a small
child would understand.

The rhythm matters as much as the content. Nobody enjoys being interrogated,
so a question is offered only when the child says something with no content of
its own ("hello", "yeah"), or when a couple of ordinary turns have passed. Only
ever one question per reply, and it is added by this handler rather than by the
model, so "only one" is a guarantee rather than a request.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, ClassVar

from app.assistant.conversation import Conversation
from app.assistant.handlers import Response, ResponseHandler
from app.assistant.questionBank import (
    MAXIMUM_DIFFICULTY,
    MINIMUM_DIFFICULTY,
    Question,
    QuestionBank,
    containsAnswer,
    normalise,
    render,
)
from app.assistant.sentences import splitIntoSentences

if TYPE_CHECKING:
    from app.intelligence.llmProvider import LlmProvider

logger = logging.getLogger(__name__)

# At most this many sentences are spoken back. A model that ignores the
# instruction to be brief is cut rather than allowed to lecture a toddler.
MAXIMUM_SENTENCES = 2

# Consecutive right answers in a topic before harder questions are offered,
# and consecutive wrong ones before easier ones are. Two rather than one,
# because a three-year-old guesses and one lucky "moo" is not mastery.
PROMOTION_STREAK = 2

SYSTEM_PROMPT = (
    "You are {assistant}, and you are talking to a small child of about three "
    "years old. Follow these rules on every single reply:\n"
    "- Reply with ONE short sentence. Two at the very most.\n"
    "- Use only simple words a three-year-old knows.\n"
    "- Be warm, playful and encouraging.\n"
    "- Never say anything frightening, violent, sad, rude or grown-up.\n"
    "- Never use markdown, emoji, lists, brackets or symbols. Your words are "
    "spoken out loud.\n"
    "- If you cannot tell what the child said, say something friendly and "
    "simple rather than asking them to repeat it twice.\n"
    "- Never claim to have done anything in the real world."
)

# Said when there is no language model, so toddler mode still works on a
# machine with nothing running. Rotated so it does not repeat immediately.
ACKNOWLEDGEMENTS = (
    "Okay!",
    "That's nice.",
    "Good talking!",
    "Mmm, I see.",
)

NAME_PATTERNS = (
    re.compile(r"\bmy name(?:'s| is)\s+(?P<name>[a-z]+)", re.I),
    re.compile(r"\bi(?:'m| am)\s+(?P<name>[a-z]+)", re.I),
    re.compile(r"\bcall me\s+(?P<name>[a-z]+)", re.I),
    re.compile(r"\bit(?:'s| is)\s+(?P<name>[a-z]+)", re.I),
    re.compile(r"\bthis is\s+(?P<name>[a-z]+)", re.I),
)

# Words that arrive where a name would be but are not one.
NOT_NAMES = frozenset(
    {
        "a", "and", "dont", "good", "hello", "hi", "know", "me", "my", "no",
        "not", "ok", "okay", "sure", "thanks", "the", "um", "yeah", "yes",
        "you", "your",
    }
)

# An utterance carrying no content of its own. These are the moments to offer a
# question, because the child has opened a conversation without supplying one.
CONTENTLESS = frozenset(
    {
        "", "aha", "eh", "hello", "hey", "hi", "hiya", "huh", "mm", "mmm", "no",
        "nope", "oh", "ok", "okay", "um", "uh", "what", "yay", "yeah", "yep",
        "yes", "hello there", "hi there", "i dont know", "dont know",
    }
)

GREETING = "Hello! It's lovely to talk to you."
FAREWELL = "Bye bye! That was fun."
NAME_REQUEST = "What's your name?"

# Said before revealing the answer to a question nobody attempted, so that the
# reveal does not arrive as a correction of something never said.
REVEAL_LEAD_INS = (
    "Shall I tell you?",
    "I know this one!",
    "Let me help you.",
)

# Said before offering another question after a silence.
NUDGE_LEAD_INS = (
    "Here's another one.",
    "Let's try a different one.",
    "I've got one for you.",
)


class ToddlerHandler(ResponseHandler):
    """Holds a simple conversation, with questions mixed in."""

    name: ClassVar[str] = "toddler"

    def __init__(
        self,
        bank: QuestionBank,
        *,
        llm: LlmProvider | None = None,
        assistantName: str = "Assistant",
        askForName: bool = True,
        promptAfterSeconds: float = 7.0,
        giveUpAfterPrompts: int = 3,
    ) -> None:
        self._bank = bank
        self._llm = llm
        self._assistantName = assistantName
        self._askForName = askForName
        self._promptAfterSeconds = promptAfterSeconds
        self._giveUpAfterPrompts = giveUpAfterPrompts

        # Session state. Deliberately held here rather than on a Session:
        # there is one child in the room, one speaker, and one conversation.
        self._childName: str | None = bank.childName
        self._childAge: int | None = bank.childAge
        self._pending: Question | None = None
        self._askCounts: dict[str, int] = {}
        self._phraseCounts: dict[str, int] = {}
        self._turnsSinceQuestion = 0
        self._awaitingName = False
        self._acknowledgementIndex = 0
        self._lastReplyAsked = False
        # Consecutive things said into a silence. Reset the moment the child
        # says anything at all.
        self._unansweredPrompts = 0
        # How the child is doing, per topic: the difficulty being offered, and
        # the run of right or wrong answers working towards changing it.
        self._topicLevels: dict[str, int] = {}
        self._topicStreaks: dict[str, int] = {}
        self._lastTopic: str | None = None

    # --- Properties -------------------------------------------------------

    @property
    def bank(self) -> QuestionBank:
        return self._bank

    @property
    def childName(self) -> str | None:
        return self._childName

    @property
    def pendingQuestion(self) -> Question | None:
        """The question waiting for an answer, if any. Exposed for tests."""
        return self._pending

    @property
    def usesLlm(self) -> bool:
        return self._llm is not None

    @property
    def languageModel(self) -> LlmProvider | None:
        return self._llm

    @property
    def supportsStreaming(self) -> bool:
        # Replies are one or two sentences and are often assembled from a
        # deterministic phrase plus a question, so there is nothing worth
        # streaming: the whole reply is ready at once.
        return False

    # --- Entering and leaving --------------------------------------------

    def reset(self) -> None:
        """Forget the session, keeping anything the file supplied.

        Called when toddler mode is entered, so a second child --- or the same
        one tomorrow --- does not inherit the last session's answers.
        """
        self._childName = self._bank.childName
        self._childAge = self._bank.childAge
        self._pending = None
        self._askCounts.clear()
        self._phraseCounts.clear()
        self._turnsSinceQuestion = 0
        self._awaitingName = False
        self._lastReplyAsked = False
        self._unansweredPrompts = 0
        self._topicLevels.clear()
        self._topicStreaks.clear()
        self._lastTopic = None

    def openingLine(self) -> str:
        """What to say on entering toddler mode.

        Asks for a name when none is known, because every personal question in
        the bank is held back until one is.
        """
        if self._childName:
            return f"Hello {self._childName}! It's lovely to talk to you."

        if self._askForName:
            self._awaitingName = True
            return f"{GREETING} {NAME_REQUEST}"

        return GREETING

    def closingLine(self) -> str:
        if self._childName:
            return f"Bye bye {self._childName}! That was fun."
        return FAREWELL

    # --- Filling a silence ------------------------------------------------

    @property
    def promptAfterSeconds(self) -> float | None:
        """How long to wait before saying something unprompted.

        None once the child has stopped replying: an assistant still asking
        questions of an empty room is worse company than a quiet one, and a
        toddler who has wandered off has genuinely finished.
        """
        if self._promptAfterSeconds <= 0:
            return None
        if self._hasGivenUp:
            return None
        return self._promptAfterSeconds

    @property
    def _hasGivenUp(self) -> bool:
        return self._unansweredPrompts >= self._giveUpAfterPrompts

    async def promptWhenSilent(self) -> Response | None:
        """Fill a silence: reveal the answer, or offer another question.

        Revealing comes first because the child was asked something and did not
        answer. Hearing "a sheep says baa" is the point of the exercise, and it
        is a better use of the silence than repeating the question at them.
        """
        if self._hasGivenUp:
            return None

        # Waiting on a name is not a question to reveal the answer to. Nobody
        # else can say what the child is called, so let it go and carry on.
        if self._awaitingName:
            self._awaitingName = False

        self._unansweredPrompts += 1

        if self._pending is not None:
            return self._revealAnswer(self._pending)

        return self._offerAnotherQuestion()

    def _revealAnswer(self, question: Question) -> Response:
        """Say what the answer was, without implying the child got it wrong."""
        self._pending = None
        self._turnsSinceQuestion = 0

        # The first 'incorrect' phrase states the answer plainly; later ones
        # may open with "not quite", which would be unfair to someone who said
        # nothing at all.
        spoken = render(question.incorrect[0], self._facts()) if question.incorrect else None
        if not spoken:
            return Response(text="", handledBy=f"{self.name}:silent")

        leadIn = self._nextPhrase("reveal", REVEAL_LEAD_INS)
        logger.debug("Revealing the answer to %r after a silence", question.id)
        self._lastReplyAsked = "?" in spoken
        return Response(text=f"{leadIn} {spoken}", handledBy=f"{self.name}:reveal")

    def _offerAnotherQuestion(self) -> Response:
        """Try a different question, in case the last one was not interesting."""
        question = self._chooseQuestion()
        if question is None:
            return Response(text="", handledBy=f"{self.name}:silent")

        leadIn = self._nextPhrase("nudge", NUDGE_LEAD_INS)
        asked = self._ask(question)
        self._lastReplyAsked = True
        return Response(text=f"{leadIn} {asked}", handledBy=f"{self.name}:question")

    # --- Turns ------------------------------------------------------------

    async def respond(self, text: str, conversation: Conversation) -> Response:
        # The child is back. Whatever was said, they are engaged again, so the
        # count that eventually silences the assistant starts over.
        if text.strip():
            self._unansweredPrompts = 0

        response = await self._reply(text, conversation)
        # Remembered so the next turn knows whether the child has just been
        # asked something. Driving a real conversation, every single reply
        # ended in a question --- the bank's cadence was respected, but the
        # model asked one of its own on every turn in between, which is
        # exactly the interrogation this mode is meant to avoid.
        self._lastReplyAsked = "?" in response.text
        return response

    async def _reply(self, text: str, conversation: Conversation) -> Response:
        cleaned = text.strip()
        if not cleaned:
            return Response(text="", handledBy=self.name)

        if self._awaitingName:
            return await self._handleNameReply(cleaned, conversation)

        if self._pending is not None:
            judged = self._judge(self._pending, cleaned)
            if judged is not None:
                return judged
            # They changed the subject rather than answering. Pressing a
            # three-year-old for the answer they have already moved on from is
            # how the whole thing stops being fun.
            logger.debug("Dropping unanswered question %r", self._pending.id)
            self._pending = None

        return await self._converse(cleaned, conversation)

    # --- The name ---------------------------------------------------------

    async def _handleNameReply(self, text: str, conversation: Conversation) -> Response:
        name = extractName(text)
        if name is None:
            self._awaitingName = False
            # Asking twice is worse than carrying on without it. Questions
            # needing a name stay held back until someone volunteers one.
            logger.debug("No name in %r; continuing without one", text)
            return await self._converse(text, conversation)

        self._childName = name
        self._awaitingName = False
        self._turnsSinceQuestion = 0

        greeting = f"Hello {name}! That's a lovely name."
        question = self._chooseQuestion()
        if question is None:
            return Response(text=greeting, handledBy=self.name)
        return Response(text=f"{greeting} {self._ask(question)}", handledBy=self.name)

    # --- Judging an answer ------------------------------------------------

    def _judge(self, question: Question, utterance: str) -> Response | None:
        """Mark an answer, or return None if it was not an attempt at one."""
        # "Can you name something red?" has no single right answer. Marking a
        # child wrong for saying "fire engine" would be absurd, so anything
        # they offer is accepted.
        if question.isOpen:
            return self._settle(question, correct=True)

        facts = self._facts()
        expected = [
            filled
            for answer in question.answers
            if (filled := render(answer, facts)) is not None
        ]

        if any(containsAnswer(utterance, answer) for answer in expected):
            return self._settle(question, correct=True)

        if not _looksLikeAnAnswer(utterance):
            return None

        return self._settle(question, correct=False)

    def _settle(self, question: Question, *, correct: bool) -> Response:
        """Say whether the answer was right and put the question away."""
        self._pending = None
        self._turnsSinceQuestion = 0
        self._recordOutcome(question, correct=correct)

        phrases = question.correct if correct else question.incorrect
        spoken = self._nextPhrase(f"{question.id}:{correct}", phrases)

        # A phrase whose placeholders cannot be filled would otherwise be
        # spoken with braces in it. Questions are only asked when they can be
        # filled, so this is a guard against a file edited mid-session.
        filled = render(spoken, self._facts())
        if filled is None:
            filled = "That's right!" if correct else "Not quite."

        verdict = "correct" if correct else "incorrect"
        logger.debug("Answer to %r judged %s", question.id, verdict)
        return Response(text=filled, handledBy=f"{self.name}:{verdict}")

    def _nextPhrase(self, key: str, phrases: tuple[str, ...]) -> str:
        """Rotate through the available phrasings so replies vary."""
        if not phrases:
            return ""
        index = self._phraseCounts.get(key, 0)
        self._phraseCounts[key] = index + 1
        return phrases[index % len(phrases)]

    # --- Conversation -----------------------------------------------------

    async def _converse(self, text: str, conversation: Conversation) -> Response:
        """Reply, and offer a question if the moment is right."""
        question = self._chooseQuestion() if self._shouldAsk(text) else None

        spoken = await self._say(text, conversation, appendingQuestion=question is not None)

        if question is None:
            self._turnsSinceQuestion += 1
            if not spoken:
                spoken = self._nextAcknowledgement()
            return Response(text=spoken, usedLlm=self.usesLlm, handledBy=self.name)

        asked = self._ask(question)
        combined = f"{_terminated(spoken)} {asked}" if spoken else asked
        return Response(text=combined, usedLlm=self.usesLlm, handledBy=f"{self.name}:question")

    def _shouldAsk(self, utterance: str) -> bool:
        """Whether to offer a question this turn.

        Contentless utterances are the clearest invitation: a child who says
        "hello" and nothing else is waiting to be given something.
        """
        if self._pending is not None:
            return False
        if normalise(utterance) in CONTENTLESS:
            return True
        return self._turnsSinceQuestion >= self._bank.askEveryTurns

    def _chooseQuestion(self) -> Question | None:
        """The next question to ask, pitched at how the child is doing.

        Each topic keeps its own level. A child who names animal sounds
        reliably is moved on to what a cow eats and where a fish lives, while
        still being asked easy colours if colours are where they struggle.
        Ordering, in decreasing priority:

        1. Questions not yet asked, so the bank is worked through.
        2. Difficulty at or below the child's level for that topic --- being
           stretched is good, being baffled is not.
        3. The hardest of those, which is what makes the level mean anything.
        4. A different topic from the last question, for variety.
        5. The file's own order, so a parent can still sequence them.
        """
        candidates = self._bank.askable(self._facts())
        if not candidates:
            return None

        order = {question.id: index for index, question in enumerate(self._bank.questions)}

        def rank(question: Question) -> tuple[int, int, int, int, int]:
            level = self.levelFor(question.topic)
            return (
                self._askCounts.get(question.id, 0),
                1 if question.difficulty > level else 0,
                # Negated so the hardest question the child can handle wins.
                -question.difficulty if question.difficulty <= level else question.difficulty,
                1 if question.topic == self._lastTopic else 0,
                order[question.id],
            )

        return min(candidates, key=rank)

    # --- How the child is doing -------------------------------------------

    def levelFor(self, topic: str) -> int:
        """The difficulty this child is being offered in ``topic``."""
        return self._topicLevels.get(topic, MINIMUM_DIFFICULTY)

    def _recordOutcome(self, question: Question, *, correct: bool) -> None:
        """Move a topic's level after an answer.

        Two right in a row moves up, two wrong moves down. A staircase rather
        than a single answer, because a three-year-old guesses, and one lucky
        "moo" should not promote them to arithmetic.
        """
        topic = question.topic
        streak = self._topicStreaks.get(topic, 0)
        # A run in the other direction resets rather than accumulating.
        streak = max(1, streak + 1) if correct else min(-1, streak - 1)
        self._topicStreaks[topic] = streak

        level = self.levelFor(topic)
        if streak >= PROMOTION_STREAK and level < MAXIMUM_DIFFICULTY:
            self._topicLevels[topic] = level + 1
            self._topicStreaks[topic] = 0
            logger.info("Moving %s up to difficulty %d", topic, level + 1)
        elif streak <= -PROMOTION_STREAK and level > MINIMUM_DIFFICULTY:
            self._topicLevels[topic] = level - 1
            self._topicStreaks[topic] = 0
            logger.info("Moving %s back to difficulty %d", topic, level - 1)

    def _ask(self, question: Question) -> str:
        """Record that a question is being asked, and render it."""
        asked = render(question.ask, self._facts())
        if asked is None:  # pragma: no cover - askable() already checked
            return ""

        self._pending = question
        self._askCounts[question.id] = self._askCounts.get(question.id, 0) + 1
        self._turnsSinceQuestion = 0
        self._lastTopic = question.topic
        logger.debug(
            "Asking %r (%s, difficulty %d)", question.id, question.topic, question.difficulty
        )
        return asked

    # --- The model --------------------------------------------------------

    async def _say(
        self, text: str, conversation: Conversation, *, appendingQuestion: bool
    ) -> str:
        """The conversational part of the reply.

        Empty is a valid answer: when a question is being appended, a bare
        question is often better than a filler sentence in front of it.
        """
        if self._llm is None:
            return "" if appendingQuestion else self._nextAcknowledgement()

        # A question is allowed only when the last reply did not carry one, so
        # the child is never asked something two turns running by the model.
        # A question from the bank is a separate decision, made on its own
        # cadence, and overrides this either way.
        noQuestions = appendingQuestion or self._lastReplyAsked

        messages = self._buildMessages(
            text,
            conversation,
            appendingQuestion=appendingQuestion,
            allowQuestion=not noQuestions,
        )

        try:
            result = await self._llm.generate(messages)
        except Exception as error:  # noqa: BLE001 - a child should not hear a traceback
            logger.warning("Toddler reply fell back to a fixed phrase: %s", error)
            return "" if appendingQuestion else self._nextAcknowledgement()

        trimmed = _trim(result.text, dropQuestions=noQuestions)
        # Trimming can empty a reply that was nothing but a question. When
        # nothing is being appended, something still has to be said.
        if not trimmed and not appendingQuestion:
            return self._nextAcknowledgement()
        return trimmed

    def _buildMessages(
        self,
        text: str,
        conversation: Conversation,
        *,
        appendingQuestion: bool,
        allowQuestion: bool,
    ) -> list[dict[str, str]]:
        """The prompt, the context, the history, and what was just said."""
        note = self._contextNote(
            appendingQuestion=appendingQuestion, allowQuestion=allowQuestion
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(assistant=self._assistantName)},
            {"role": "system", "content": note},
        ]
        messages.extend(
            {"role": turn.role.value, "content": turn.text} for turn in conversation.turns
        )
        messages.append({"role": "user", "content": text})
        return messages

    def _contextNote(self, *, appendingQuestion: bool, allowQuestion: bool) -> str:
        """Everything the model needs to hold the thread of the conversation."""
        lines = ["What you know right now:"]

        if self._childName:
            lines.append(f"- You are talking to {self._childName}.")
        else:
            lines.append("- You do not know the child's name yet.")

        if self._childAge is not None:
            lines.append(f"- {self._childName or 'The child'} is {self._childAge} years old.")

        answered = [
            question for question in self._bank.questions if self._askCounts.get(question.id)
        ]
        if answered:
            facts = self._facts()
            asked = ", ".join(
                render(question.ask, facts) or question.id for question in answered[-4:]
            )
            lines.append(f"- You have already asked: {asked}")

        if appendingQuestion:
            lines.append(
                "- A question will be added to the end of your reply automatically. "
                "So do NOT ask a question yourself. Just say one short friendly "
                "sentence about what the child said."
            )
        elif not allowQuestion:
            lines.append(
                "- You asked the child a question on your last turn, so do NOT ask "
                "another one now. Just say one short friendly sentence about what "
                "they said."
            )
        else:
            lines.append(
                "- Reply to what the child said. You may ask one simple question "
                "back if it keeps the conversation going."
            )

        return "\n".join(lines)

    def _nextAcknowledgement(self) -> str:
        phrase = ACKNOWLEDGEMENTS[self._acknowledgementIndex % len(ACKNOWLEDGEMENTS)]
        self._acknowledgementIndex += 1
        return phrase

    # --- Helpers ----------------------------------------------------------

    def _facts(self) -> dict[str, str]:
        return self._bank.facts(self._childName, self._childAge)

    def describe(self) -> str:
        parts = [self._bank.describe()]
        if self._llm is not None:
            parts.append(self._llm.describe())
        return "; ".join(parts)


def extractName(text: str) -> str | None:
    """Pull a name out of however a small child answers "what's your name?".

    Handles "Sam", "I'm Sam" and "my name is Sam". Returns None when the
    reply plainly is not a name, so that the assistant carries on rather than
    deciding the child is called "dunno".
    """
    for pattern in NAME_PATTERNS:
        match = pattern.search(text)
        if match:
            candidate = match.group("name")
            if _isPlausibleName(candidate):
                return candidate.capitalize()

    # A bare answer: one or two words, the first of which is the name.
    words = normalise(text).split()
    if 1 <= len(words) <= 2 and _isPlausibleName(words[0]):
        return words[0].capitalize()

    return None


def _isPlausibleName(candidate: str) -> bool:
    return (
        candidate.isalpha()
        and len(candidate) >= 2
        and candidate.lower() not in NOT_NAMES
    )


def _terminated(text: str) -> str:
    """End a sentence properly before another is joined to it.

    A model that replies "hi there little one" with no full stop would
    otherwise run straight into the question: "hi there little one What sound
    does a sheep make?", which the speech synthesiser reads as one breath.
    """
    stripped = text.strip()
    if not stripped or stripped[-1] in ".!?,:;":
        return stripped
    return f"{stripped}."


def _looksLikeAnAnswer(utterance: str) -> bool:
    """Whether an utterance was an attempt at the pending question.

    Short and not itself a question. A child who replies to "what sound does a
    sheep make?" with "where is the dog?" has moved on, and marking that wrong
    would be both unkind and wrong.
    """
    if utterance.strip().endswith("?"):
        return False
    return len(normalise(utterance).split()) <= 5


def _trim(text: str, *, dropQuestions: bool = False) -> str:
    """Strip anything unspeakable and cut the reply down to size.

    ``dropQuestions`` removes any question the model asked, and is used when a
    question from the bank is about to be appended. The system prompt asks for
    the same thing, but a three-billion-parameter model does not reliably
    comply: driving a real conversation produced both "what's your favourite
    colour? what sound does a cow make?" and the same question asked twice,
    once by the model guessing it and once by the append. Only one question
    per reply is a promise to a three-year-old, so it is kept here rather than
    requested in a prompt.
    """
    from app.assistant.llmHandler import sanitiseForSpeech

    cleaned = sanitiseForSpeech(text)
    if not cleaned:
        return ""

    sentences = splitIntoSentences(cleaned, minimumCharacters=1)

    if dropQuestions:
        # One sentence in front of the question, at most: the reply is spoken
        # aloud to a small child, and the question is the part that matters.
        sentences = [sentence for sentence in sentences if not sentence.rstrip().endswith("?")]
        return sentences[0].strip() if sentences else ""

    return " ".join(sentences[:MAXIMUM_SENTENCES]).strip()
