"""The questions toddler mode asks, and how it judges the answers.

Kept in a JSON file rather than in code because the person who knows what a
particular child is learning this month is the parent, not the programmer. The
file is read at startup and written back whenever it is edited, so a question
added mid-session survives a restart.

Judging is deliberately deterministic. A small child answering "baa" wants to
hear "that's right" immediately, and a model round trip to decide whether "baa"
means "baa" is both slower and less reliable than a list of expected answers.
The model's job in toddler mode is phrasing the conversation around the
questions, not marking them.

Placeholders in braces are filled from what the assistant knows: ``{name}`` and
``{age}``. A question whose placeholders cannot all be filled is simply not
asked, which is what makes "how old is {name}?" wait until someone has said.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_QUESTIONS_PATH = Path("config/toddler.json")

# Toddlers count out loud and Whisper writes what it hears, so "three" and "3"
# both arrive depending on the sentence. Answers are compared with both
# normalised to digits.
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

PLACEHOLDER = re.compile(r"\{(\w+)\}")


class QuestionBankError(Exception):
    """Raised when the questions file is missing or malformed."""


def normalise(text: str) -> str:
    """Reduce an utterance to comparable words.

    Punctuation goes, case goes, and spelled numbers become digits, so that
    "Baa!", "baa" and "he says baa" all compare against the same token, and
    "three" matches an expected answer of "3".
    """
    lowered = text.lower()
    stripped = re.sub(r"[^a-z0-9 ]+", " ", lowered)
    collapsed = re.sub(r"\s+", " ", stripped).strip()
    return " ".join(NUMBER_WORDS.get(word, word) for word in collapsed.split())


def containsAnswer(utterance: str, answer: str) -> bool:
    """Whether ``utterance`` contains ``answer`` as whole words.

    Substring matching rather than equality because children answer in
    sentences: "the sheep says baa" is a correct answer to what sound a sheep
    makes. Word boundaries keep "bar" from matching "barn".
    """
    needle = normalise(answer)
    if not needle:
        return False
    haystack = normalise(utterance)
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


def render(template: str, facts: dict[str, str]) -> str | None:
    """Fill placeholders from ``facts``, or return None if any is unknown.

    Returning None rather than raising is what lets the bank quietly hold back
    a question about a name nobody has said yet.
    """
    missing = [name for name in PLACEHOLDER.findall(template) if not facts.get(name)]
    if missing:
        return None
    return PLACEHOLDER.sub(lambda match: facts[match.group(1)], template)


MINIMUM_DIFFICULTY = 1
MAXIMUM_DIFFICULTY = 5


@dataclass(frozen=True, slots=True)
class Question:
    """Something to ask, what counts as right, and what to say either way."""

    id: str
    ask: str
    answers: tuple[str, ...] = ()
    correct: tuple[str, ...] = ("That's right!",)
    incorrect: tuple[str, ...] = ("Not quite.",)
    topic: str = "general"
    # 1 is a two-year-old naming a colour, 5 is arithmetic or a riddle. What a
    # child is offered climbs within a topic as they get things right, so the
    # numbers only have to be ordered sensibly relative to each other.
    difficulty: int = MINIMUM_DIFFICULTY

    @property
    def isOpen(self) -> bool:
        """Whether anything counts as an answer.

        "Can you name something red?" has no single right answer, and marking a
        child wrong for saying "fire engine" would be absurd.
        """
        return not self.answers

    @property
    def placeholders(self) -> frozenset[str]:
        """Every placeholder used anywhere in the question.

        The whole question is checked, not just the prompt: it is no use asking
        "how old are you?" if the reply would have to say "{name} is {age}".
        """
        parts = (self.ask, *self.answers, *self.correct, *self.incorrect)
        return frozenset(name for part in parts for name in PLACEHOLDER.findall(part))

    def canBeAsked(self, facts: dict[str, str]) -> bool:
        return all(facts.get(name) for name in self.placeholders)

    def toDict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ask": self.ask,
            "answers": list(self.answers),
            "correct": list(self.correct),
            "incorrect": list(self.incorrect),
            "topic": self.topic,
            "difficulty": self.difficulty,
        }

    @classmethod
    def fromDict(cls, data: dict[str, Any]) -> Question:
        ask = str(data.get("ask", "")).strip()
        if not ask:
            raise QuestionBankError(f"A question has no 'ask' text: {data!r}")

        return cls(
            id=str(data.get("id") or slugify(ask)),
            ask=ask,
            answers=_asStrings(data.get("answers"), "answers", ask),
            correct=_asStrings(data.get("correct"), "correct", ask) or ("That's right!",),
            incorrect=_asStrings(data.get("incorrect"), "incorrect", ask) or ("Not quite.",),
            topic=str(data.get("topic") or "general"),
            difficulty=_asDifficulty(data.get("difficulty"), ask),
        )


@dataclass(slots=True)
class QuestionBank:
    """The questions, plus what is known about the child."""

    questions: list[Question] = field(default_factory=list)
    childName: str | None = None
    childAge: int | None = None
    # How many ordinary conversational turns pass before another question is
    # offered. The brief for this mode is explicit that a child should not be
    # interrogated, and this is the dial that controls it.
    askEveryTurns: int = 2
    path: Path | None = None

    # --- Loading and saving ----------------------------------------------

    @classmethod
    def load(cls, path: Path = DEFAULT_QUESTIONS_PATH) -> QuestionBank:
        """Read the bank, or return an empty one if the file does not exist.

        A missing file is not an error: toddler mode still holds a
        conversation, it simply has nothing prepared to ask.
        """
        if not path.is_file():
            logger.info("No questions file at %s; starting with an empty bank", path)
            return cls(path=path)

        try:
            # utf-8-sig for the same reason settings.json uses it: Notepad and
            # PowerShell both write a byte order mark.
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except OSError as error:
            raise QuestionBankError(f"Could not read {path}: {error}") from error
        except json.JSONDecodeError as error:
            raise QuestionBankError(f"{path} is not valid JSON: {error}") from error

        if not isinstance(data, dict):
            raise QuestionBankError(f"{path} must contain a JSON object at the top level")

        entries = data.get("questions", [])
        if not isinstance(entries, list):
            raise QuestionBankError(f"{path}: 'questions' must be a list")

        return cls(
            questions=[Question.fromDict(entry) for entry in entries],
            childName=_asOptionalName(data.get("childName")),
            childAge=_asOptionalAge(data.get("childAge"), path),
            askEveryTurns=max(0, int(data.get("askEveryTurns", 2))),
            path=path,
        )

    def save(self, path: Path | None = None) -> Path:
        """Write the bank back, creating the directory if needed."""
        destination = path or self.path or DEFAULT_QUESTIONS_PATH
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.toDict(), indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.path = destination
        logger.info("Wrote %d question(s) to %s", len(self.questions), destination)
        return destination

    def toDict(self) -> dict[str, Any]:
        return {
            "childName": self.childName,
            "childAge": self.childAge,
            "askEveryTurns": self.askEveryTurns,
            "questions": [question.toDict() for question in self.questions],
        }

    # --- Using the bank ---------------------------------------------------

    def __len__(self) -> int:
        return len(self.questions)

    def facts(self, childName: str | None = None, childAge: int | None = None) -> dict[str, str]:
        """What is known, for filling placeholders.

        The arguments win over the stored values so that a name given by voice
        applies for the session without being written to the file.
        """
        name = childName or self.childName
        age = childAge if childAge is not None else self.childAge
        return {
            "name": name or "",
            "age": str(age) if age is not None else "",
        }

    def askable(self, facts: dict[str, str]) -> list[Question]:
        """Questions every placeholder of which can be filled."""
        return [question for question in self.questions if question.canBeAsked(facts)]

    def find(self, identifier: str) -> Question | None:
        """Look a question up by id, or by its text if that fails."""
        wanted = identifier.strip()
        for question in self.questions:
            if question.id == wanted:
                return question

        normalised = normalise(wanted)
        for question in self.questions:
            if normalise(question.ask) == normalised:
                return question
        return None

    def add(self, question: Question) -> Question:
        """Add a question, replacing any that is already asking the same thing.

        Matched on the wording as well as the id, because a question added by
        hand carries whatever id its author chose. Deduplicating on the id
        alone would let the same question be added twice under two names, and
        the child would then be asked it twice.
        """
        wording = normalise(question.ask)
        self.questions = [
            existing
            for existing in self.questions
            if existing.id != question.id and normalise(existing.ask) != wording
        ]
        self.questions.append(question)
        return question

    def remove(self, identifier: str) -> Question | None:
        """Remove a question by id or text, returning what was removed."""
        found = self.find(identifier)
        if found is None:
            return None
        self.questions = [existing for existing in self.questions if existing.id != found.id]
        return found

    def describe(self) -> str:
        known = self.childName or "nobody yet"
        return f"{len(self.questions)} question(s), for {known}"


def slugify(text: str) -> str:
    """A stable id derived from the question itself."""
    words = normalise(text).split()
    return "-".join(words[:5]) or "question"


def _asStrings(value: object, field_: str, ask: str) -> tuple[str, ...]:
    """Accept either a single string or a list of them."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(item for item in value if item.strip())
    raise QuestionBankError(
        f"'{field_}' for {ask!r} must be a string or a list of strings, got {type(value).__name__}"
    )


def _asDifficulty(value: object, ask: str) -> int:
    """Read a difficulty, clamped rather than rejected.

    A questions file is written by hand by a parent, and a 7 typed where a 5
    was meant should not stop the assistant starting.
    """
    if value is None or value == "":
        return MINIMUM_DIFFICULTY

    try:
        level = int(value)
    except (TypeError, ValueError) as error:
        raise QuestionBankError(
            f"'difficulty' for {ask!r} must be a whole number from "
            f"{MINIMUM_DIFFICULTY} to {MAXIMUM_DIFFICULTY}"
        ) from error

    return max(MINIMUM_DIFFICULTY, min(MAXIMUM_DIFFICULTY, level))


def _asOptionalName(value: object) -> str | None:
    if value is None:
        return None
    name = str(value).strip()
    return name or None


def _asOptionalAge(value: object, path: Path) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise QuestionBankError(f"{path}: 'childAge' must be a whole number") from error
