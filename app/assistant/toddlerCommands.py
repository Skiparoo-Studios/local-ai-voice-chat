"""Editing the toddler question bank from the ordinary prompt.

The person who adds questions is a parent at a keyboard, not the child, so the
editing commands belong to normal mode rather than toddler mode. They are kept
out of the REPL itself so that they can be tested without one, and so that the
same parsing could be reached from anywhere else that wants it.

Every command that changes the bank writes it straight back to disk. Losing a
question because the assistant was closed without a save step would be a poor
trade for the one line it takes to avoid.
"""

from __future__ import annotations

import logging

from app.assistant.handlers import ResponseHandler
from app.assistant.modeSwitch import ModeSwitchingHandler
from app.assistant.questionBank import (
    MAXIMUM_DIFFICULTY,
    MINIMUM_DIFFICULTY,
    Question,
    QuestionBankError,
    slugify,
)

logger = logging.getLogger(__name__)

USAGE = """
  :toddler                        show the mode and the questions
  :toddler on | off               enter or leave toddler mode
  :toddler name <name>            who the assistant is talking to
  :toddler age <number>           how old they are
  :toddler every <number>         ordinary turns between questions
  :toddler add <question> | <answers> | <correct> | <incorrect> | <topic> | <1-5>
  :toddler remove <id or question>

  Answers are separated by commas; everything after them is optional. Leave
  the answers empty for a question where anything counts, such as "can you
  name something red?". The last field is difficulty, 1 to 5: what a child is
  offered climbs within a topic as they get things right.
  {name} and {age} in any field are filled in when the question is asked.

  For example:
    :toddler add What sound does a pig make? | oink, oik | Yes, a pig oinks!
    :toddler add What is a baby cow called? | calf | Yes! | A calf. | animals | 4
""".rstrip()


def runToddlerCommand(handler: ResponseHandler | None, argument: str) -> list[str]:
    """Run one ``:toddler`` command, returning the lines to show.

    Returning lines rather than printing keeps this testable and lets a caller
    other than the terminal use it.
    """
    if not isinstance(handler, ModeSwitchingHandler):
        return [
            "  toddler mode is not available.",
            "  Set toddler.allowModePhrases to true, or start with --toddler.",
        ]

    command, _, rest = argument.strip().partition(" ")
    command = command.lower()
    rest = rest.strip()

    if not command:
        return _summarise(handler)
    if command == "on":
        return [f"  {handler.enterToddlerMode().text}"]
    if command == "off":
        return [f"  {handler.leaveToddlerMode().text}"]
    if command == "name":
        return _setName(handler, rest)
    if command == "age":
        return _setAge(handler, rest)
    if command == "every":
        return _setCadence(handler, rest)
    if command == "add":
        return _add(handler, rest)
    if command in {"remove", "delete"}:
        return _remove(handler, rest)

    return [f"  unknown toddler command {command!r}", USAGE]


# --- Reading -------------------------------------------------------------


def _summarise(handler: ModeSwitchingHandler) -> list[str]:
    bank = handler.toddler.bank
    where = "toddler mode" if handler.inToddlerMode else "normal mode"

    lines = [
        f"  currently in {where}",
        f"  child:  {bank.childName or 'not set'}"
        + (f", aged {bank.childAge}" if bank.childAge is not None else ""),
        f"  asking: one question every {bank.askEveryTurns} ordinary turn(s)",
        f"  file:   {bank.path or 'unsaved'}",
    ]

    if not len(bank):
        lines.append("  no questions yet; :toddler add to write one")
        return lines

    lines.append(f"  {len(bank)} question(s), by topic:")

    grouped: dict[str, list] = {}
    for question in bank.questions:
        grouped.setdefault(question.topic, []).append(question)

    for topic, questions in grouped.items():
        # The level is where this child has got to in this topic, which is
        # session state rather than anything in the file.
        level = handler.toddler.levelFor(topic)
        lines.append(f"    {topic}  ({len(questions)} question(s), now asking level {level})")
        for question in sorted(questions, key=lambda item: item.difficulty):
            answers = ", ".join(question.answers) if question.answers else "anything"
            lines.append(
                f"      {question.difficulty}  [{question.id}] {question.ask}  ->  {answers}"
            )

    return lines


# --- Writing -------------------------------------------------------------


def _setName(handler: ModeSwitchingHandler, value: str) -> list[str]:
    if not value:
        return ["  usage: :toddler name <name>"]

    name = value.strip().split()[0].capitalize()
    handler.toddler.bank.childName = name
    # Applied to the running session as well, so a question about the name
    # becomes askable without leaving and re-entering the mode.
    handler.toddler.reset()
    return _persist(handler, f"child's name set to {name}")


def _setAge(handler: ModeSwitchingHandler, value: str) -> list[str]:
    try:
        age = int(value.strip())
    except ValueError:
        return ["  usage: :toddler age <number>"]

    if not 0 <= age <= 18:
        return ["  that does not look like a child's age"]

    handler.toddler.bank.childAge = age
    handler.toddler.reset()
    return _persist(handler, f"age set to {age}")


def _setCadence(handler: ModeSwitchingHandler, value: str) -> list[str]:
    try:
        turns = int(value.strip())
    except ValueError:
        return ["  usage: :toddler every <number>"]

    if turns < 0:
        return ["  that must be zero or more"]

    handler.toddler.bank.askEveryTurns = turns
    if turns == 0:
        return _persist(handler, "a question will be offered every turn")
    return _persist(handler, f"a question every {turns} ordinary turn(s)")


def _add(handler: ModeSwitchingHandler, rest: str) -> list[str]:
    """Parse ``question | answers | correct | incorrect`` and store it."""
    if not rest:
        return ["  usage: :toddler add <question> | <answers> | <correct> | <incorrect>"]

    fields = [field.strip() for field in rest.split("|")]
    ask = fields[0]
    if not ask:
        return ["  the question cannot be empty"]

    answers = tuple(
        answer.strip() for answer in fields[1].split(",") if answer.strip()
    ) if len(fields) > 1 else ()

    correct = (fields[2],) if len(fields) > 2 and fields[2] else ("That's right!",)
    incorrect = (fields[3],) if len(fields) > 3 and fields[3] else ("Not quite.",)
    topic = fields[4] if len(fields) > 4 and fields[4] else "custom"

    difficulty = MINIMUM_DIFFICULTY
    if len(fields) > 5 and fields[5]:
        try:
            difficulty = int(fields[5])
        except ValueError:
            return [
                f"  difficulty must be a number from "
                f"{MINIMUM_DIFFICULTY} to {MAXIMUM_DIFFICULTY}"
            ]
        difficulty = max(MINIMUM_DIFFICULTY, min(MAXIMUM_DIFFICULTY, difficulty))

    question = Question(
        id=slugify(ask),
        ask=ask,
        answers=answers,
        correct=correct,
        incorrect=incorrect,
        topic=topic,
        difficulty=difficulty,
    )

    existed = handler.toddler.bank.find(question.ask) is not None
    handler.toddler.bank.add(question)
    verb = "replaced" if existed else "added"
    return _persist(handler, f"{verb} [{question.id}] {ask}")


def _remove(handler: ModeSwitchingHandler, rest: str) -> list[str]:
    if not rest:
        return ["  usage: :toddler remove <id or question>"]

    removed = handler.toddler.bank.remove(rest)
    if removed is None:
        return [f"  no question matching {rest!r}"]
    return _persist(handler, f"removed [{removed.id}] {removed.ask}")


def _persist(handler: ModeSwitchingHandler, message: str) -> list[str]:
    """Write the bank back, reporting where it went or why it could not."""
    try:
        path = handler.toddler.bank.save()
    except (OSError, QuestionBankError) as error:
        logger.error("Could not save the question bank: %s", error)
        return [f"  {message}, but saving failed: {error}"]
    return [f"  {message}", f"  saved to {path}"]
