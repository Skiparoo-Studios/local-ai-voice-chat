"""Conversation history.

Kept in memory only. The brief requires persistent storage to be opt-in, so
nothing here writes to disk.

History is bounded, because Stage 4 sends it to a language model with a finite
context window and an unbounded list would eventually fail or cost more per
turn than it is worth.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

DEFAULT_MAX_TURNS = 20


class Role(StrEnum):
    """Who produced a turn."""

    user = "user"
    assistant = "assistant"
    system = "system"


@dataclass(frozen=True, slots=True)
class Turn:
    """One thing said, by either party."""

    role: Role
    text: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def toDict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.text,
            "timestamp": self.timestamp.isoformat(),
        }


class Conversation:
    """An ordered, bounded history of turns."""

    def __init__(self, maxTurns: int = DEFAULT_MAX_TURNS, systemPrompt: str | None = None) -> None:
        if maxTurns <= 0:
            raise ValueError(f"maxTurns must be positive, got {maxTurns}")
        self._turns: deque[Turn] = deque(maxlen=maxTurns)
        self._systemPrompt = systemPrompt

    @property
    def maxTurns(self) -> int:
        return self._turns.maxlen or DEFAULT_MAX_TURNS

    @property
    def systemPrompt(self) -> str | None:
        return self._systemPrompt

    @property
    def turns(self) -> tuple[Turn, ...]:
        return tuple(self._turns)

    def __len__(self) -> int:
        return len(self._turns)

    def add(self, role: Role, text: str) -> Turn:
        """Append a turn, discarding the oldest if the history is full."""
        turn = Turn(role=role, text=text)
        self._turns.append(turn)
        return turn

    def addUser(self, text: str) -> Turn:
        return self.add(Role.user, text)

    def addAssistant(self, text: str) -> Turn:
        return self.add(Role.assistant, text)

    @property
    def lastUserTurn(self) -> Turn | None:
        return self._lastWithRole(Role.user)

    @property
    def lastAssistantTurn(self) -> Turn | None:
        return self._lastWithRole(Role.assistant)

    def _lastWithRole(self, role: Role) -> Turn | None:
        for turn in reversed(self._turns):
            if turn.role is role:
                return turn
        return None

    def clear(self) -> None:
        self._turns.clear()

    def toMessages(self) -> list[dict[str, str]]:
        """Render as the message list language-model providers expect.

        Placed here rather than in a provider so that no provider-specific
        shape leaks into the runtime.
        """
        messages: list[dict[str, str]] = []
        if self._systemPrompt:
            messages.append({"role": Role.system.value, "content": self._systemPrompt})
        messages.extend({"role": turn.role.value, "content": turn.text} for turn in self._turns)
        return messages

    def describe(self) -> str:
        return f"{len(self._turns)} turn(s), keeping the most recent {self.maxTurns}"
