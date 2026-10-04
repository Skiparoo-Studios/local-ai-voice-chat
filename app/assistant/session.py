"""Assistant sessions.

A session is one conversational context: its history, and the state a client
displays. Stage 3 uses exactly one, but sessions are identified from the start
because Stage 7 serves several clients at once and Stage 8 gives each remote
device its own.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.assistant.conversation import DEFAULT_MAX_TURNS, Conversation
from app.events import AssistantState


def newSessionId() -> str:
    return uuid.uuid4().hex[:12]


@dataclass(slots=True)
class Session:
    """One conversational context."""

    sessionId: str = field(default_factory=newSessionId)
    conversation: Conversation = field(default_factory=Conversation)
    state: AssistantState = AssistantState.idle
    startedAt: datetime = field(default_factory=lambda: datetime.now(UTC))
    turnCount: int = 0

    @classmethod
    def create(
        cls,
        maxTurns: int = DEFAULT_MAX_TURNS,
        systemPrompt: str | None = None,
    ) -> Session:
        return cls(conversation=Conversation(maxTurns=maxTurns, systemPrompt=systemPrompt))

    @property
    def ageSeconds(self) -> float:
        return (datetime.now(UTC) - self.startedAt).total_seconds()

    def describe(self) -> str:
        return f"session {self.sessionId} ({self.turnCount} turns, {self.ageSeconds:.0f}s old)"


class SessionRegistry:
    """Sessions by id.

    Trivial now. It exists so that the runtime never assumes a single session,
    which is what would otherwise need unpicking in Stage 7.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(
        self,
        maxTurns: int = DEFAULT_MAX_TURNS,
        systemPrompt: str | None = None,
    ) -> Session:
        session = Session.create(maxTurns=maxTurns, systemPrompt=systemPrompt)
        self._sessions[session.sessionId] = session
        return session

    def get(self, sessionId: str) -> Session | None:
        return self._sessions.get(sessionId)

    def remove(self, sessionId: str) -> None:
        self._sessions.pop(sessionId, None)

    def __len__(self) -> int:
        return len(self._sessions)

    @property
    def sessionIds(self) -> tuple[str, ...]:
        return tuple(self._sessions)
