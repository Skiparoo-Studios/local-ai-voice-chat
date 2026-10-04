"""Conversation history and sessions."""

from __future__ import annotations

import pytest

from app.assistant.conversation import Conversation, Role, Turn
from app.assistant.session import Session, SessionRegistry
from app.events import AssistantState


class TestTurns:
    def testAddingTurnsPreservesOrder(self):
        conversation = Conversation()

        conversation.addUser("turn the light off")
        conversation.addAssistant("Done.")

        assert [turn.text for turn in conversation.turns] == ["turn the light off", "Done."]
        assert [turn.role for turn in conversation.turns] == [Role.user, Role.assistant]

    def testLengthCountsTurns(self):
        conversation = Conversation()
        conversation.addUser("one")
        conversation.addAssistant("two")

        assert len(conversation) == 2

    def testClearRemovesEverything(self):
        conversation = Conversation()
        conversation.addUser("one")

        conversation.clear()

        assert len(conversation) == 0

    def testLastTurnsByRole(self):
        conversation = Conversation()
        conversation.addUser("first")
        conversation.addAssistant("reply")
        conversation.addUser("second")

        assert conversation.lastUserTurn.text == "second"
        assert conversation.lastAssistantTurn.text == "reply"

    def testLastTurnsAreNoneWhenEmpty(self):
        conversation = Conversation()

        assert conversation.lastUserTurn is None
        assert conversation.lastAssistantTurn is None

    def testTurnsAreImmutable(self):
        turn = Turn(role=Role.user, text="hello")

        with pytest.raises((AttributeError, TypeError)):
            turn.text = "changed"


class TestBounding:
    """History is bounded so Stage 4 cannot overflow a context window."""

    def testOldestTurnsAreDiscarded(self):
        conversation = Conversation(maxTurns=3)

        for index in range(5):
            conversation.addUser(str(index))

        assert [turn.text for turn in conversation.turns] == ["2", "3", "4"]

    def testMaxTurnsMustBePositive(self):
        with pytest.raises(ValueError, match="maxTurns"):
            Conversation(maxTurns=0)

    def testDescriptionMentionsTheLimit(self):
        conversation = Conversation(maxTurns=5)
        conversation.addUser("one")

        assert "1 turn" in conversation.describe()
        assert "5" in conversation.describe()


class TestMessageRendering:
    """The runtime must not know a provider's message shape."""

    def testRendersRoleAndContent(self):
        conversation = Conversation()
        conversation.addUser("hello")
        conversation.addAssistant("hi")

        assert conversation.toMessages() == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

    def testSystemPromptLeadsTheMessages(self):
        conversation = Conversation(systemPrompt="You are helpful.")
        conversation.addUser("hello")

        messages = conversation.toMessages()

        assert messages[0] == {"role": "system", "content": "You are helpful."}
        assert len(messages) == 2

    def testNoSystemPromptMeansNoSystemMessage(self):
        conversation = Conversation()
        conversation.addUser("hello")

        assert all(message["role"] != "system" for message in conversation.toMessages())


class TestSession:
    def testSessionsGetDistinctIdentifiers(self):
        assert Session.create().sessionId != Session.create().sessionId

    def testSessionStartsIdle(self):
        assert Session.create().state is AssistantState.idle

    def testSessionCarriesItsConversation(self):
        session = Session.create(maxTurns=5)

        session.conversation.addUser("hello")

        assert len(session.conversation) == 1
        assert session.conversation.maxTurns == 5

    def testSystemPromptReachesTheConversation(self):
        session = Session.create(systemPrompt="Be brief.")

        assert session.conversation.systemPrompt == "Be brief."


class TestSessionRegistry:
    def testCreatedSessionsAreRetrievable(self):
        registry = SessionRegistry()
        session = registry.create()

        assert registry.get(session.sessionId) is session
        assert len(registry) == 1

    def testUnknownSessionIsNone(self):
        assert SessionRegistry().get("nope") is None

    def testSessionsCanBeRemoved(self):
        registry = SessionRegistry()
        session = registry.create()

        registry.remove(session.sessionId)

        assert registry.get(session.sessionId) is None

    def testRemovingAnUnknownSessionIsHarmless(self):
        SessionRegistry().remove("nope")
