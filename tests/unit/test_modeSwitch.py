"""Entering and leaving toddler mode by phrase, from any mode."""

from __future__ import annotations

import pytest

from app.assistant.conversation import Conversation
from app.assistant.handlers import EchoHandler, createResponseHandler
from app.assistant.modeSwitch import ModeSwitchingHandler
from app.assistant.questionBank import Question, QuestionBank
from app.assistant.toddlerHandler import ToddlerHandler
from app.config import Settings

SHEEP = Question(
    id="sheep",
    ask="What sound does a sheep make?",
    answers=("baa",),
    correct=("That's correct!",),
    incorrect=("A sheep says baa.",),
)


def buildSwitch(*, startInToddlerMode: bool = False, allowPhrases: bool = True):
    bank = QuestionBank(questions=[SHEEP], childName="Sam", childAge=3)
    toddler = ToddlerHandler(bank, assistantName="Assistant")
    return ModeSwitchingHandler(
        EchoHandler(),
        toddler,
        startInToddlerMode=startInToddlerMode,
        allowPhrases=allowPhrases,
    )


class TestSwitchingByPhrase:
    @pytest.mark.parametrize(
        "phrase",
        [
            "start toddler mode",
            "Start toddler mode",
            "begin toddler mode",
            "enter toddler mode",
            "switch to toddler mode",
            "toddler mode on",
            "start toddler mode.",
            # What speech recognition actually produces on a bad day.
            "start toddle mode",
        ],
    )
    async def testEnteringIsRecognised(self, phrase):
        switch = buildSwitch()

        await switch.respond(phrase, Conversation())

        assert switch.inToddlerMode

    @pytest.mark.parametrize(
        "phrase",
        [
            "end toddler mode",
            "stop toddler mode",
            "exit toddler mode",
            "toddler mode off",
            "normal mode",
            "back to normal mode",
        ],
    )
    async def testLeavingIsRecognised(self, phrase):
        switch = buildSwitch(startInToddlerMode=True)

        await switch.respond(phrase, Conversation())

        assert not switch.inToddlerMode

    async def testEnteringGreetsTheChild(self):
        switch = buildSwitch()

        response = await switch.respond("start toddler mode", Conversation())

        assert "Sam" in response.text

    async def testLeavingSaysGoodbye(self):
        switch = buildSwitch(startInToddlerMode=True)

        response = await switch.respond("end toddler mode", Conversation())

        assert "Bye" in response.text

    async def testThePhraseNeverReachesTheOrdinaryHandler(self):
        """The echo handler would repeat it; the router might act on it."""
        switch = buildSwitch()

        response = await switch.respond("start toddler mode", Conversation())

        assert not response.text.startswith("You said:")

    async def testPhrasesCanBeTurnedOff(self):
        switch = buildSwitch(allowPhrases=False)

        response = await switch.respond("start toddler mode", Conversation())

        assert not switch.inToddlerMode
        assert response.text.startswith("You said:")

    async def testEnteringTwiceIsHarmless(self):
        switch = buildSwitch(startInToddlerMode=True)

        response = await switch.respond("start toddler mode", Conversation())

        assert switch.inToddlerMode
        assert response.text

    async def testLeavingWhenNotThereIsHarmless(self):
        switch = buildSwitch()

        response = await switch.respond("end toddler mode", Conversation())

        assert not switch.inToddlerMode
        assert response.text

    async def testAMentionInPassingDoesNotSwitch(self):
        """'My toddler mode idea' is conversation, not a command."""
        switch = buildSwitch()

        await switch.respond("I was telling you about toddler mode yesterday", Conversation())

        assert not switch.inToddlerMode


class TestStartingWithTheFlag:
    async def testTheChildIsGreetedOnTheFirstThingTheySay(self):
        """--toddler has nobody to greet at startup, so it waits for them."""
        switch = buildSwitch(startInToddlerMode=True)

        first = await switch.respond("hello", Conversation())

        assert "Sam" in first.text
        assert "lovely to talk to you" in first.text

    async def testTheGreetingHappensOnlyOnce(self):
        switch = buildSwitch(startInToddlerMode=True)

        await switch.respond("hello", Conversation())
        second = await switch.respond("hello", Conversation())

        assert "lovely to talk to you" not in second.text

    async def testTheNameIsAskedForWhenStartingWithTheFlag(self):
        """Without the held greeting this never happened, and every question
        about the child stayed silently out of reach."""
        bank = QuestionBank(questions=[SHEEP], childName=None)
        switch = ModeSwitchingHandler(
            EchoHandler(), ToddlerHandler(bank), startInToddlerMode=True
        )

        first = await switch.respond("hello", Conversation())
        await switch.respond("Sam", Conversation())

        assert "name" in first.text.lower()
        assert switch.toddler.childName == "Sam"

    async def testAnUnheardGreetingIsNotReplayedAfterSwitchingBack(self):
        switch = buildSwitch(startInToddlerMode=True)

        await switch.respond("end toddler mode", Conversation())
        entered = await switch.respond("start toddler mode", Conversation())
        next_ = await switch.respond("hello", Conversation())

        assert "lovely to talk to you" in entered.text
        assert "lovely to talk to you" not in next_.text


class TestDelegation:
    async def testNormalModeUsesTheConfiguredHandler(self):
        switch = buildSwitch()

        response = await switch.respond("hello there", Conversation())

        assert response.text == "You said: hello there"

    async def testToddlerModeUsesTheToddlerHandler(self):
        switch = buildSwitch(startInToddlerMode=True)
        await switch.respond("hello", Conversation())  # the held greeting

        response = await switch.respond("hello", Conversation())

        assert response.text.endswith("What sound does a sheep make?")

    async def testTheSessionIsForgottenOnEntry(self):
        """A second child should not inherit the first one's answers."""
        switch = buildSwitch(startInToddlerMode=True)
        await switch.respond("hello", Conversation())  # the held greeting
        await switch.respond("hello", Conversation())
        assert switch.toddler.pendingQuestion is not None

        await switch.respond("end toddler mode", Conversation())
        await switch.respond("start toddler mode", Conversation())

        assert switch.toddler.pendingQuestion is None

    async def testStreamingFollowsTheActiveHandler(self):
        switch = buildSwitch()

        fragments = [
            fragment
            async for fragment in switch.respondStream("start toddler mode", Conversation())
        ]

        assert switch.inToddlerMode
        assert "".join(fragments)

    def testStreamingSupportIsReportedForTheActiveHandler(self):
        switch = buildSwitch(startInToddlerMode=True)

        # Toddler replies are assembled whole, so there is nothing to stream.
        assert not switch.supportsStreaming


def unwrapped(handler):
    """Look past the sentry layer, which now wraps everything.

    These tests are about which handler configuration chose, not about what
    stands in front of it.
    """
    from app.assistant.sentryHandler import SentryHandler

    return handler.base if isinstance(handler, SentryHandler) else handler


class TestConfiguration:
    def testTheSwitchIsWrappedAroundTheConfiguredHandler(self, tmp_path):
        settings = Settings(
            assistant={"handler": "echo"},
            toddler={"questionsFile": tmp_path / "toddler.json"},
        )

        handler = unwrapped(createResponseHandler(settings))

        assert isinstance(handler, ModeSwitchingHandler)
        assert isinstance(handler.base, EchoHandler)

    def testToddlerFlagStartsInToddlerMode(self, tmp_path):
        settings = Settings(
            assistant={"handler": "echo"},
            toddler={"enabled": True, "questionsFile": tmp_path / "toddler.json"},
        )

        handler = unwrapped(createResponseHandler(settings))

        assert handler.inToddlerMode

    def testNoWrapperWhenBothSwitchesAreOff(self, tmp_path):
        settings = Settings(
            assistant={"handler": "echo"},
            toddler={"allowModePhrases": False, "questionsFile": tmp_path / "toddler.json"},
        )

        handler = unwrapped(createResponseHandler(settings))

        assert isinstance(handler, EchoHandler)

    def testABrokenQuestionsFileDoesNotStopTheAssistant(self, tmp_path, caplog):
        """The parent needs telling, not a machine that will not start."""
        path = tmp_path / "toddler.json"
        path.write_text("{ broken", encoding="utf-8")
        settings = Settings(assistant={"handler": "echo"}, toddler={"questionsFile": path})

        handler = unwrapped(createResponseHandler(settings))

        assert isinstance(handler, EchoHandler)
        assert "Toddler mode disabled" in caplog.text

    def testTheShippedQuestionsFileIsWiredUpByDefault(self):
        handler = unwrapped(createResponseHandler(Settings(assistant={"handler": "echo"})))

        assert isinstance(handler, ModeSwitchingHandler)
        assert len(handler.toddler.bank) > 0
