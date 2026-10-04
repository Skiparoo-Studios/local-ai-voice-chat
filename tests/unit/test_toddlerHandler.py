"""Toddler mode: judging answers, and the rhythm of asking them."""

from __future__ import annotations

import pytest

from app.assistant.conversation import Conversation
from app.assistant.questionBank import (
    MAXIMUM_DIFFICULTY,
    MINIMUM_DIFFICULTY,
    Question,
    QuestionBank,
)
from app.assistant.toddlerHandler import ToddlerHandler, extractName
from app.intelligence.llmProvider import LlmProvider
from app.intelligence.providers.scriptedLlmProvider import ScriptedLlmProvider

SHEEP = Question(
    id="sheep",
    ask="What sound does a sheep make?",
    answers=("baa", "bar"),
    correct=("That's correct! A sheep says baa.",),
    incorrect=("A sheep makes a baa sound.",),
)

HOW_OLD = Question(
    id="how-old",
    ask="How old is {name}?",
    answers=("{age}",),
    correct=("That's right, {name} is {age}!",),
    incorrect=("{name} is actually {age}.",),
)

TEETH = Question(
    id="teeth",
    ask="Did {name} brush his teeth today?",
    answers=("yes", "yeah", "i did"),
    correct=("Good job buddy, that will keep the germs away.",),
    incorrect=("Let's go and brush them.",),
)


def buildHandler(
    questions: tuple[Question, ...] = (SHEEP,),
    *,
    childName: str | None = "Sam",
    childAge: int | None = 3,
    askEveryTurns: int = 2,
    llm: LlmProvider | None = None,
    promptAfterSeconds: float = 7.0,
    giveUpAfterPrompts: int = 3,
) -> ToddlerHandler:
    bank = QuestionBank(
        questions=list(questions),
        childName=childName,
        childAge=childAge,
        askEveryTurns=askEveryTurns,
    )
    return ToddlerHandler(
        bank,
        llm=llm,
        assistantName="Assistant",
        promptAfterSeconds=promptAfterSeconds,
        giveUpAfterPrompts=giveUpAfterPrompts,
    )


class CapturingProvider(ScriptedLlmProvider):
    """Records the messages it was sent, so the context can be inspected."""

    def __init__(self, replies=("Okay!",)) -> None:
        super().__init__(replies=replies)
        self.messages: list[list[dict]] = []

    async def generate(self, messages, tools=None):
        self.messages.append(messages)
        return await super().generate(messages, tools)


class FailingProvider(ScriptedLlmProvider):
    async def generate(self, messages, tools=None):
        raise RuntimeError("the model service is down")


class TestJudgingAnswers:
    async def testACorrectAnswerIsPraisedWithoutTheModel(self):
        """A three-year-old saying 'baa' should not wait for a model."""
        handler = buildHandler(llm=ScriptedLlmProvider())
        await handler.respond("hello", Conversation())
        assert handler.pendingQuestion is SHEEP

        response = await handler.respond("baa", Conversation())

        assert response.text == "That's correct! A sheep says baa."
        assert not response.usedLlm
        assert handler.pendingQuestion is None

    async def testAnAnswerInsideASentenceCounts(self):
        handler = buildHandler()
        await handler.respond("hello", Conversation())

        response = await handler.respond("the sheep says baa", Conversation())

        assert "correct" in response.text.lower()

    async def testAWrongAnswerIsCorrectedGently(self):
        handler = buildHandler()
        await handler.respond("hello", Conversation())

        response = await handler.respond("moo", Conversation())

        assert response.text == "A sheep makes a baa sound."
        assert handler.pendingQuestion is None

    async def testChangingTheSubjectIsNotMarkedWrong(self):
        """Pressing a child for an answer they have moved on from is how this
        stops being fun."""
        handler = buildHandler()
        await handler.respond("hello", Conversation())

        response = await handler.respond("where is the doggy?", Conversation())

        assert "baa" not in response.text
        assert handler.pendingQuestion is None

    async def testPlaceholdersAreFilledInThePraise(self):
        handler = buildHandler((HOW_OLD,), childName="Sam", childAge=3)
        await handler.respond("hello", Conversation())

        response = await handler.respond("three", Conversation())

        assert response.text == "That's right, Sam is 3!"

    async def testTheCorrectionNamesTheChildToo(self):
        handler = buildHandler((HOW_OLD,), childName="Sam", childAge=3)
        await handler.respond("hello", Conversation())

        response = await handler.respond("seven", Conversation())

        assert response.text == "Sam is actually 3."

    async def testAPersonalYesNoQuestionWorks(self):
        handler = buildHandler((TEETH,), childName="Sam")
        opening = await handler.respond("hello", Conversation())
        assert opening.text == "Did Sam brush his teeth today?"

        response = await handler.respond("yes", Conversation())

        assert response.text == "Good job buddy, that will keep the germs away."


class TestAskingRhythm:
    async def testAContentlessUtteranceIsAnsweredWithAQuestion(self):
        """'Hello' with nothing after it is an invitation to be given something."""
        handler = buildHandler()

        response = await handler.respond("hello", Conversation())

        assert response.text.endswith("What sound does a sheep make?")

    async def testTheChildIsNotBombardedWithQuestions(self):
        handler = buildHandler((SHEEP, HOW_OLD, TEETH), askEveryTurns=2)

        first = await handler.respond("I saw a big red bus today", Conversation())
        second = await handler.respond("it went past the shop", Conversation())

        assert "?" not in first.text
        assert "?" not in second.text

    async def testAQuestionArrivesAfterTheConfiguredNumberOfTurns(self):
        handler = buildHandler((SHEEP,), askEveryTurns=2)

        await handler.respond("I saw a big red bus today", Conversation())
        await handler.respond("it went past the shop", Conversation())
        third = await handler.respond("and then we went home", Conversation())

        assert third.text.endswith("What sound does a sheep make?")

    async def testOnlyOneQuestionIsEverAsked(self):
        handler = buildHandler((SHEEP, HOW_OLD, TEETH))

        response = await handler.respond("hello", Conversation())

        assert response.text.count("?") == 1

    async def testQuestionsAreNotRepeatedWhileOthersRemain(self):
        handler = buildHandler((SHEEP, HOW_OLD), askEveryTurns=0)

        first = await handler.respond("hello", Conversation())
        await handler.respond("baa", Conversation())
        second = await handler.respond("hello", Conversation())

        assert first.text != second.text

    async def testAnEmptyBankSimplyNeverAsks(self):
        handler = buildHandler(())

        response = await handler.respond("hello", Conversation())

        assert response.text
        assert "?" not in response.text


def easy(topic: str, index: int, difficulty: int) -> Question:
    """A question of a given topic and difficulty, answered with its own id."""
    return Question(
        id=f"{topic}-{index}",
        ask=f"{topic} question {index}?",
        answers=(f"answer{index}",),
        correct=("Yes!",),
        incorrect=("No.",),
        topic=topic,
        difficulty=difficulty,
    )


class TestAdaptiveDifficulty:
    """A child who names animal sounds reliably should be moved on to what a
    cow eats, while still being asked easy colours if colours are hard."""

    async def testTheEasiestIsAskedFirst(self):
        handler = buildHandler((easy("animals", 5, 5), easy("animals", 1, 1)), askEveryTurns=0)

        response = await handler.respond("hello", Conversation())

        assert "animals question 1?" in response.text

    async def testTwoRightAnswersOfferHarderQuestions(self):
        questions = (easy("animals", 1, 1), easy("animals", 2, 1), easy("animals", 3, 2))
        handler = buildHandler(questions, askEveryTurns=0)

        for index in (1, 2):
            await handler.respond("hello", Conversation())
            await handler.respond(f"answer{index}", Conversation())

        assert handler.levelFor("animals") == 2

    async def testOneLuckyGuessIsNotMastery(self):
        """A three-year-old guesses. One right answer must not promote them."""
        handler = buildHandler((easy("animals", 1, 1), easy("animals", 2, 2)), askEveryTurns=0)

        await handler.respond("hello", Conversation())
        await handler.respond("answer1", Conversation())

        assert handler.levelFor("animals") == 1

    async def testTwoWrongAnswersOfferEasierQuestions(self):
        questions = tuple(easy("counting", index, 3) for index in range(1, 4))
        handler = buildHandler(questions, askEveryTurns=0)
        handler._topicLevels["counting"] = 3

        for _ in range(2):
            await handler.respond("hello", Conversation())
            await handler.respond("purple", Conversation())

        assert handler.levelFor("counting") == 2

    async def testARightAnswerBreaksARunOfWrongOnes(self):
        questions = tuple(easy("animals", index, 1) for index in range(1, 5))
        handler = buildHandler(questions, askEveryTurns=0)
        handler._topicLevels["animals"] = 3

        await handler.respond("hello", Conversation())
        await handler.respond("purple", Conversation())
        await handler.respond("hello", Conversation())
        await handler.respond("answer2", Conversation())

        assert handler.levelFor("animals") == 3

    async def testTopicsAreTrackedSeparately(self):
        """Being good at animals says nothing about being good at counting.

        Only animals are offered, so the run of right answers all lands in one
        topic --- otherwise the variety rule interleaves them and neither topic
        gets two in a row.
        """
        questions = (easy("animals", 1, 1), easy("animals", 2, 1), easy("animals", 3, 2))
        handler = buildHandler(questions, askEveryTurns=0)

        for index in (1, 2):
            await handler.respond("hello", Conversation())
            await handler.respond(f"answer{index}", Conversation())

        assert handler.levelFor("animals") == 2
        assert handler.levelFor("counting") == MINIMUM_DIFFICULTY

    async def testNothingHarderThanTheLevelIsOffered(self):
        handler = buildHandler((easy("animals", 9, 5), easy("animals", 1, 1)), askEveryTurns=0)

        first = await handler.respond("hello", Conversation())

        assert "question 1?" in first.text

    async def testAQuestionTooHardIsUsedOnlyWhenNothingElseRemains(self):
        handler = buildHandler((easy("animals", 9, 5),), askEveryTurns=0)

        response = await handler.respond("hello", Conversation())

        assert "question 9?" in response.text

    async def testTheLevelStopsAtTheTop(self):
        questions = tuple(easy("animals", index, 5) for index in range(1, 21))
        handler = buildHandler(questions, askEveryTurns=0)
        handler._topicLevels["animals"] = MAXIMUM_DIFFICULTY

        for index in range(1, 11):
            await handler.respond("hello", Conversation())
            await handler.respond(f"answer{index}", Conversation())

        assert handler.levelFor("animals") == MAXIMUM_DIFFICULTY

    async def testTheLevelStopsAtTheBottom(self):
        questions = tuple(easy("animals", index, 1) for index in range(1, 21))
        handler = buildHandler(questions, askEveryTurns=0)

        for _ in range(10):
            await handler.respond("hello", Conversation())
            await handler.respond("purple", Conversation())

        assert handler.levelFor("animals") == MINIMUM_DIFFICULTY

    async def testTopicsVaryBetweenConsecutiveQuestions(self):
        handler = buildHandler(
            (easy("animals", 1, 1), easy("animals", 2, 1), easy("colours", 3, 1)),
            askEveryTurns=0,
        )

        first = await handler.respond("hello", Conversation())
        await handler.respond("answer1", Conversation())
        second = await handler.respond("hello", Conversation())

        assert "animals" in first.text
        assert "colours" in second.text

    async def testProgressIsForgottenOnReset(self):
        handler = buildHandler((easy("animals", 1, 1),), askEveryTurns=0)
        handler._topicLevels["animals"] = 4

        handler.reset()

        assert handler.levelFor("animals") == MINIMUM_DIFFICULTY


class TestOpenQuestions:
    """Some questions have no single right answer."""

    async def testAnythingIsAcceptedWhenThereAreNoAnswers(self):
        question = Question(
            id="red-thing",
            ask="Can you name something red?",
            answers=(),
            correct=("Good thinking!",),
            incorrect=("Not quite.",),
        )
        handler = buildHandler((question,))

        await handler.respond("hello", Conversation())
        response = await handler.respond("a fire engine", Conversation())

        assert response.text == "Good thinking!"

    async def testAnOpenQuestionCountsTowardsProgress(self):
        question = lambda index: Question(  # noqa: E731
            id=f"open-{index}", ask=f"Open {index}?", answers=(), topic="reasoning"
        )
        handler = buildHandler((question(1), question(2)), askEveryTurns=0)

        for _ in range(2):
            await handler.respond("hello", Conversation())
            await handler.respond("something", Conversation())

        assert handler.levelFor("reasoning") == 2

    async def testTheShippedOpenQuestionsAcceptAnything(self):
        from pathlib import Path

        bank = QuestionBank.load(Path("config/toddler.json"))
        openQuestions = [question for question in bank.questions if question.isOpen]

        assert openQuestions
        for question in openQuestions:
            assert question.correct == question.incorrect, question.id


class TestTheChildsName:
    @pytest.mark.parametrize(
        "reply,expected",
        [
            ("Sam", "Sam"),
            ("sam", "Sam"),
            ("My name is Sam", "Sam"),
            ("I'm Sam", "Sam"),
            ("it's Sam", "Sam"),
            ("call me Sam", "Sam"),
        ],
    )
    def testANameIsFoundHoweverItIsGiven(self, reply, expected):
        assert extractName(reply) == expected

    @pytest.mark.parametrize("reply", ["I don't know", "no", "um", "yes please", "3"])
    def testANonAnswerIsNotTakenAsAName(self, reply):
        """Otherwise the assistant decides the child is called 'dunno'."""
        assert extractName(reply) is None

    async def testTheNameIsAskedForWhenNoneIsKnown(self):
        handler = buildHandler(childName=None)

        assert "name" in handler.openingLine().lower()

    async def testGivingANameUnlocksThePersonalQuestions(self):
        handler = buildHandler((HOW_OLD,), childName=None, childAge=3)
        handler.openingLine()

        response = await handler.respond("Sam", Conversation())

        assert handler.childName == "Sam"
        assert "How old is Sam?" in response.text

    async def testAQuestionNeedingANameIsHeldBackUntilThereIsOne(self):
        handler = buildHandler((HOW_OLD,), childName=None, childAge=3)

        response = await handler.respond("hello", Conversation())

        assert "How old" not in response.text

    async def testAKnownNameIsGreetedRatherThanAskedFor(self):
        handler = buildHandler(childName="Sam")

        assert "Sam" in handler.openingLine()
        assert "your name" not in handler.openingLine().lower()

    async def testAnUnrecognisedReplyDoesNotStallTheConversation(self):
        """Asking twice is worse than carrying on without a name."""
        handler = buildHandler(childName=None)
        handler.openingLine()

        response = await handler.respond("I don't know", Conversation())

        assert response.text
        assert "what's your name" not in response.text.lower()


class TestTheModel:
    async def testTheModelIsToldNotToAskWhenAQuestionIsBeingAppended(self):
        """Otherwise the child gets two questions at once."""
        provider = CapturingProvider()
        handler = buildHandler(llm=provider)

        await handler.respond("hello", Conversation())

        system = " ".join(
            message["content"]
            for message in provider.messages[0]
            if message["role"] == "system"
        )
        assert "do NOT ask a question yourself" in system

    async def testTheModelIsGivenTheChildsNameAndAge(self):
        provider = CapturingProvider()
        handler = buildHandler(childName="Sam", childAge=3, llm=provider)

        await handler.respond("I saw a bus", Conversation())

        system = " ".join(
            message["content"]
            for message in provider.messages[0]
            if message["role"] == "system"
        )
        assert "Sam" in system
        assert "3 years old" in system

    async def testTheModelIsToldToKeepItSimple(self):
        provider = CapturingProvider()
        handler = buildHandler(llm=provider)

        await handler.respond("I saw a bus", Conversation())

        system = provider.messages[0][0]["content"]
        assert "three years old" in system
        assert "ONE short sentence" in system

    async def testHistoryIsPassedThrough(self):
        provider = CapturingProvider()
        handler = buildHandler(llm=provider)
        conversation = Conversation()
        conversation.addUser("I have a red car")
        conversation.addAssistant("A red car! How lovely.")

        await handler.respond("it is fast", conversation)

        contents = [message["content"] for message in provider.messages[0]]
        assert "I have a red car" in contents

    async def testAQuestionFromTheModelIsRemovedWhenOneIsBeingAppended(self):
        """qwen2.5:3b ignores the instruction not to ask. Driving a real
        conversation produced "Hello there! What's your favorite color? What
        sound does a cow make?" --- two questions at a three-year-old."""
        provider = ScriptedLlmProvider(
            replies=("Hello there! What's your favourite colour?",)
        )
        handler = buildHandler((SHEEP,), llm=provider)

        response = await handler.respond("hello", Conversation())

        assert response.text == "Hello there! What sound does a sheep make?"

    async def testTheModelGuessingTheQuestionDoesNotDuplicateIt(self):
        """Also seen for real: the model asked the same question the bank was
        about to append, and the child heard it twice in a row."""
        provider = ScriptedLlmProvider(
            replies=("Great adventure! What sound does a sheep make?",)
        )
        handler = buildHandler((SHEEP,), llm=provider)

        response = await handler.respond("hello", Conversation())

        assert response.text.count("What sound does a sheep make?") == 1

    async def testAModelReplyOfNothingButAQuestionLeavesJustTheBankQuestion(self):
        provider = ScriptedLlmProvider(replies=("What is your favourite colour?",))
        handler = buildHandler((SHEEP,), llm=provider)

        response = await handler.respond("hello", Conversation())

        assert response.text == "What sound does a sheep make?"

    async def testTheModelMayStillAskWhenNoQuestionIsAppended(self):
        """Stripping every question would make the conversation one-sided."""
        provider = ScriptedLlmProvider(replies=("A bus! What colour was it?",))
        handler = buildHandler((), llm=provider)

        response = await handler.respond("I saw a bus", Conversation())

        assert "What colour was it?" in response.text

    async def testTheModelMayNotAskTwoTurnsRunning(self):
        """Driving a real conversation, every single reply ended in a question:
        the bank's cadence was respected, but the model asked one of its own on
        every turn in between."""
        provider = ScriptedLlmProvider(
            replies=("A bus! What colour was it?", "How lovely! Where did it go?")
        )
        handler = buildHandler((), llm=provider)

        first = await handler.respond("I saw a bus", Conversation())
        second = await handler.respond("it was red", Conversation())

        assert first.text.endswith("?")
        assert "?" not in second.text

    async def testTheModelMayAskAgainOnceATurnHasPassedWithout(self):
        provider = ScriptedLlmProvider(
            replies=("A bus! What colour was it?", "How lovely.", "Where did it go?")
        )
        handler = buildHandler((), llm=provider)

        await handler.respond("I saw a bus", Conversation())
        await handler.respond("it was red", Conversation())
        third = await handler.respond("to the shops", Conversation())

        assert third.text.endswith("?")

    async def testABankQuestionIsStillAskedAfterOneFromTheModel(self):
        """The bank's cadence is the primary control and is not overridden."""
        provider = ScriptedLlmProvider(replies=("A bus! What colour was it?", "How lovely."))
        handler = buildHandler((SHEEP,), askEveryTurns=1, llm=provider)

        await handler.respond("I saw a bus", Conversation())
        second = await handler.respond("it was red", Conversation())

        assert second.text.endswith("What sound does a sheep make?")

    async def testAnUnpunctuatedLeadInIsClosedBeforeTheQuestion(self):
        """Seen for real: "hi there little one What sound does a sheep make?",
        which the synthesiser reads as a single breath."""
        provider = ScriptedLlmProvider(replies=("hi there little one",))
        handler = buildHandler((SHEEP,), llm=provider)

        response = await handler.respond("hello", Conversation())

        assert response.text == "hi there little one. What sound does a sheep make?"

    async def testAnAlreadyPunctuatedLeadInIsLeftAlone(self):
        provider = ScriptedLlmProvider(replies=("Hello there!",))
        handler = buildHandler((SHEEP,), llm=provider)

        response = await handler.respond("hello", Conversation())

        assert response.text == "Hello there! What sound does a sheep make?"

    async def testALongReplyIsCutShort(self):
        """A model that ignores the instruction is trimmed, not obeyed."""
        provider = ScriptedLlmProvider(
            replies=("One sentence here. And a second one. And a third one too. A fourth.",)
        )
        handler = buildHandler((), llm=provider)

        response = await handler.respond("I saw a bus", Conversation())

        assert response.text.count(".") <= 2

    async def testMarkdownNeverReachesTheSpeaker(self):
        provider = ScriptedLlmProvider(replies=("**Wow!** That is a `big` bus.",))
        handler = buildHandler((), llm=provider)

        response = await handler.respond("I saw a bus", Conversation())

        assert "*" not in response.text
        assert "`" not in response.text

    async def testAFailingModelFallsBackToAFixedPhrase(self):
        """A child should never hear a traceback, or silence."""
        handler = buildHandler((), llm=FailingProvider())

        response = await handler.respond("I saw a bus", Conversation())

        assert response.text

    async def testAFailingModelStillDeliversTheQuestion(self):
        handler = buildHandler((SHEEP,), llm=FailingProvider())

        response = await handler.respond("hello", Conversation())

        assert "What sound does a sheep make?" in response.text


class TestWithoutAModel:
    async def testQuestionsAndAnswersWorkWithNoModelAtAll(self):
        """The whole point of Layer 1 thinking, applied here."""
        handler = buildHandler()

        asked = await handler.respond("hello", Conversation())
        answered = await handler.respond("baa", Conversation())

        assert asked.text == "What sound does a sheep make?"
        assert answered.text == "That's correct! A sheep says baa."
        assert not answered.usedLlm

    async def testAnOrdinaryRemarkGetsAnAcknowledgement(self):
        handler = buildHandler()

        response = await handler.respond("I saw a big red bus today", Conversation())

        assert response.text


class TestFillingASilence:
    """A toddler who says nothing is not finished, they are thinking --- until
    they have wandered off, at which point the assistant should stop."""

    async def testTheAnswerIsRevealedWhenNobodyAttemptsIt(self):
        handler = buildHandler()
        await handler.respond("hello", Conversation())
        assert handler.pendingQuestion is SHEEP

        prompt = await handler.promptWhenSilent()

        assert "A sheep makes a baa sound" in prompt.text
        assert handler.pendingQuestion is None

    async def testTheRevealDoesNotImplyTheChildGotItWrong(self):
        """They said nothing at all; "not quite" would be unfair."""
        handler = buildHandler()
        await handler.respond("hello", Conversation())

        prompt = await handler.promptWhenSilent()

        assert not prompt.text.lower().startswith("not quite")
        assert prompt.text.startswith(("Shall I tell you?", "I know this one!", "Let me help you."))

    async def testAnotherQuestionIsOfferedWhenNoneIsPending(self):
        handler = buildHandler((SHEEP, HOW_OLD))
        await handler.respond("hello", Conversation())
        await handler.respond("baa", Conversation())
        assert handler.pendingQuestion is None

        prompt = await handler.promptWhenSilent()

        assert prompt.text.endswith("?")
        assert handler.pendingQuestion is not None

    async def testRevealIsFollowedByAnotherQuestion(self):
        """Reveal, then try something else, rather than repeating one question."""
        handler = buildHandler((SHEEP, HOW_OLD), giveUpAfterPrompts=5)
        await handler.respond("hello", Conversation())

        first = await handler.promptWhenSilent()
        second = await handler.promptWhenSilent()

        assert "baa" in first.text
        assert "How old is Sam?" in second.text

    async def testItGivesUpAfterEnoughSilence(self):
        handler = buildHandler((SHEEP, HOW_OLD, TEETH), giveUpAfterPrompts=3)
        await handler.respond("hello", Conversation())

        for _ in range(3):
            assert await handler.promptWhenSilent() is not None

        assert await handler.promptWhenSilent() is None
        assert handler.promptAfterSeconds is None

    async def testItStopsAskingOnceItHasGivenUp(self):
        """An assistant still asking questions of an empty room is worse
        company than one that has stopped."""
        handler = buildHandler(giveUpAfterPrompts=1)
        await handler.promptWhenSilent()

        assert handler.promptAfterSeconds is None

    async def testTheChildComingBackStartsItOverAgain(self):
        handler = buildHandler((SHEEP, HOW_OLD), giveUpAfterPrompts=2)
        await handler.promptWhenSilent()
        await handler.promptWhenSilent()
        assert handler.promptAfterSeconds is None

        await handler.respond("hello", Conversation())

        assert handler.promptAfterSeconds == 7.0
        assert await handler.promptWhenSilent() is not None

    async def testAnEmptyBankPromptsNothing(self):
        handler = buildHandler(())

        assert (await handler.promptWhenSilent()).isEmpty

    async def testWaitingForANameIsNotRevealed(self):
        """Nobody else can say what the child is called."""
        handler = buildHandler((SHEEP,), childName=None)
        handler.openingLine()

        prompt = await handler.promptWhenSilent()

        assert "name" not in prompt.text.lower()

    async def testTheDelayIsConfigurable(self):
        assert buildHandler(promptAfterSeconds=12.0).promptAfterSeconds == 12.0

    async def testPromptingCanBeTurnedOff(self):
        handler = buildHandler(promptAfterSeconds=0.0)

        assert handler.promptAfterSeconds is None

    async def testAPromptDoesNotCountAsTheChildSpeaking(self):
        """Otherwise the give-up count would never reach its limit."""
        handler = buildHandler((SHEEP, HOW_OLD, TEETH), giveUpAfterPrompts=2)

        await handler.promptWhenSilent()
        await handler.promptWhenSilent()

        assert await handler.promptWhenSilent() is None

    async def testOrdinaryHandlersNeverSpeakUnprompted(self):
        """The default is silence: an assistant talking to an empty room is a
        worse assistant."""
        from app.assistant.handlers import EchoHandler, RuleHandler

        for handler in (EchoHandler(), RuleHandler()):
            assert handler.promptAfterSeconds is None
            assert await handler.promptWhenSilent() is None


class TestSessionState:
    async def testResetForgetsTheSessionButKeepsTheFile(self):
        handler = buildHandler(childName="Sam")
        await handler.respond("hello", Conversation())
        assert handler.pendingQuestion is not None

        handler.reset()

        assert handler.pendingQuestion is None
        assert handler.childName == "Sam"

    async def testResetForgetsANameGivenByVoice(self):
        handler = buildHandler(childName=None)
        handler.openingLine()
        await handler.respond("Sam", Conversation())
        assert handler.childName == "Sam"

        handler.reset()

        assert handler.childName is None

    async def testEmptyInputProducesNoReply(self):
        handler = buildHandler()

        assert (await handler.respond("   ", Conversation())).isEmpty

    def testItDescribesItself(self):
        handler = buildHandler()

        assert "1 question(s)" in handler.describe()
        assert "Sam" in handler.describe()
