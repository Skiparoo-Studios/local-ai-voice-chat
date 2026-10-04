"""Editing the question bank from the ordinary prompt."""

from __future__ import annotations

from app.assistant.handlers import EchoHandler
from app.assistant.modeSwitch import ModeSwitchingHandler
from app.assistant.questionBank import Question, QuestionBank
from app.assistant.toddlerCommands import runToddlerCommand
from app.assistant.toddlerHandler import ToddlerHandler

SHEEP = Question(
    id="sheep",
    ask="What sound does a sheep make?",
    answers=("baa",),
    correct=("That's correct!",),
)


def buildSwitch(tmp_path, questions=(SHEEP,)):
    bank = QuestionBank(questions=list(questions), path=tmp_path / "toddler.json")
    return ModeSwitchingHandler(EchoHandler(), ToddlerHandler(bank))


def joined(lines: list[str]) -> str:
    return "\n".join(lines)


class TestReading:
    def testTheBankIsListed(self, tmp_path):
        switch = buildSwitch(tmp_path)

        output = joined(runToddlerCommand(switch, ""))

        assert "normal mode" in output
        assert "What sound does a sheep make?" in output
        assert "[sheep]" in output

    def testQuestionsAreGroupedByTopicWithTheirDifficulty(self, tmp_path):
        switch = buildSwitch(
            tmp_path,
            questions=(
                Question(id="a", ask="Animal one?", topic="animals", difficulty=1),
                Question(id="b", ask="Animal two?", topic="animals", difficulty=4),
                Question(id="c", ask="Colour one?", topic="colours", difficulty=2),
            ),
        )

        output = joined(runToddlerCommand(switch, ""))

        assert "animals  (2 question(s), now asking level 1)" in output
        assert "colours  (1 question(s), now asking level 1)" in output
        # Easiest first within a topic, which is the order they are offered in.
        assert output.index("Animal one?") < output.index("Animal two?")

    def testTheLevelShownFollowsHowTheChildIsDoing(self, tmp_path):
        switch = buildSwitch(
            tmp_path, questions=(Question(id="a", ask="Animal?", topic="animals"),)
        )
        switch.toddler._topicLevels["animals"] = 3

        assert "now asking level 3" in joined(runToddlerCommand(switch, ""))

    def testAnEmptyBankSaysSo(self, tmp_path):
        switch = buildSwitch(tmp_path, questions=())

        assert "no questions yet" in joined(runToddlerCommand(switch, ""))

    def testAnOpenQuestionIsShownAsAcceptingAnything(self, tmp_path):
        switch = buildSwitch(tmp_path, questions=(Question(id="q", ask="What did you play?"),))

        assert "anything" in joined(runToddlerCommand(switch, ""))


class TestSwitchingMode:
    def testModeCanBeChangedFromThePrompt(self, tmp_path):
        switch = buildSwitch(tmp_path)

        runToddlerCommand(switch, "on")
        assert switch.inToddlerMode

        runToddlerCommand(switch, "off")
        assert not switch.inToddlerMode


class TestEditing:
    def testAQuestionIsAddedAndSaved(self, tmp_path):
        switch = buildSwitch(tmp_path)

        output = joined(runToddlerCommand(switch, "add What sound does a pig make? | oink, oik"))

        assert "added" in output
        assert (tmp_path / "toddler.json").is_file()

        reloaded = QuestionBank.load(tmp_path / "toddler.json")
        added = reloaded.find("What sound does a pig make?")
        assert added is not None
        assert added.answers == ("oink", "oik")

    def testTheRepliesCanBeGivenToo(self, tmp_path):
        switch = buildSwitch(tmp_path)

        runToddlerCommand(
            switch, "add What does a pig say? | oink | Yes, pigs oink! | A pig says oink."
        )

        added = QuestionBank.load(tmp_path / "toddler.json").find("What does a pig say?")
        assert added.correct == ("Yes, pigs oink!",)
        assert added.incorrect == ("A pig says oink.",)

    def testAddingTheSameQuestionReplacesIt(self, tmp_path):
        switch = buildSwitch(tmp_path)

        runToddlerCommand(switch, "add What sound does a sheep make? | baa, bar")
        output = joined(runToddlerCommand(switch, "add What sound does a sheep make? | baa"))

        assert "replaced" in output
        assert len(QuestionBank.load(tmp_path / "toddler.json")) == 1

    def testTheTopicAndDifficultyCanBeGiven(self, tmp_path):
        switch = buildSwitch(tmp_path)

        runToddlerCommand(
            switch, "add What is a baby cow called? | calf | Yes! | A calf. | animals | 4"
        )

        added = QuestionBank.load(tmp_path / "toddler.json").find("What is a baby cow called?")
        assert added.topic == "animals"
        assert added.difficulty == 4

    def testAnOutOfRangeDifficultyIsClamped(self, tmp_path):
        switch = buildSwitch(tmp_path)

        runToddlerCommand(switch, "add Hard one? | yes | Yes! | No. | custom | 9")

        assert QuestionBank.load(tmp_path / "toddler.json").find("Hard one?").difficulty == 5

    def testANonNumericDifficultyIsRefused(self, tmp_path):
        switch = buildSwitch(tmp_path)

        output = joined(runToddlerCommand(switch, "add Q? | a | Yes! | No. | custom | hard"))

        assert "difficulty must be a number" in output

    def testANewQuestionDefaultsToTheEasiestLevel(self, tmp_path):
        switch = buildSwitch(tmp_path)

        runToddlerCommand(switch, "add What sound does a pig make? | oink")

        added = QuestionBank.load(tmp_path / "toddler.json").find("What sound does a pig make?")
        assert added.difficulty == 1
        assert added.topic == "custom"

    def testAQuestionIsRemovedByIdOrText(self, tmp_path):
        switch = buildSwitch(tmp_path)

        output = joined(runToddlerCommand(switch, "remove What sound does a sheep make?"))

        assert "removed" in output
        assert len(QuestionBank.load(tmp_path / "toddler.json")) == 0

    def testRemovingSomethingAbsentSaysSo(self, tmp_path):
        switch = buildSwitch(tmp_path)

        assert "no question matching" in joined(runToddlerCommand(switch, "remove giraffe"))

    def testTheNameIsSetAndSaved(self, tmp_path):
        switch = buildSwitch(tmp_path)

        runToddlerCommand(switch, "name sam")

        assert QuestionBank.load(tmp_path / "toddler.json").childName == "Sam"
        # Applied to the running session too, without re-entering the mode.
        assert switch.toddler.childName == "Sam"

    def testTheAgeIsSetAndSaved(self, tmp_path):
        switch = buildSwitch(tmp_path)

        runToddlerCommand(switch, "age 3")

        assert QuestionBank.load(tmp_path / "toddler.json").childAge == 3

    def testAnImplausibleAgeIsRefused(self, tmp_path):
        switch = buildSwitch(tmp_path)

        assert "does not look like" in joined(runToddlerCommand(switch, "age 97"))

    def testTheCadenceIsSetAndSaved(self, tmp_path):
        switch = buildSwitch(tmp_path)

        runToddlerCommand(switch, "every 4")

        assert QuestionBank.load(tmp_path / "toddler.json").askEveryTurns == 4


class TestMisuse:
    def testUsageIsShownForAnUnknownCommand(self, tmp_path):
        switch = buildSwitch(tmp_path)

        output = joined(runToddlerCommand(switch, "frobnicate"))

        assert "unknown toddler command" in output
        assert ":toddler add" in output

    def testUsageIsShownForAnIncompleteAdd(self, tmp_path):
        switch = buildSwitch(tmp_path)

        assert "usage" in joined(runToddlerCommand(switch, "add"))

    def testANonNumericAgeIsRefused(self, tmp_path):
        switch = buildSwitch(tmp_path)

        assert "usage" in joined(runToddlerCommand(switch, "age nearly four"))

    def testAHandlerWithoutToddlerModeExplainsItself(self):
        output = joined(runToddlerCommand(EchoHandler(), ""))

        assert "not available" in output
        assert "--toddler" in output
