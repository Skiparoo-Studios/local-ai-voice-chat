"""The question bank: matching a small child's answers, and round-tripping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.assistant.questionBank import (
    Question,
    QuestionBank,
    QuestionBankError,
    containsAnswer,
    normalise,
    render,
    slugify,
)


class TestNormalising:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Baa!", "baa"),
            ("  BAA  ", "baa"),
            ("He says, baa.", "he says baa"),
            ("three", "3"),
            ("I am three years old", "i am 3 years old"),
        ],
    )
    def testUtterancesReduceToComparableWords(self, text, expected):
        assert normalise(text) == expected


class TestMatchingAnswers:
    @pytest.mark.parametrize(
        "utterance",
        ["baa", "Baa!", "the sheep says baa", "baa baa black sheep"],
    )
    def testAnswerIsFoundInsideASentence(self, utterance):
        """Children answer in sentences, not single words."""
        assert containsAnswer(utterance, "baa")

    def testWordBoundariesAreRespected(self):
        """Otherwise 'bar' matches 'barn' and everything is correct."""
        assert not containsAnswer("the sheep is in the barn", "bar")

    def testSpelledNumbersMatchDigits(self):
        """Whisper writes 'three' as often as '3'."""
        assert containsAnswer("I am three", "3")
        assert containsAnswer("I am 3", "three")

    def testAnEmptyAnswerMatchesNothing(self):
        assert not containsAnswer("anything at all", "")


class TestPlaceholders:
    def testFilledFromWhatIsKnown(self):
        assert render("How old is {name}?", {"name": "Sam", "age": "3"}) == "How old is Sam?"

    def testUnknownPlaceholderHoldsTheQuestionBack(self):
        """A question about a name nobody has said is not asked at all."""
        assert render("How old is {name}?", {"name": "", "age": "3"}) is None

    def testAQuestionKnowsWhatItNeeds(self):
        question = Question(
            id="how-old",
            ask="How old is {name}?",
            answers=("{age}",),
            correct=("{name} is {age}!",),
        )

        assert question.placeholders == frozenset({"name", "age"})
        assert not question.canBeAsked({"name": "Sam", "age": ""})
        assert question.canBeAsked({"name": "Sam", "age": "3"})

    def testPlaceholdersInAnswersCountToo(self):
        """The prompt alone is not enough: the reply has to be sayable."""
        question = Question(id="q", ask="How old are you?", answers=("{age}",))

        assert question.placeholders == frozenset({"age"})


class TestLoading:
    def testAMissingFileGivesAnEmptyBank(self, tmp_path):
        """Toddler mode still talks; it just has nothing prepared."""
        bank = QuestionBank.load(tmp_path / "absent.json")

        assert len(bank) == 0
        assert bank.path == tmp_path / "absent.json"

    def testTheShippedFileLoads(self):
        bank = QuestionBank.load(Path("config/toddler.json"))

        assert len(bank) > 0
        assert bank.find("sheep-sound") is not None

    def testASingleStringIsAcceptedForAnswers(self):
        question = Question.fromDict({"ask": "What sound does a cow make?", "answers": "moo"})

        assert question.answers == ("moo",)

    def testAnIdIsDerivedFromTheQuestion(self):
        question = Question.fromDict({"ask": "What colour is the sky?"})

        assert question.id == "what-colour-is-the-sky"

    def testBadJsonSaysWhichFile(self, tmp_path):
        path = tmp_path / "toddler.json"
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(QuestionBankError, match="not valid JSON"):
            QuestionBank.load(path)

    def testAQuestionWithNoTextIsRejected(self, tmp_path):
        path = tmp_path / "toddler.json"
        path.write_text(json.dumps({"questions": [{"answers": ["baa"]}]}), encoding="utf-8")

        with pytest.raises(QuestionBankError, match="no 'ask' text"):
            QuestionBank.load(path)

    def testANonNumericAgeIsRejected(self, tmp_path):
        path = tmp_path / "toddler.json"
        path.write_text(json.dumps({"childAge": "nearly four"}), encoding="utf-8")

        with pytest.raises(QuestionBankError, match="whole number"):
            QuestionBank.load(path)

    def testAByteOrderMarkIsTolerated(self, tmp_path):
        """Notepad and PowerShell both write one."""
        path = tmp_path / "toddler.json"
        path.write_text(json.dumps({"childName": "Sam"}), encoding="utf-8-sig")

        assert QuestionBank.load(path).childName == "Sam"


class TestSaving:
    def testEditsSurviveARoundTrip(self, tmp_path):
        path = tmp_path / "toddler.json"
        bank = QuestionBank(path=path, childName="Sam", childAge=3)
        bank.add(Question(id="pig", ask="What sound does a pig make?", answers=("oink",)))
        bank.save()

        reloaded = QuestionBank.load(path)

        assert reloaded.childName == "Sam"
        assert reloaded.childAge == 3
        assert reloaded.find("pig") is not None

    def testTheDirectoryIsCreated(self, tmp_path):
        path = tmp_path / "nested" / "toddler.json"

        QuestionBank(path=path).save()

        assert path.is_file()


class TestEditing:
    def testAddingReplacesAQuestionWithTheSameId(self):
        bank = QuestionBank()
        bank.add(Question(id="pig", ask="What does a pig say?", answers=("oink",)))
        bank.add(Question(id="pig", ask="What does a pig say?", answers=("oink", "onk")))

        assert len(bank) == 1
        assert bank.find("pig").answers == ("oink", "onk")

    def testAQuestionIsFoundByItsTextAsWellAsItsId(self):
        bank = QuestionBank([Question(id="pig", ask="What does a pig say?")])

        assert bank.find("What does a pig say?") is not None
        assert bank.find("what does a pig say") is not None

    def testRemovingReportsWhatWentAndWhatDidNot(self):
        bank = QuestionBank([Question(id="pig", ask="What does a pig say?")])

        assert bank.remove("pig").id == "pig"
        assert bank.remove("pig") is None

    def testSlugsAreStable(self):
        assert slugify("What sound does a sheep make?") == "what-sound-does-a-sheep"
