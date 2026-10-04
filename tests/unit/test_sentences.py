"""Splitting a model's fragment stream into speakable pieces."""

from __future__ import annotations

import pytest

from app.assistant.sentences import SentenceStreamer, splitIntoSentences


def feedAll(streamer: SentenceStreamer, text: str, fragmentSize: int = 3) -> list[str]:
    """Feed text in small fragments, as a model would produce it."""
    pieces: list[str] = []
    for start in range(0, len(text), fragmentSize):
        pieces.extend(streamer.feed(text[start : start + fragmentSize]))
    remainder = streamer.flush()
    if remainder:
        pieces.append(remainder)
    return pieces


class TestSentenceBoundaries:
    def testReleasesAtAFullStop(self):
        streamer = SentenceStreamer()

        pieces = streamer.feed("I have turned the kitchen light off. And then")

        assert pieces == ["I have turned the kitchen light off."]

    def testQuestionAndExclamationEndSentences(self):
        # The minimum length is set aside here so that only boundary detection
        # is under test; the two rules interacting is covered separately.
        assert splitIntoSentences(
            "Is that what you wanted? I think so! Right", minimumCharacters=0
        ) == ["Is that what you wanted?", "I think so!", "Right"]

    def testShortSentencesMergeUnderTheDefaultMinimum(self):
        assert splitIntoSentences("Is that right? Yes it is.") == [
            "Is that right? Yes it is."
        ]

    def testNewlineEndsAPiece(self):
        pieces = splitIntoSentences("The light is now off\nAnything else you need?")

        assert len(pieces) == 2

    def testClosingQuotesStayWithTheSentence(self):
        pieces = splitIntoSentences('She said "the light is off." Then she left.')

        assert pieces[0].endswith('."')

    def testIncompleteTextIsHeldBack(self):
        streamer = SentenceStreamer()

        assert streamer.feed("I have turned the kitchen") == []
        assert streamer.pending == "I have turned the kitchen"


class TestMinimumLength:
    """Very short pieces sound clipped, so they wait for more text."""

    def testShortSentenceWaitsForTheNext(self):
        streamer = SentenceStreamer(minimumCharacters=20)

        pieces = streamer.feed("Yes. The light is now off in the kitchen. ")

        assert pieces == ["Yes. The light is now off in the kitchen."]

    def testShortSentenceIsStillReleasedOnFlush(self):
        streamer = SentenceStreamer(minimumCharacters=20)
        streamer.feed("Yes.")

        assert streamer.flush() == "Yes."

    def testAMinimumOfZeroReleasesEverySentence(self):
        streamer = SentenceStreamer(minimumCharacters=0)

        assert streamer.feed("Yes. No. ") == ["Yes.", "No."]


class TestLengthLimit:
    """A model producing no punctuation must still be spoken."""

    def testOverLongBufferIsSplitOnWhitespace(self):
        streamer = SentenceStreamer(maximumCharacters=50)
        text = " ".join(["word"] * 40)

        pieces = feedAll(streamer, text)

        assert len(pieces) > 1
        assert all(len(piece) <= 50 for piece in pieces)

    def testSplitDoesNotBreakWords(self):
        streamer = SentenceStreamer(maximumCharacters=50)

        pieces = feedAll(streamer, " ".join(["kitchen"] * 30))

        assert all(word == "kitchen" for piece in pieces for word in piece.split())

    def testMaximumMustNotBeBelowMinimum(self):
        with pytest.raises(ValueError, match="maximumCharacters"):
            SentenceStreamer(minimumCharacters=50, maximumCharacters=10)

    def testNegativeMinimumIsRejected(self):
        with pytest.raises(ValueError, match="minimumCharacters"):
            SentenceStreamer(minimumCharacters=-1)


class TestAbbreviations:
    """A full stop after 'e.g.' does not end a sentence."""

    def testAbbreviationDoesNotSplit(self):
        pieces = splitIntoSentences(
            "You could ask about the weather, e.g. whether it will rain later today."
        )

        assert len(pieces) == 1

    def testTitleDoesNotSplit(self):
        pieces = splitIntoSentences("I spoke to Dr. Smith about the appointment today.")

        assert len(pieces) == 1

    def testRealSentenceEndStillSplits(self):
        pieces = splitIntoSentences(
            "I turned off the light. The kitchen is dark now, as requested."
        )

        assert len(pieces) == 2


class TestStreamFidelity:
    """Nothing may be lost or duplicated between input and output."""

    def testAllContentSurvivesFragmentation(self):
        text = (
            "I have turned the kitchen light off. The room should be dark now. "
            "Would you like the lamp on instead?"
        )

        joined = " ".join(feedAll(SentenceStreamer(), text, fragmentSize=1))

        assert joined.split() == text.split()

    def testFragmentSizeDoesNotChangeTheResult(self):
        text = "The light is off. The heating is on. Anything else you need?"

        oneByOne = feedAll(SentenceStreamer(), text, fragmentSize=1)
        inChunks = feedAll(SentenceStreamer(), text, fragmentSize=17)

        assert oneByOne == inChunks

    def testFlushEmptiesTheBuffer(self):
        streamer = SentenceStreamer()
        streamer.feed("partial text")

        streamer.flush()

        assert streamer.pending == ""
        assert streamer.flush() is None

    def testEmptyInputProducesNothing(self):
        streamer = SentenceStreamer()

        assert streamer.feed("") == []
        assert streamer.flush() is None

    def testWhitespaceOnlyProducesNothing(self):
        streamer = SentenceStreamer()
        streamer.feed("   \n  ")

        assert streamer.flush() is None


class TestFirstPieceLatency:
    """The point of streaming: speak before generation finishes."""

    def testFirstPieceArrivesBeforeTheRestOfTheReply(self):
        streamer = SentenceStreamer()
        text = "I have turned the kitchen light off. "

        released = streamer.feed(text)

        assert released == ["I have turned the kitchen light off."]
        assert streamer.pending == ""
