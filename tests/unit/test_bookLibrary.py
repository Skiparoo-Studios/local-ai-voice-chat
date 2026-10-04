"""Finding books and their chapters.

Built on fixtures rather than the real library, so these pass on a machine
with no books at all. The shapes are taken from a real audiobook library,
including one file that is malformed in a way that mattered.
"""

from __future__ import annotations

import pytest

from app.books.library import (
    BookLibrary,
    cleanTitle,
    loadTextBook,
    separateWeldedHeadings,
    spokenNumber,
)

HARRY_POTTER = "Harry Potter and the Chamber of Secrets, Book 2 [B0EXAMPLE2]"


@pytest.fixture
def library(tmp_path):
    """Two shelves, laid out the way an audiobook library usually is."""
    audio = tmp_path / "Books"
    text = tmp_path / "TextBooks"
    audio.mkdir()
    text.mkdir()

    def book(folder: str, parts: list[str]) -> None:
        directory = audio / folder
        directory.mkdir()
        stem = folder.rsplit(" [", 1)[0]
        for index, title in enumerate(parts, start=1):
            (directory / f"{stem} [X] - {index:02d} - {title}.mp3").write_bytes(b"")

    book(HARRY_POTTER, ["Opening Credits", "Chapter 1_ The Worst Birthday",
                        "Chapter 2_ Dobby's Warning", "End Credits"])
    book("Harry Potter and the Goblet of Fire, Book 4 [B0EXAMPLE3]",
         ["Opening Credits", "Chapter 1_ The Riddle House"])
    book("Digital Fortress [B0EXAMPLE1]", ["Chapter 1", "Chapter 2"])

    (text / "hobbit.txt").write_text(
        "The Hobbit\n\nCHAPTER ONE\nAN UNEXPECTED PARTY\n"
        "In a hole in the ground there lived a hobbit.\n\n"
        "CHAPTER TWO\nROAST MUTTON\nUp jumped Bilbo.\n",
        encoding="utf-8",
    )
    # Only .txt is read for now; an epub must not be offered as readable.
    (text / "Something Else.epub").write_bytes(b"not really an epub")

    return BookLibrary(audio, text)


class TestDiscovery:
    def testAudiobooksAreFoundByFolder(self, library):
        titles = [book.title for book in library.audioBooks()]

        assert "Digital Fortress" in titles
        assert len(titles) == 3

    def testTheAsinIsStrippedFromTheTitle(self, library):
        assert all("[" not in book.title for book in library.audioBooks())

    def testChaptersAreOrderedByTheirNumber(self, library):
        book = library.find("chamber of secrets").book

        assert [chapter.number for chapter in book.chapters] == [1, 2, 3, 4]

    def testAnUnderscoreBecomesAColon(self, library):
        """A colon arrives as an underscore, Windows forbidding one."""
        book = library.find("chamber of secrets").book

        assert book.chapterNumbered(2).title == "Chapter 1: The Worst Birthday"

    def testCreditsAreNotStoryChapters(self, library):
        book = library.find("chamber of secrets").book

        assert len(book.chapters) == 4
        assert len(book.storyChapters) == 2
        assert book.storyChapters[0].title.startswith("Chapter 1")

    def testTextBooksAreFound(self, library):
        assert [book.title for book in library.textBooks()] == ["hobbit"]

    def testOnlyPlainTextIsOffered(self, library):
        """epub and pdf come later; until then they must not be listed."""
        assert all(book.path.suffix == ".txt" for book in library.textBooks())

    def testAudiobooksComeFirst(self, library):
        assert library.all()[0].isAudio

    def testAMissingDirectoryIsNotAnError(self, tmp_path):
        empty = BookLibrary(tmp_path / "nope", tmp_path / "also nope")

        assert empty.audioBooks() == []
        assert empty.textBooks() == []


class TestFindingABook:
    def testATitleIsMatchedLoosely(self, library):
        assert library.find("digital fortress").book.title == "Digital Fortress"

    def testASubtitleIsEnoughToIdentifyOne(self, library):
        match = library.find("chamber of secrets")

        assert match.found
        assert "Chamber of Secrets" in match.book.title

    def testAnAmbiguousTitleReturnsEveryCandidate(self, library):
        """Seven books are called Harry Potter something, so the caller has to
        ask rather than pick one."""
        match = library.find("harry potter")

        assert match.ambiguous
        assert not match.found
        assert len(match.alternatives) == 2

    def testABookThatIsNotThereIsNotInvented(self, library):
        assert not library.find("moby dick").found

    def testAnEmptyRequestMatchesNothing(self, library):
        assert not library.find("   ").found

    def testAudiobooksWinOverTextBooks(self, tmp_path):
        """A recording of a narrator beats the assistant reading aloud."""
        audio, text = tmp_path / "a", tmp_path / "t"
        (audio / "The Hobbit [B01]").mkdir(parents=True)
        (audio / "The Hobbit [B01]" / "The Hobbit [B01] - 01 - Chapter 1.mp3").write_bytes(b"")
        text.mkdir()
        (text / "The Hobbit.txt").write_text("CHAPTER ONE\nwords\n", encoding="utf-8")

        assert BookLibrary(audio, text).find("the hobbit").book.isAudio


class TestSuggestions:
    def testAFewTitlesAreOffered(self, library):
        assert len(library.suggestions(2)) == 2

    def testTheyAreSpreadAcrossTheShelf(self, library):
        """Taking the first three alphabetically would offer three Harry
        Potters to somebody who owns seven of them."""
        offered = [book.title for book in library.suggestions(3)]

        assert len(set(offered)) == 3

    def testASmallLibraryOffersEverythingItHas(self, tmp_path):
        empty = BookLibrary(tmp_path / "none", tmp_path / "none")

        assert empty.suggestions(3) == []


class TestFindingAChapter:
    def testByNumber(self, library):
        book = library.find("chamber of secrets").book

        assert book.findChapter("chapter 2").title == "Chapter 2: Dobby's Warning"

    def testBySpelledNumber(self, library):
        book = library.find("chamber of secrets").book

        assert book.findChapter("chapter two").title == "Chapter 2: Dobby's Warning"

    def testByOrdinal(self, library):
        book = library.find("chamber of secrets").book

        assert book.findChapter("the second one").title == "Chapter 2: Dobby's Warning"

    def testNumbersCountStoryChaptersNotFiles(self, library):
        """Opening credits are file 1, so "chapter one" must not mean them."""
        book = library.find("chamber of secrets").book

        assert book.findChapter("chapter one").title == "Chapter 1: The Worst Birthday"

    def testByName(self, library):
        book = library.find("chamber of secrets").book

        assert book.findChapter("dobby").number == 3

    def testFromTheBeginningIsNotAChapter(self, library):
        """The caller handles it; it must not resolve to a numbered chapter."""
        book = library.find("chamber of secrets").book

        assert book.findChapter("from the beginning") is None

    @pytest.mark.parametrize(
        "spoken,expected",
        [("chapter 3", 3), ("chapter three", 3), ("the third", 3), ("nothing", None)],
    )
    def testSpokenNumbers(self, spoken, expected):
        assert spokenNumber(spoken) == expected


class TestTextChapters:
    def testHeadingsBecomeChapters(self, library):
        book = library.textBooks()[0]

        assert len(book.chapters) == 2

    def testTheFollowingLineJoinsTheTitle(self, library):
        """Files put "CHAPTER ONE" and "AN UNEXPECTED PARTY" on separate
        lines, and only the pair is worth announcing."""
        book = library.textBooks()[0]

        assert book.chapters[0].title == "Chapter One: An Unexpected Party"

    def testTheProseIsRecoverable(self, library):
        book = library.textBooks()[0]

        assert "hole in the ground" in book.textFor(book.chapters[0])

    def testChaptersDoNotBleedIntoEachOther(self, library):
        book = library.textBooks()[0]

        assert "Roast Mutton" not in book.textFor(book.chapters[0])

    def testAFileWithNoHeadingsIsOneChapter(self, tmp_path):
        path = tmp_path / "plain.txt"
        path.write_text("Just some prose, with no headings at all.\n", encoding="utf-8")

        book = loadTextBook(path)

        assert len(book.chapters) == 1
        assert "prose" in book.textFor(book.chapters[0])

    def testProseIsNotMistakenForAHeading(self, tmp_path):
        path = tmp_path / "plain.txt"
        path.write_text(
            "CHAPTER ONE\nTHE START\n"
            "The chapter that follows is long and ends in a full stop.\n",
            encoding="utf-8",
        )

        assert len(loadTextBook(path).chapters) == 1


class TestWeldedHeadings:
    """One heading in seventeen, in a real file, had been welded onto the end
    of the preceding paragraph. Left alone that chapter cannot be asked for and
    the one before it is twice its proper length."""

    def testAHeadingStuckToAParagraphIsSeparated(self):
        lines = ["He grunted and offered him another rock cake. CHAPTER NINE "]

        assert separateWeldedHeadings(lines) == [
            "He grunted and offered him another rock cake.",
            "CHAPTER NINE",
        ] or separateWeldedHeadings(lines) == (
            "He grunted and offered him another rock cake.",
            "CHAPTER NINE",
        )

    def testItProducesAnExtraChapter(self, tmp_path):
        path = tmp_path / "welded.txt"
        path.write_text(
            "CHAPTER EIGHT\nTHE POTIONS MASTER\nHe read the story again. CHAPTER NINE \n"
            "THE MIDNIGHT DUEL\nHarry had never believed it.\n",
            encoding="utf-8",
        )

        book = loadTextBook(path)

        assert [chapter.number for chapter in book.chapters] == [1, 2]
        assert book.findChapter("chapter nine").title == "Chapter Nine: The Midnight Duel"

    def testProseMentioningAChapterIsLeftAlone(self):
        """Lower case, and mid sentence, so it is speech and not a heading."""
        lines = ["She said, read me chapter nine before bed."]

        assert tuple(separateWeldedHeadings(lines)) == tuple(lines)

    def testAHeadingAlreadyOnItsOwnLineIsUntouched(self):
        lines = ["CHAPTER NINE", "THE MIDNIGHT DUEL"]

        assert tuple(separateWeldedHeadings(lines)) == tuple(lines)


class TestTitleCleaning:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Digital Fortress [B0EXAMPLE1]", "Digital Fortress"),
            ("The Tales of Beedle the Bard [9990000001]", "The Tales of Beedle the Bard"),
            ("No identifier here", "No identifier here"),
        ],
    )
    def testIdentifiersAreStripped(self, raw, expected):
        assert cleanTitle(raw) == expected
