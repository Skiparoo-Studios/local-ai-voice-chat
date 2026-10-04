# Reading books aloud

Two libraries, searched in order. An audiobook is a recording of a
professional narrator; a text book is the assistant reading to you. Given
both, the recording wins.

```json
{
    "books": {
        "enabled": true,
        "audioDirectory": "books/audio",
        "textDirectory": "books/text",
        "blockSeconds": 5.0
    }
}
```

```powershell
pip install -e ".[books]"
```

The layout it expects:

```text
Books/Digital Fortress [B0EXAMPLE1]/Digital Fortress [B0EXAMPLE1] - 01 - Chapter 1.mp3
TextBooks/harrypotter.txt
```

## Asking

```text
you: read me a story
  -> Sure, what book would you like me to read? I have Angels & Demons,
     Harry Potter and the Half-Blood Prince, Book 6 or The Da Vinci Code,
     and others.
you: harry potter
  -> I have 7 of those. Did you mean Chamber of Secrets, Book 2, Deathly
     Hallows, Book 7, Goblet of Fire, Book 4 or Half-Blood Prince, Book 6?
you: the chamber of secrets
  -> I have Harry Potter and the Chamber of Secrets, Book 2. Which chapter,
     or shall I start from the beginning?
you: chapter two
  -> Chapter 2: Dobby's Warning, from Harry Potter and the Chamber of
     Secrets, Book 2. Say stop when you've had enough.
you: stop
  -> Stopped. You were on Chapter 2: Dobby's Warning.
```

**"read me a story"**, "read to me", "tell me a story", "listen to an audio
book", "put on an audiobook" and "play me an audio book" all open the
conversation. **audiobook** and **audio book** are both understood, which
matters: without the two-word spelling "play me an audio book" went past this
handler to the device router, which then asked which light to turn on. Naming the book in the same
breath — "read me digital fortress" — skips straight to the chapter question,
and a book with only one chapter just starts.

A title shared by several books lists what tells them apart, rather than
reading "Harry Potter and the" back at you four times.

**"stop"** ends it, and says where you had got to. It is only claimed while a
book is playing or a question is pending, so the router can still act on
"stop" itself the rest of the time.

## Carrying on

Stopping records where you were, in `outputs/bookmarks.json`, so resume
survives the assistant being closed:

```text
you: carry on                (also: resume, continue, keep going)
  -> Carrying on with Digital Fortress, Chapter 1, 20 minutes 45 seconds in.
```

A bare "carry on" means the last book listened to, which is nearly always what
is meant. Naming one resumes that instead, and asking for a book you are part
way through offers the choice:

```text
you: read me digital fortress
  -> I have Digital Fortress. You were on Chapter 2, 10 minutes 0 seconds in.
     Carry on, a chapter, or start from the beginning?
```

It rewinds three seconds so the join sounds deliberate rather than clipped,
and covers the block already inside the device when stop arrived. A book barely
started goes back to the beginning instead, since resuming eight seconds in is
more confusing than useful.

One position per book, not per chapter. The position is what has been written
to the speaker, which on a real device is what was heard.

A text book bookmarks a **sentence** rather than a time, because that is what
it is read in and seconds would be a guess.

## Chapters

Any of these work:

```text
from the beginning        the first chapter that is not credits
chapter three             counted among the story chapters
chapter 3
the third one
dobby                     matched against the chapter's name
```

Numbers count **story** chapters, not files. An audiobook carries opening
credits as file 1, so "chapter one" means the first chapter rather than the
copyright notice.

## While a book is playing

**Only commands are acted on. Everything else heard is ignored without a
word.** This is not a shortcut, it is the fix for the worst bug this feature
had.

Continuous listening does not stop when a book starts, so the microphone hears
the narrator. Every sentence arrived looking like something a person had said,
the assistant answered it, and the reply opened the speaker at its own sample
rate --- which closed the stream the book was already writing to. The audio
broke up after a few seconds and afterwards nothing played at all. Two writers,
one device.

So while reading, this is the whole vocabulary:

| said | effect |
|---|---|
| **stop audiobook**, stop the book, stop reading, pause the book | stops, and says where you had got to |
| next, next chapter, skip | the next chapter |
| back, previous chapter | the one before |
| what is this, which book | says what is playing |

**A bare "stop" does not stop a book, deliberately.** A narrator says it often
enough, and one that ends the chapter is far worse than having to say two
words. Say **"stop audiobook"**. Bare "stop" still works everywhere else, and
still cancels a pending question, because nothing is playing to talk over it.

Anything else produces an empty reply, which the runtime never speaks, so
there is only ever one writer to the speaker. The patterns are anchored to the
whole utterance, so a narrator saying "she told him to stop the car" does not
trigger anything either.

### Being heard over the book

One speaker and one microphone in a room is genuinely hard: while a chapter
plays, the microphone is full of narration and a spoken command competes with
it. Three things help, in order of how much:

```json
{ "books": { "volume": 0.6 } }
```

`books.volume` plays the book more quietly so the microphone has a better
chance, without reaching for the system mixer. A distinctive phrase --- "stop
audiobook" rather than "stop" --- survives a poor transcription far better
than a single short word. And a wake word, or headphones, removes the problem
rather than reducing it.

The cost is that you cannot ask an unrelated question mid-chapter --- it is
ignored rather than answered. That is the right trade: being talked over is
worse than having to ask again after saying stop.

Two things worth knowing. The recogniser runs continuously through the whole
chapter, which is a constant load it would not otherwise carry. And a narrator
whose line is transcribed as exactly "stop" will stop the book; it has not
happened in testing, but the mechanism is there.

## Text books

### Asking for the text rather than a recording

Audiobooks answer first, so a book you own both ways always plays the
recording. Say **text**, **written** or **text file** to reach the other shelf:

```text
read me the text of harry potter        the .txt
read me the harry potter text file      the .txt
read me harry potter                    the recordings, and there are seven
```

Titles are also matched with their spaces removed, because a file on disk is
called `harrypotter.txt` while the person asking says "harry potter". Without
that the text library could not be reached by voice at all. One word is not
enough --- "harry" alone will not match `harrypotter`.

Only `.txt` for now. epub and pdf are recognised well enough to say so rather
than pretending the book is not there:

```text
you: read me the hobbit
  -> I have The Hobbit but only as epub, which I can't read yet. I can do
     plain text and audiobooks.
```

Chapters are found by scanning for headings, and numbered by the order they
appear rather than by parsing "CHAPTER SEVENTEEN" — the order is always right
and the spelling is not always parseable. Asking for "chapter nine" tries the
count first and the heading's own name second, which matters more than it
sounds: see below.

A chapter is synthesised a sentence at a time, so stopping is prompt and no
more is generated than gets heard. Generating a five-thousand-word chapter
before saying any of it would take minutes and waste nearly all of it.

### Headings welded to paragraphs

One real file had a chapter heading stuck to the end of the preceding
paragraph:

```text
...Snape that he didn't want to tell Harry? CHAPTER NINE
```

Sixteen of seventeen chapters parsed; the ninth could not be asked for and the
eighth was twice its proper length. Headings in upper case at the end of a line
are put back on their own line before parsing, which recovers it. Prose that
merely mentions a chapter — lower case, mid-sentence — is left alone.

## How it behaves while reading

Reading is a background activity, not a turn. The assistant stays responsive
and keeps listening, so "stop" is heard.

- **Audio is decoded in blocks** of `blockSeconds`, never loaded whole. One
  chapter in this library is seventy-two minutes, which is ninety-five
  megabytes as sixteen-bit mono.
- **Playback paces the decoding.** Writing to the speaker blocks until the
  device accepts it, so the decoder runs only as fast as the audio is heard.
  Measured against an unpaced sink it decoded 550 blocks in a second; against
  a paced one, four.
- **Stopping is acted on within about one block**, since audio already handed
  to the device has to finish.
- **The book waits for the announcement to finish.** The reply naming the
  chapter goes through the same speaker, so the reader asks the runtime
  whether it is still talking before starting.

## What is not done yet

- **epub and pdf.** Recognised and reported, not read.
- **No resume.** Stopping forgets the position; asking again starts the
  chapter over. The chapter is remembered only for the sentence that says
  where you were.
- **No pause.** "Stop" ends it.
- **One book at a time**, which is also all one speaker can do.
- **Text chapters are read from the top**, including the heading, which is
  announced as part of the prose.
