# Recording a voice

XTTS clones from a short recording. What you record shapes the assistant more
than most people expect: it copies **how you speak**, not only how you sound.
Read the script in a bored newsreader voice and you get a bored newsreader.

## The short version

```text
voices/michael/one.wav
voices/michael/two.wav
voices/michael/three.wav
```

Then point the assistant at it:

```json
{ "textToSpeech": { "voice": "michael" } }
```

Or `:voice michael` at the prompt, or `--voice michael` on the command line.
A single `voices/michael.wav` works too; separate files are only easier to
re-record one at a time.

## How much of it is actually used

Worth knowing, because it is not obvious and it changes what to record.

| | Uses |
|---|---|
| Speaker embedding, which carries **timbre** | up to 30 seconds |
| GPT conditioning, which carries **prosody and style** | up to 30 seconds |

Multiple files are concatenated before either is computed, so three ten-second
takes are treated as one thirty-second recording. Beyond thirty seconds nothing
further is used --- more audio is not better, **cleaner** audio is better.

## What to record

Aim for **30 to 45 seconds** of speech, which leaves room to trim. Three takes
of about fifteen seconds each is comfortable; a fluffed take costs one retake
rather than the lot.

- **Record in the voice you want the assistant to have.** Calm, clear, a little
  slower than conversation, with the warmth you would use telling someone the
  kettle has boiled. This is the single thing that matters most.
- **A quiet room.** No fan, no traffic, no music. XTTS clones room tone as
  faithfully as it clones a voice, and a hum in the reference becomes a hum
  under every reply.
- **A consistent distance** from the microphone, about a hand-span away, off to
  one side so plosives do not thump.
- **Leave the level alone** between takes. The model is not asked to normalise
  loudness, so takes recorded at different levels blend badly.
- **Do not remove the breaths.** They are part of how you speak and their
  absence is audible.

Any format the library reads is fine --- `.wav`, `.flac`, `.mp3`, `.ogg`,
`.m4a`. Audio is resampled to 22 050 Hz internally, so recording at 44 100 or
48 000 is plenty and anything higher is wasted. Mono is preferred; stereo is
downmixed.

## A script to read

**[`script.md`](../script.md)**, in the project root, is the thing to have open
while recording. It is written for the purpose rather than borrowed: broad
phonetic coverage, a mix of statement, question and list, numbers and letters
spoken aloud, and phrasing in the register the assistant will actually use.

It lives on its own rather than in this document so there is one copy to read
from and one copy to maintain.

## Checking what you recorded

```powershell
python scripts\checkVoice.py michael
```

It reports duration, level, clipping, silence and background noise for each
file, and says whether it expects the sample to clone well. Then hear it:

```powershell
python -m app.main --voice michael --say "The kitchen light is now off."
```

Compare against the built-in speaker to judge it fairly:

```powershell
python -m app.main --voice "Claribel Dervla" --say "The kitchen light is now off."
```

## If it does not sound like you

- **Thin or robotic** --- usually too little audio, or all of it monotone. Add
  a take with more range in it.
- **Muffled** --- too far from the microphone, or a lossy recording. Re-record
  closer, uncompressed.
- **A hiss or hum under every reply** --- it is in the reference. Re-record
  somewhere quieter; noise reduction afterwards tends to make cloning worse,
  not better, because it leaves artefacts the model copies.
- **Right timbre, wrong manner** --- you read it rather than said it. This is
  the common one. Record again talking to somebody.

Nothing in `voices/` is committed, and neither are the embeddings cached beside
it in `voices/.cache/`. Edit or replace a sample and the cache invalidates on
its own.
