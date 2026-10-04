# Toddler mode

A simpler assistant for a small child. Short replies in words a three-year-old
knows, and a bank of questions the assistant already knows the answers to.

```bash
python -m app --mode converse --toddler
python -m app --mode wake --toddler        # hands-free, no keyboard at all
```

Or, from any mode already running, say:

> **"start toddler mode"** … **"end toddler mode"**

so a parent can hand the assistant over and take it back without touching a
keyboard. `"toddler mode on"`, `"toddler mode off"` and `"normal mode"` work
too.

## How a turn is decided

Two things happen in every reply, and keeping them apart is what makes the mode
work.

**The questions are deterministic.** What a sheep says, how old someone is and
whether they brushed their teeth are matched against a list of expected answers
in `config/toddler.json`. A child who says "baa" hears "that's correct" in
milliseconds, with no model involved. A small local model is both slower and
less consistent at deciding whether "baa" means "baa" than a list of strings
is.

**The conversation between the questions is the model's.** It is given the
child's name and age, what has already been asked, and whether a question is
about to be appended, along with a hard instruction to reply in one short
sentence. If no model is configured the mode still works — replies become short
fixed phrases and the questions carry the conversation.

Answers are matched inside whole sentences ("the sheep says baa" counts) on
word boundaries ("barn" does not match "bar"), and spelled numbers are treated
as digits, because Whisper writes "three" as often as "3".

## Not bombarding the child

Nobody enjoys being interrogated, and this is the part that took the most
adjusting against a real model.

- A question from the bank is offered when the child says something with no
  content of its own — "hello", "yeah" — or after `askEveryTurns` ordinary
  turns have passed. Two, by default.
- The model may ask a question of its own, but never two turns running.
- Only ever one question per reply. This is enforced in code, not asked for in
  the prompt: qwen2.5:3b ignored the instruction reliably enough to produce
  *"Hello there! What's your favorite color? What sound does a cow make?"*, and
  once asked the same question the bank was about to append, so the child heard
  it twice. Any question the model asks is stripped when one is being appended.

## When the child says nothing

A toddler asked a question often says nothing at all — they are thinking, or
distracted, or have wandered off. Silence is not an answer, but it is not a
reason to stop either.

After `promptAfterSeconds` (7 by default) of hearing nothing, the assistant
speaks on its own:

```text
  assistant: What sound does a sheep make?
  (7 seconds)
  assistant: Shall I tell you? A sheep makes a baa sound. Can you say baa?
  (7 seconds)
  assistant: Here's another one. What sound does a cow make?
  (7 seconds)
  assistant: I know this one! A cow makes a moo sound. Can you say moo?
  (7 seconds)
  assistant: (stays quiet)
```

The answer is revealed first, because the child was asked something and hearing
"a sheep says baa" is the point of the exercise — better than repeating the
question at someone who did not answer it. Then a different question, in case
the first was simply not interesting.

After `giveUpAfterPrompts` (3) consecutive prompts into silence, it stops.
An assistant still asking questions of an empty room is worse company than one
that has stopped. **Anything the child says starts it over**, so it comes back
to life the moment they do.

The reveal uses the first `incorrect` phrase, which is why those should state
the answer plainly. Later phrasings may open with "not quite", which would be
unfair to someone who said nothing at all — so a bare statement goes first:

```json
"incorrect": [
    "A sheep makes a baa sound. Can you say baa?",
    "Nearly! A sheep says baa."
]
```

This needs hands-free listening (`--mode wake`). There is no silence to measure
in `--mode converse`, where someone has to press Enter to speak. Set
`promptAfterSeconds` to `0` to turn it off.

## The questions file

```json
{
    "childName": "Sam",
    "childAge": 3,
    "askEveryTurns": 2,
    "questions": [
        {
            "id": "sheep-sound",
            "ask": "What sound does a sheep make?",
            "answers": ["baa", "bar", "ba", "maa"],
            "correct": ["That's correct! A sheep says baa."],
            "incorrect": ["A sheep makes a baa sound. Can you say baa?"],
            "topic": "animals",
            "difficulty": 1
        }
    ]
}
```

`correct` and `incorrect` may each list several phrasings, which are rotated
so the same answer does not always get the same reply.

The bank ships with about 110 questions across thirteen topics: animals,
colours, shapes, counting, body, food, vehicles, nature, everyday, opposites,
riddles, reasoning and about-you.

### Difficulty, and what it does

`difficulty` runs from 1 to 5. It is not a fixed sequence — **each topic keeps
its own level, and it moves with the child.**

Two right answers in a row within a topic offers harder questions there; two
wrong offers easier ones. Two rather than one, because a three-year-old
guesses, and a single lucky "moo" is not mastery. A right answer breaks a run
of wrong ones, so an unlucky patch does not drag the level down.

Answering the animal questions correctly walks up the topic on its own:

```text
[animals level 1] 1  What sound does a cow make?
[animals level 1] 1  What sound does a sheep make?
[animals level 2] 2  What sound does a lion make?
[animals level 2] 2  What animal says meow?
[animals level 3] 3  What does a horse eat?
[animals level 3] 3  Where does a fish live?
[animals level 4] 4  Can a penguin fly?
[animals level 5] 5  Which is bigger, a mouse or an elephant?
```

Meanwhile a child who is struggling with counting keeps being asked how many
noses they have. Topics are tracked independently, so being good at animals
says nothing about being good at numbers.

Selection is, in order: questions not yet asked; difficulty at or below the
child's level for that topic; the hardest of those, which is what makes the
level mean anything; a different topic from the last question, for variety;
then the file's own order, so a parent can still sequence them. Nothing harder
than the current level is offered unless nothing else is left.

`:toddler` shows where the child has got to in each topic. The levels are
session state — they start again each time toddler mode is entered, and are
not written to the file.

### Placeholders

`{name}` and `{age}` are filled in from what the assistant knows, and may
appear in any field:

```json
{
    "id": "how-old",
    "ask": "How old is {name}?",
    "answers": ["{age}"],
    "correct": ["That's right, {name} is {age}!"],
    "incorrect": ["{name} is actually {age}."]
}
```

A question whose placeholders cannot all be filled is simply not asked. That is
what makes personal questions wait until someone has said who is talking, and
it applies to the whole question, not just the prompt — there is no use asking
"how old are you?" if the reply would have to say "{name} is {age}".

### The child's name

With `childName` set to `null`, entering toddler mode asks:

> Hello! It's lovely to talk to you. What's your name?

"Sam", "I'm Sam" and "my name is Sam" all work, and the name holds for
that session only. A reply that plainly is not a name ("I don't know") is not
taken as one, and the assistant carries on rather than asking twice — the
personal questions simply stay held back.

Setting `childName` in the file makes it permanent, and the mode greets by name
instead of asking.

### Open questions

Give an empty `answers` list for a question where anything counts. Those are
never marked wrong — "can you name something red?" answered with "fire engine"
is correct, and no list could say so:

```json
{
    "id": "name-something-red",
    "ask": "Can you name something that is red?",
    "answers": [],
    "correct": ["Good thinking! Lots of things are red."],
    "incorrect": ["Good thinking! Lots of things are red."]
}
```

`incorrect` is never reached, but keeping it identical means the answer still
reads properly if someone later adds an `answers` list. The shipped bank uses
these for counting aloud, physical prompts ("can you touch your ears?") and the
kinder reasoning questions, where a toddler's answer is a good one whatever
they say.

## Editing from the ordinary prompt

Run without `--toddler` and use `:toddler`. Every command that changes anything
writes the file straight back.

```text
:toddler                        show the mode and the questions
:toddler on | off               enter or leave toddler mode
:toddler name Sam             who the assistant is talking to
:toddler age 3                  how old they are
:toddler every 4                ordinary turns between questions
:toddler add <question> | <answers> | <correct> | <incorrect> | <topic> | <1-5>
:toddler remove <id or question>
```

Answers are separated by commas and everything after them is optional:

```text
:toddler add What sound does a pig make? | oink, oik | Yes, a pig says oink!
:toddler add What is a baby cow called? | calf | Yes! | A calf. | animals | 4
```

Give a `topic` and `difficulty` to slot a question into the adaptive
progression; without them it becomes a `custom` question at difficulty 1, which
means it is offered early and to everyone. Leave the answers empty for a
question where anything counts.

Adding a question that already exists replaces it, matched on the wording as
well as the id, so the same question cannot end up in the file twice under two
names and be asked twice.

## Configuration

```json
"toddler": {
    "enabled": false,
    "questionsFile": "config/toddler.json",
    "allowModePhrases": true,
    "askForName": true,
    "promptAfterSeconds": 7.0,
    "giveUpAfterPrompts": 3
}
```

`enabled` starts in toddler mode, which is what `--toddler` sets.
`allowModePhrases` controls whether the spoken phrases switch modes; with both
off, nothing about toddler mode is loaded at all.

A questions file that will not parse disables the mode and logs why. It does
not stop the assistant starting: the parent needs telling, not a machine that
will not boot.

## What to expect from a small model

The conversational half is only as good as the model behind it. On
qwen2.5:3b it is warm and roughly age-appropriate, but it slips into American
spelling ("favorite"), and it does not follow instructions reliably — which is
why every rule that matters to a child is enforced in code rather than asked
for in the prompt.

The system prompt forbids anything frightening, violent, sad or adult, and
replies are cut to two sentences. That is a constraint on a model, not a
guarantee about one. This is a mode to leave a child with while you are in the
room, not instead of being in it.

## What it does not do

- One child at a time. The session state — the name, what has been asked, what
  was right — belongs to the assistant, not to a session, because there is one
  speaker in one room. Two children talking to it over the API share a state.
- The name given by voice is not written to the file. `:toddler name` is.
- Editing is from the terminal prompt only. `--mode wake` has no prompt, so
  a wake-word setup needs the questions prepared in advance, or the file edited
  by hand.
