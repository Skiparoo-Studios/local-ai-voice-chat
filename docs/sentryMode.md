# Standing down

"Goodbye" stops the assistant answering anything at all. "Hello" brings it
back.

```text
you: goodbye
  -> Goodbye. Say hello when you want me.
you: what is the time
  -> (nothing)
you: read me a story
  -> (nothing)
you: hello
  -> Hello. I'm listening.
you: what is the time
  -> It's 11:10 PM.
```

Worth having because with `always-awake` and no wake word, every conversation
in the room reaches the recogniser and some of it gets answered. This is the
off switch that does not involve closing anything.

## What counts

**Standing down:** goodbye, bye, good night, see you later, that's all, stand
down, go to sleep, leave me alone --- optionally prefixed with "okay" or
followed by "then".

**Waking:** hello, hi, hey, good morning, wake up, are you there, come back.

Both are matched against the **whole utterance**, so a book read aloud saying
"he said goodbye at the station" does not dismiss it, and "she said hello to
the man in the hat" does not wake it.

`quit` and `exit` are deliberately not farewells --- those close the program,
which is a different thing, and the rules handler already answers them.

## While dormant

It is **silent**, not refusing. An assistant that says "I am asleep" every time
somebody speaks in the room has not been dismissed at all.

- Nothing is answered, whatever it is.
- Toddler mode does not fill silences.
- A book playing is **stopped**, since carrying on happily after goodbye is not
  what anyone means by it. The position is bookmarked, so "hello" then "carry
  on" picks it back up.

## Configuration

```json
{
    "sentry": {
        "enabled": true,
        "dormant": false,
        "greetOnWaking": true
    }
}
```

`dormant: true` starts it dismissed, so nothing is answered until somebody says
hello --- reasonable for a device in a shared room.
`greetOnWaking: false` wakes it silently.

It is the outermost layer, in front of toddler mode and books both, because
standing down has to mean everything stops answering rather than everything
except whichever mode happens to be active.

## It is not a security control

Anyone who can be heard can say hello. It stops the assistant chattering, not
anyone from using it.
