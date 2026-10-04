# Choosing a speech synthesiser

Two real providers, and the choice between them is not close on any single
axis — it depends entirely on what you want.

| | XTTS-v2 (`xtts`) | Piper (`piper`) |
|---|---|---|
| Clones your voice | **yes**, from ~30 s of audio | no, trained voices only |
| Licence on the model | **CPML — non-commercial, output included** | per-voice, mostly MIT / CC |
| Licence on the engine | MPL-2.0 (coqui-tts) | GPL-3.0-or-later (bundles espeak-ng) |
| Restricts what you may do with the audio | **yes** | no |
| Load time | 26.6 s | **2.0 s** |
| Speed | 1.32 s for a sentence, on a GPU | **0.20 s for a sentence, on a CPU** |
| Real-time factor | needs a GPU to be usable | **0.054 on CPU** |
| Download size | ~1.8 GB | ~60 MB per voice |
| Runs on a Raspberry Pi | no | **yes** |

```json
{ "textToSpeech": { "provider": "piper", "voice": "en_GB-cori-medium" } }
```

### `model` and `voice` mean different things per provider

| | `model` | `voice` |
|---|---|---|
| `xtts` | the coqui model id | a folder under `paths.voices` with your recordings |
| `piper` | **unused** | the published voice name, e.g. `en_GB-cori-medium` |

A Piper voice *is* its model, so `textToSpeech.model` does nothing when
`provider` is `piper`. Leaving an XTTS model name there while running Piper is
harmless — it is never read — but it reads like a mistake, so set it to `""`.
XTTS falls back to its own default when the field is empty, so clearing it does
not break switching back.

**Pick XTTS if you want your own voice.** That is the thing Piper cannot do at
all, and it is the reason XTTS remains the default.

**Pick Piper for anything else** — a Pi, a machine whose GPU is busy with the
language model, or a use the CPML forbids.

## Licences, briefly

The two are restricted in completely different ways, and conflating them is
easy.

- **XTTS-v2 is CPML**: non-commercial use only, and the restriction follows the
  audio. You may not sell a recording made with it.
- **Piper is GPL-3.0-or-later**: no restriction on use, and **no restriction on
  the output**. Audio you generate is yours. The obligation is copyleft on
  anyone *distributing software that incorporates it*.

So for commercial use Piper works and XTTS does not, but if you ship a product
built on this assistant plus Piper, the GPL applies to that combined work.
Neither is distributed here; both install on request. See `NOTICE.md`.

Piper's **voices** are licensed individually — mostly MIT or Creative Commons.
Check the one you choose at
<https://huggingface.co/rhasspy/piper-voices>.

## Piper voices

Voices download on first use into `models/piper/`, about 60 MB each:

```bash
python -m piper.download_voices --list        # everything available
python -m piper.download_voices en_GB-cori-medium
```

Or just name one in settings and let the provider fetch it. Set
`textToSpeech.allowVoiceDownload` to `false` on a machine that should never
reach the network.

Multi-speaker voices select a speaker after a hash:

```json
{ "textToSpeech": { "voice": "en_US-libritts-high#42" } }
```

`textToSpeech.speechRate` slows it down — above 1 is slower, which is worth
having for a small child:

```json
{ "textToSpeech": { "provider": "piper", "speechRate": 1.3 } }
```

### Choosing between voices

Both British medium voices were tested by synthesising a phrase and
transcribing it back with Whisper. They fail in different places:

| | `en_GB-cori-medium` | `en_GB-alba-medium` |
|---|---|---|
| "What sound does a **sheep** make?" | correct, twice from two | heard as "**ship**" once in two |
| "**Yes.**" | sometimes 3.5 s of babble | correct, 0.67 s |

`cori` is the default because toddler mode asks about sheep constantly, and
`alba` is a Scottish voice whose vowel makes "sheep" genuinely ambiguous.

### The "Yes." quirk

`en_GB-cori-medium` is unreliable on the exact string `"Yes."`. Measured over
several runs it produced 0.7 s, 1.5 s, 2.3 s, 3.4 s and 4.4 s for one word, and
one run transcribed as `"Yes! vvvvvvvvvvvv"`.

This is the model, not this code — raw `piper` produces the same bytes. It is a
VITS duration predictor given almost no phonemes to work from. Any of these
avoids it:

```text
"Yes."         unreliable
"Yes"          fine, 0.7 s
"Yes?"         fine, 0.66 s
"Yes please."  fine, 1.0 s
```

`No.`, `Done.`, `Okay!`, `Hello.`, `Stopped.` and `That's right!` were all fine
across three runs each, so this is narrow rather than a general problem with
short replies. The assistant's own acknowledgement is `"Yes?"`, which is
unaffected.

## Adding another

The registry takes `"module:class"` and imports lazily, so an unselected
provider costs nothing:

```python
from app.speech.textToSpeech import registerProvider
registerProvider("kokoro", "mypackage.kokoroProvider:KokoroProvider")
```

Implement `TextToSpeechProvider`: `fromSettings`, `load`, `synthesise` and
`sampleRate`. Override `synthesiseStream` and `supportsStreaming` if the model
can emit audio before it has finished, which is what lets the runtime start
speaking early. `app/speech/models/piperProvider.py` is about 300 lines and is
the shorter of the two to copy.

Kokoro is the obvious next candidate: Apache-2.0 throughout, engine and voices,
which neither of the current two manages.
