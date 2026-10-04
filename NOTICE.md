# Third-party components

The source code in this repository is under the MIT licence (see `LICENSE`).
That licence covers the code and nothing else. The models this software can
download and run are third-party works under their own terms, and **the MIT
licence grants you no rights over any of them**.

**No model weights are distributed in this repository.** Everything below is
fetched at runtime, on first use, by the library that needs it. `models/` and
`voices/` contain only placeholder files, and the `.gitignore` excludes
`*.pth`, `*.onnx`, `*.bin` and `*.safetensors` so that they stay that way.

## XTTS-v2 — non-commercial only

> This project supports XTTS-v2 as a text-to-speech backend. XTTS-v2 is a
> third-party model licensed separately under the **Coqui Public Model Licence
> (CPML)**. The CPML restricts the model **and its outputs** to non-commercial
> uses. The licence of this project does not grant any rights to use XTTS-v2
> commercially. Users are responsible for ensuring their use of third-party
> models complies with the applicable licence.

Two consequences worth stating plainly, because they are easy to miss:

- **The restriction follows the audio, not just the model.** Speech generated
  by XTTS-v2 is an output of the model and carries the same non-commercial
  restriction. A recording made with it is not yours to sell.
- **If you pass the model, a modification of it, or its output to someone
  else, the CPML requires you to give them the licence terms or its URL.**
  Point them at <https://coqui.ai/cpml> or at this file.

The model is not bundled. It downloads on first use, and the library requires
you to accept the licence explicitly before it will do so:

```powershell
$env:COQUI_TOS_AGREED = "1"
```

Refusing that is a supported outcome: the assistant reports the licence and
stops, rather than downloading anything.

## XTTS is one provider, not the assistant

Speech synthesis sits behind `TextToSpeechProvider`, and XTTS is one
implementation registered in it:

```text
TextToSpeechProvider
    ├── XttsProvider    xtts    CPML, non-commercial only
    ├── PiperProvider   piper   GPL-3.0 engine, per-voice model licences
    └── ToneProvider    tone    no model, no third-party terms
```

### Piper, and why GPL is not the same problem

`piper-tts` is **GPL-3.0-or-later** (it was MIT up to 1.2.0; the relicence at
1.3.0 came with bundling espeak-ng, which is itself GPL). That reads alarming
next to an MIT project, so the difference from the CPML is worth stating:

|  | XTTS-v2 (CPML) | Piper (GPL-3.0) |
|---|---|---|
| Restricts commercial **use** | yes | **no** |
| Restricts the **audio generated** | yes | **no** |
| Obligation when **distributing a combined work** | pass on the terms | licence the combined work under GPL-3.0 |

So Piper makes commercial *use* possible where XTTS forbids it, and the audio
is unencumbered. What it asks in return is copyleft on anyone shipping a
product that incorporates it. That is a decision for whoever ships such a
product, not for someone running this at home.

It is in its own `piper` extra so that installing this project does not pull
GPL code unless asked. Piper **voices** are licensed individually, mostly MIT
or Creative Commons — check the one you use at
<https://huggingface.co/rhasspy/piper-voices>.

Switching is configuration, not code:

```json
{ "textToSpeech": { "provider": "tone" } }
```

Adding another is a registry entry, and needs no change to the runtime:

```python
from app.speech.textToSpeech import registerProvider
registerProvider("piper", "mypackage.piperProvider:PiperProvider")
```

So a deployment that never selects `xtts` never downloads it, never runs it,
and is not subject to the CPML at all. A permissively licensed synthesiser —
Piper and Kokoro are the obvious candidates — would make the whole system
commercially usable without touching anything outside one provider file.

## Everything else

Read at runtime from the installed packages. Verify with
`pip show <package>`; versions are what this project was developed against.

### Models downloaded at runtime

| Model | Used by | Licence |
|---|---|---|
| XTTS-v2 | speech synthesis (`xtts`) | **CPML — non-commercial only** |
| Piper voices | speech synthesis (`piper`) | per voice, mostly MIT / CC |
| Whisper (`small` by default) | speech recognition | MIT |
| Silero VAD | voice activity detection | MIT |
| openWakeWord pretrained models | wake word | Apache-2.0 |

Whisper, Silero and openWakeWord are all permissively licensed. **XTTS-v2 is
the only component in this project that restricts what you may do with its
output**, and selecting a different synthesiser avoids it entirely.

### Python packages

| Package | Licence |
|---|---|
| coqui-tts | MPL-2.0 |
| **piper-tts** | **GPL-3.0-or-later** |
| faster-whisper | MIT |
| ctranslate2 | MIT |
| silero-vad | MIT |
| openwakeword | Apache-2.0 |
| onnxruntime | MIT |
| torch | Apache-2.0 |
| transformers | Apache-2.0 |
| numpy | BSD-3-Clause |
| sounddevice | MIT |
| **soxr** | **LGPL-2.1-or-later** |
| fastapi | MIT |
| uvicorn | BSD-3-Clause |
| websockets | BSD-3-Clause |
| pydantic | MIT |
| httpx | BSD-3-Clause |

Note that `coqui-tts` (the implementation, MPL-2.0) and XTTS-v2 (the model,
CPML) are separately licensed. Coqui the company shutting down did not change
either.

`soxr` is the only dependency here that is not permissive. It is LGPL, used as
an installed library rather than statically linked, and only for resampling
when a capture device does not offer 16 kHz. If that matters to you, a device
that captures at 16 kHz natively avoids it — the import is lazy and only
happens when a rate conversion is actually needed.

## This is not legal advice

It is a good-faith summary to help you check your own position, assembled from
the packages installed and their published terms. If commercial use is
involved, read the licences themselves.
