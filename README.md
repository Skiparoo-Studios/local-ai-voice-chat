# Local Voice Assistant

A local-first voice assistant: speech recognition, language understanding, home
automation and speech synthesis, running on your own hardware with replaceable
providers at every layer.

See `local_voice_assistant_project_architecture.md` for the architecture and
`PLAN.md` for the staged implementation plan.

> **Licensing — read this before any commercial use.**
>
> This project's own code is MIT (`LICENSE`). **No model weights are
> distributed in this repository**; they download at runtime.
>
> This project supports XTTS-v2 as a text-to-speech backend. XTTS-v2 is a
> third-party model licensed separately under the Coqui Public Model Licence
> (CPML). The CPML restricts the model **and its outputs** to non-commercial
> uses. The licence of this project does not grant any rights to use XTTS-v2
> commercially. Users are responsible for ensuring their use of third-party
> models complies with the applicable licence.
>
> XTTS is one provider behind `TextToSpeechProvider`, not the assistant itself.
> A deployment that selects a different synthesiser never downloads it and is
> not subject to the CPML. Every other model here — Whisper, Silero, openWake-
> Word — is permissively licensed. See `NOTICE.md`.

**Current stage: 9 --- clients.** Say "hey Jarvis" and it listens, answers in
about 1.6 seconds, and goes back to waiting. Entirely local. It can control
lights and switches, resolving obvious commands without troubling the language
model at all. It runs as an HTTP and WebSocket service with a browser client,
and other machines can act as microphones and speakers for it without hosting
any models themselves.

## Requirements

- Python 3.12 or newer (developed against 3.13)
- Optional: an NVIDIA GPU for practical speech-model performance

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Provider dependencies install separately, so a machine only needs the parts it
runs:

```powershell
pip install -e ".[tts]"          # Coqui XTTS, torch, audio output
pip install -e ".[stt]"          # faster-whisper, audio input
pip install -e ".[llm]"          # local and remote language models
pip install -e ".[wake]"         # wake word and voice activity detection
pip install -e ".[automation]"   # Home Assistant, MQTT
pip install -e ".[api]"          # FastAPI service
```

## Running

### Speaking (Stage 1)

```powershell
python -m app.main                                    # interactive prompt
python -m app.main --provider tone                    # no model needed
python -m app.main --say "Hello there."               # speak one line and exit
python -m app.main --list-devices                     # audio outputs
python -m app.main --config config\settings.json --log-level DEBUG
```

At the prompt, type text to hear it spoken. Commands:

```text
:help            show help
:voices          list available voices
:voice <name>    switch voice
:save [path]     save the last utterance as a WAV file
:info            provider, device and voice
:quit            exit (Ctrl-C and Ctrl-D also work)
```

The model loads once at startup and stays resident, so only the first prompt
pays for it. Each utterance reports its real-time factor --- below 1.0 means
synthesis is faster than playback.

### The tone provider

`--provider tone` produces tones rather than speech, with a pitch derived from
the voice name. It needs no model, no download and no GPU, so it is the quickest
way to check that audio output works, and it is what the tests use.

### Listening (Stage 2)

```powershell
python -m app.main --mode listen                      # push-to-talk prompt
python -m app.main --transcribe recording.wav         # transcribe a file and exit
python -m app.main --mode listen --provider scripted  # no model needed
python -m app.main --list-input-devices               # audio inputs
```

Press Enter to start recording, speak, then press Enter again to stop. Commands:

```text
:help            show help
:save [path]     save the last recording as a WAV file
:file <path>     transcribe a WAV file instead of the microphone
:info            provider, device and language
:quit            exit
```

Check the microphone is actually delivering audio before wondering why nothing
transcribes:

```powershell
python scripts\benchmarkStt.py --check-microphone
```

If the peak level is effectively zero, the default input is likely a virtual
device. List the real ones and set `audio.inputDevice`:

```powershell
python -m app.main --list-input-devices
```

`--provider scripted` returns canned phrases instead of transcribing, the
counterpart to the tone provider.

### Conversing (Stage 3)

```powershell
python -m app.main --mode converse                    # speak, hear a reply
python -m app.main --mode converse --no-microphone    # type instead of speaking
```

Press Enter to speak, then Enter again when you have finished. Typing a message
instead runs the same turn without the microphone. Commands:

```text
:help            show help
:history         the conversation so far
:clear           forget the conversation
:info            providers, session and state
:quit            exit
```

Each turn reports where its time went:

```text
  you:       hello there
  assistant: Hello. How can I help?
  stt 0.25s  think 0.00s  tts 1.32s  = 1.57s to first audio
```

Both models load at startup and stay resident. A warm-up runs automatically,
which removes about a second from the first turn; disable it with
`assistant.warmUpOnStart` if you would rather start faster.

### Language model replies (Stage 4)

Replies come from `assistant.handler`:

| | |
|---|---|
| `rules` (default) | a handful of deterministic phrases, no model |
| `llm` | the configured language model |
| `echo` | repeats what was said |

```powershell
python -m app.main --mode converse --handler llm
python -m app.main --mode converse --handler llm --no-streaming
```

With Ollama:

```powershell
ollama pull qwen2.5:3b
```

```json
{
    "assistant": { "handler": "llm" },
    "llm": { "provider": "ollama", "model": "qwen2.5:3b" }
}
```

Any OpenAI-compatible server works too --- llama.cpp, LM Studio, vLLM --- by
setting `llm.provider` to `openai-compatible` and `llm.baseUrl` to its address
(usually ending in `/v1`). API keys belong in `VOICE_LLM__APIKEY`, never in the
file.

**Keeping the model resident matters more than it looks.** Ollama unloads a
model five minutes after its last use, and loading it again costs about three
seconds --- so an assistant spoken to every ten minutes would pay that on every
single turn. `llm.keepAlive` defaults to `30m` for that reason. On a machine
dedicated to the assistant, set it to `-1` and the model never unloads:

```json
{ "llm": { "keepAlive": "-1" } }
```

The model is also warmed at startup, alongside the speech models, so the first
thing you say does not pay to load it.

**Replies start being spoken before they are finished**, at two levels. Each
sentence is synthesised as the model generates it, and each sentence starts
playing before *it* is finished --- XTTS produces the beginning of an utterance
long before the end.

Together those take a short reply from 0.65 s to first sound down to 0.26 s,
and a longer one from 2.67 s to 0.14 s. Turn the first off with
`assistant.streaming` or `--no-streaming`; the second with
`textToSpeech.streamChunkSize`:

```json
{ "textToSpeech": { "streamChunkSize": 10 } }
```

Smaller reaches sound sooner but costs more in total. Below about 5 the model
generates barely faster than the audio plays, and a long reply can stutter. `0`
waits for whole utterances.

The system prompt asks for short, plain replies, because a model left to its
own habits answers with headings and bullet points, which sound terrible read
aloud. Markdown that arrives anyway is stripped before synthesis. Override the
prompt with `assistant.systemPrompt`.

### Home automation (Stage 5)

Set `assistant.handler` to `router` and it can control devices:

```powershell
python -m app.main --mode converse --handler router
```

Try it against a simulated house first, which needs no Home Assistant:

```json
{
    "assistant": { "handler": "router" },
    "automation": { "provider": "simulated" },
    "llm": { "provider": "ollama", "model": "qwen2.5:3b" }
}
```

```text
> turn off the hallway light
  assistant: I've turned the hallway light off.
  stt 0.00s  think 0.00s  tts 0.76s  = 0.76s to first audio

> it is dark in the bedroom
  assistant: I've turned the bedroom ceiling light on.
```

The first went through **deterministic pattern matching, with no model
involved** --- roughly 2000 times faster than asking one. The second needed the
model to work out what was meant.

For real devices, set `automation.provider` to `home-assistant`, give
`automation.baseUrl` the address of your instance, and put a long-lived access
token in the environment:

```powershell
$env:VOICE_AUTOMATION__ACCESSTOKEN = "your-token"
```

#### What it is allowed to do

The assistant can act on the physical environment, so what it may do is an
explicit list rather than whatever the model proposes:

```json
{
    "automation": {
        "allowedActions": ["light.*", "switch.turnOn", "switch.turnOff"],
        "allowStateChanges": true
    }
}
```

- Anything not matching `allowedActions` is refused and logged, whether it came
  from a pattern or from the model. **Locks, covers and alarms are absent by
  default** and have to be added deliberately.
- `allowStateChanges: false` stops everything that changes the world while
  leaving read-only tools working.
- Every invocation is validated against the tool's schema before it runs, so
  malformed model output is rejected rather than partially applied.
- Ambiguous requests ask rather than guess: "turn off the kitchen light" with
  two kitchen lights replies "Did you mean the Kitchen Ceiling light or the
  Kitchen Under cabinet light?"

Be aware that a small model will occasionally *say* it has done something
without calling the tool. The prompt used for tool turns forbids this
explicitly, which reduces it but does not eliminate it. Commands you repeat
often are better added to the deterministic patterns, where the question cannot
arise.

### Hands-free (Stage 6)

```powershell
pip install -e ".[wake]"
python -m app.main --mode wake
```

Say **"hey Jarvis"**, then speak. Capture ends when you stop talking rather
than after a fixed time, and saying the wake word again while it is speaking
interrupts it.

```json
{
    "wakeWord": {
        "provider": "openwakeword",
        "word": "hey_jarvis",
        "threshold": 0.5,
        "allowBargeIn": true
    },
    "vad": {
        "provider": "silero",
        "silenceFrames": 25
    }
}
```

The pretrained wake words are `alexa`, `hey_jarvis`, `hey_mycroft`,
`hey_rhasspy`, `timer` and `weather`; a path to your own `.onnx` model works
too. They download once on first use.

Tuning, if it feels wrong:

| Symptom | Setting |
|---|---|
| Cuts you off mid-sentence | raise `vad.silenceFrames` (each is 32 ms) |
| Waits too long after you finish | lower `vad.silenceFrames` |
| First word missing | raise `vad.prerollFrames` |
| Wakes when it shouldn't | raise `wakeWord.threshold` |
| Doesn't wake | lower `wakeWord.threshold` |
| Reacts to coughs | raise `vad.minimumSeconds` |
| Answers itself | raise `audio.echoGuardSeconds`, or use headphones |

Set `wakeWord.provider` to `always-awake` to skip the wake word and treat every
utterance as addressed to the assistant --- reasonable with a headset, not with
a device in a room.

#### Not listening to itself

Capture never stops, so when a turn ends the microphone has two things still
arriving: audio recorded while the assistant was speaking and still queued, and
the tail of the reply itself. PortAudio's `write()` returns with about 180 ms
left in the device buffer, which the room then reverberates. With
`always-awake` there is no wake word to wait for, so the listener goes straight
back to capturing into both — and the assistant answers itself.

Queued audio is discarded when a turn ends, and the microphone is then ignored
for `audio.echoGuardSeconds` (0.4 by default):

```json
{ "audio": { "echoGuardSeconds": 0.4 } }
```

Raise it if the speakers are loud or close to the microphone. Set it to `0`
with headphones, where the problem does not exist and the guard only costs
responsiveness. It is deliberately not applied after a barge-in: there the
person is already mid-sentence, and a guard would swallow the start of it.

This is suppression, not acoustic echo cancellation. A speaker pointed straight
at the microphone in a bare room can still get through; headphones or a
directional microphone is the real fix.

Listening costs about **3.4% of one core** and no GPU, so it is cheap to leave
running.

**Make sure `audio.inputDevice` points at a real microphone.** On this machine
the default input is a virtual device that returns silence. Check with:

```powershell
python -m app.main --list-input-devices
python scripts\benchmarkStt.py --check-microphone
```

A numeric `audio.inputDevice` selects by index; a string selects by name.

### As a service (Stage 7)

```powershell
pip install -e ".[api]"
python -m app.main --mode serve
```

Binds to `127.0.0.1:8000`. Open **http://127.0.0.1:8000/** for the browser
client, or `/docs` for the interactive API documentation.

| Endpoint | |
|---|---|
| `GET /` | the browser client |
| `GET /health` | liveness, never authenticated |
| `GET /status` | state, session, providers, uptime |
| `POST /assistant/message` | send text, get the reply |
| `POST /speech/transcribe` | upload a WAV, get the transcript |
| `POST /speech/synthesise` | send text, get a WAV |
| `GET /automation/devices` | what can be controlled |
| `POST /automation/execute` | perform an action |
| `WS /ws/assistant` | live events, and a way to send messages |

```powershell
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/assistant/message `
     -H "Content-Type: application/json" `
     -d '{\"text\":\"turn off the hallway light\"}'
```

Models load once at startup, about 80 seconds, and stay resident.

#### Exposing it to other machines

The assistant can act on your home, so it binds to loopback and **refuses to
start on any other address without a token**:

```powershell
$env:VOICE_API__AUTHTOKEN = "a-long-random-string"
$env:VOICE_API__HOST = "0.0.0.0"
python -m app.main --mode serve
```

Every endpoint except `/health` then requires `Authorization: Bearer <token>`.
WebSocket clients may pass `?token=<token>` instead, since browsers cannot set
headers on a handshake.

The allow-list applies to API clients exactly as it does to the language model:
an action outside `automation.allowedActions` returns 403 whoever asked for it.

Request logging records method, path, status and duration, and never bodies —
those carry transcripts and replies.

#### The WebSocket

Clients receive every event: state changes, transcriptions, responses, errors.
Several clients may connect at once and all see the same assistant.

```json
{"type": "assistant.state", "state": "thinking"}
{"type": "assistant.response", "text": "I've turned the hallway light off."}
```

Send `{"type": "assistant.message", "text": "..."}` to drive it,
`{"type": "assistant.interrupt"}` to stop it speaking, or
`{"type": "assistant.status"}` to ask what it is doing.

Each WebSocket client gets **its own conversation**, so two people talking to
the assistant from different rooms do not share context. There is still only
one assistant — turns are serialised, because there is one GPU and one speaker
and two replies spoken over each other would be worse than one waiting.

### A remote microphone and speaker (Stage 8)

On the machine with the GPU:

```powershell
$env:VOICE_API__AUTHTOKEN = "a-long-random-string"
$env:VOICE_API__HOST = "0.0.0.0"
python -m app.main --mode serve
```

On the other machine — a laptop, a small PC, a Raspberry Pi:

```powershell
pip install -e ".[wake,api]"
$env:VOICE_CLIENT__TOKEN = "a-long-random-string"
python -m app.main --mode client --server http://192.168.1.10:8000
```

It loads **no speech models and needs no GPU**. Say the wake word, speak, and
the reply plays on that machine's speaker.

Check the connection without speaking:

```powershell
python -m app.main --mode client --server http://192.168.1.10:8000 --send "hello"
```

#### What crosses the wire

Audio travels as **raw 16-bit PCM in binary WebSocket frames**, bracketed by
JSON control messages on the text channel:

```text
client -> server   {"type": "audio.start", "sampleRate": 16000}
                   <binary frames>
                   {"type": "audio.end"}

server -> client   {"type": "audio.reply", "sampleRate": 24000, "serverSeconds": 2.1}
                   <binary frames>
                   {"type": "audio.replyEnd"}
```

The client runs the wake word and voice activity detection itself, so nothing
is transmitted until someone actually speaks. A 2.6 second request is 81 KB and
uploads in about four milliseconds.

**The network adds about 15 ms** to a two-second round trip on a home network,
which is why the audio is uncompressed: an Opus encoder and decoder would add
dependencies and latency to save bandwidth nobody is short of. The reply header
reports `serverSeconds`, so a client can tell a slow model from a slow link.

Measure it yourself:

```powershell
python scripts\benchmarkRemote.py --server http://192.168.1.10:8000 --token ...
```

### The browser client (Stage 9)

Open the assistant's address in any browser — phone or desktop, no install:

```text
http://127.0.0.1:8000/          on the machine running it
http://192.168.1.10:8000/       from anywhere on the network
```

Enter the token if one is configured, press **Connect**, then hold the button
(or the space bar) and speak. Typing works too. The page shows the assistant's
state, the conversation, and on-off controls for whatever it can reach.

It is one HTML file with no build step, no framework and no external requests.
Audio is captured through an `AudioWorklet` at 16 kHz and sent as the same
binary frames the Python client uses, so the two clients exercise one protocol.

For a permanent device in another room, `docs/raspberryPiClient.md` covers what
to install on a Pi, the systemd unit, and how many clients the server supports
at once (unlimited connected; one turn at a time).

Note that browsers only permit microphone access on `localhost` or over HTTPS.
Reaching the page by LAN address from another device will connect and accept
typing, but push-to-talk needs a certificate — the Python client
(`--mode client`) has no such restriction and is the better choice for a
device in another room.

### Toddler mode

A simpler assistant for a small child: short replies, and questions it knows
the answers to.

```bash
python -m app --mode converse --toddler
```

It can also be reached from any other mode by saying **"start toddler mode"**,
and left with **"end toddler mode"**, so a parent can hand the assistant over
without touching a keyboard.

```text
> hi
  assistant: Hello Sam! It's lovely to talk to you.
> hello
  assistant: Hi there! What sound does a sheep make?
> baa
  assistant: That's correct! A sheep says baa.
```

Hands-free (`--mode wake`), a child who says nothing for seven seconds gets the
answer rather than silence, then a different question, and after three of those
the assistant goes quiet until they speak again.

About 110 questions ship across thirteen topics, each rated 1 to 5. **Every
topic keeps its own level and it moves with the child**: two right answers in
a row offers harder ones, two wrong offers easier, so a child who knows their
animal sounds progresses to what a horse eats while still being asked easy
colours if colours are where they struggle.

The questions live in `config/toddler.json` and are edited from the ordinary
prompt with `:toddler`. Answers are judged against the file rather than by the
model, so "baa" is praised in milliseconds; the model handles the conversation
around them. `docs/toddlerMode.md` explains the file, the rhythm of asking, and
what to expect from a small local model.

### Reading books

Two libraries: audiobooks as folders of numbered files, and plain text read in
the assistant's own voice. Recordings win where both exist.

```powershell
pip install -e ".[books]"
```

```json
{ "books": { "audioDirectory": "books/audio", "textDirectory": "books/text" } }
```

```text
you: read me a story
  -> Sure, what book would you like me to read? I have ...
you: the chamber of secrets
  -> Which chapter, or shall I start from the beginning?
you: chapter two
  -> Chapter 2: Dobby's Warning. Say stop when you've had enough.
```

Reading is a background activity rather than a turn, so the assistant keeps
listening and "stop" is heard. Audio is decoded in blocks and paced by
playback, so a seventy-two minute chapter never sits in memory.
`docs/readingBooks.md` covers chapters, text books and what is not done yet.

### Standing down

```text
you: goodbye     -> Goodbye. Say hello when you want me.
you: ...         -> (nothing, whatever it is)
you: hello       -> Hello. I'm listening.
```

Useful with `always-awake`, where every conversation in the room otherwise
reaches the recogniser. It is the outermost layer, so it silences toddler mode
and books too, and stops a book that is playing. `docs/sentryMode.md`.

### Why there is no C++

The brief permits native code "only where profiling justifies it". Nothing
measured across nine stages does: voice activity detection costs 0.52 ms per
32 ms frame, the wake word 1.81 ms per 80 ms frame, idle listening 3.4% of one
core, and the network 15 ms per turn. The two real costs are synthesis and the
language model, neither of which is Python. `docs/benchmarks.md` has the
figures.

## Voices

Put a recording in `voices/`, as either:

```text
voices/michael.wav          a single sample
voices/michael/*.wav        several samples, which clone better
```

Then set `textToSpeech.voice` to `michael`, or use `:voice michael`. XTTS reads
up to thirty seconds; beyond that nothing further is used, so cleaner audio
beats more of it. Speaker embeddings are computed once and cached in
`voices/.cache/`, keyed on the samples so that editing them invalidates the
cache.

Check a recording before wondering why the clone is poor:

```powershell
python scripts\checkVoice.py michael
```

**`docs/recordingAVoice.md` has a script to read and what to watch for.** The
short version: record in the voice you want the assistant to have, because
XTTS copies how you speak as faithfully as how you sound.

Nothing in `voices/` is committed.

### Speech synthesis providers

XTTS is one implementation behind `TextToSpeechProvider`, not the assistant
itself. Switching is configuration:

```text
TextToSpeechProvider
    ├── XttsProvider    xtts    cloned voices. CPML, non-commercial only
    ├── PiperProvider   piper   CPU, fast, no restriction on the audio
    └── ToneProvider    tone    no model at all, for tests and headless runs
```

| | `xtts` | `piper` |
|---|---|---|
| Clones your voice | **yes** | no |
| Restricts the audio you generate | **yes**, non-commercial | no |
| Load / synthesis | 26.6 s / 1.32 s on a GPU | **2.0 s / 0.20 s on a CPU** |
| Download | ~1.8 GB | ~60 MB per voice |
| Runs on a Raspberry Pi | no | **yes** |

```powershell
pip install -e ".[piper]"
```

```json
{ "textToSpeech": { "provider": "piper", "voice": "en_GB-cori-medium" } }
```

XTTS stays the default because cloning your own voice is the thing Piper cannot
do. `docs/speechProviders.md` covers voices, speaking rate, the licence
difference and one measured quirk worth knowing about.

```json
{ "textToSpeech": { "provider": "tone" } }
```

Adding one is a registry entry and needs no change to the runtime:

```python
from app.speech.textToSpeech import registerProvider
registerProvider("piper", "mypackage.piperProvider:PiperProvider")
```

A permissively licensed synthesiser — Piper and Kokoro are the obvious
candidates — would make the whole system commercially usable without touching
anything outside one provider file. Nothing else in the project carries a
non-commercial restriction.

### XTTS licence

XTTS-v2 is published under the Coqui Public Model Licence, which permits
**non-commercial use only**. The restriction covers the model *and its
outputs*: audio generated with it is not yours to sell.

The model is not distributed here. It downloads on first use, and only after
you accept the licence explicitly:

```powershell
$env:COQUI_TOS_AGREED = "1"
```

Declining is a supported outcome — the assistant reports the licence and stops
rather than downloading anything. And if you pass on the model, a modification
of it, or **its output**, the CPML requires you to pass on the licence terms or
its URL with them.

`NOTICE.md` has the full third-party position, including every other model and
package and its licence.

## Configuration

Copy the example and edit it:

```powershell
Copy-Item config\settings.example.json config\settings.json
```

`config/settings.json` is git-ignored. Settings resolve in three layers, each
overriding the one before:

1. Defaults in `app/config/models.py`
2. The JSON file (`config/settings.json` by default)
3. Environment variables prefixed `VOICE_`, using `__` between section and key

**Secrets belong in layer 3**, never in the JSON file:

```powershell
$env:VOICE_AUTOMATION__ACCESSTOKEN = "your-home-assistant-token"
$env:VOICE_LLM__APIKEY = "..."
$env:VOICE_API__AUTHTOKEN = "..."
```

A `.env` file in the project root works too, and is git-ignored.

Unknown configuration keys are rejected rather than ignored, so a typo fails at
startup instead of silently reverting to a default.

## Security defaults

The assistant can act on the physical environment, so the API binds to
`127.0.0.1` by default. Configuring a non-loopback `api.host` without also
setting `api.authToken` is refused at startup.

## Testing

```powershell
pytest                        # unit tests; model tests skip by default
pytest -m "not integration"   # skip tests needing models or hardware
ruff check .
```

Tests that load XTTS are gated so the suite runs on a machine without models:

```powershell
$env:COQUI_TOS_AGREED = "1"
$env:VOICE_RUN_MODEL_TESTS = "1"
pytest -m integration
```

## Benchmarking

```powershell
python scripts\benchmarkTts.py --provider xtts --device cuda
python scripts\benchmarkTts.py --provider xtts --device cpu --runs 2
python scripts\benchmarkStt.py --device cuda
python scripts\benchmarkStt.py --device cpu
python scripts\benchmarkTurn.py
```

The first run after loading pays for CUDA kernel compilation, so the scripts
report it separately from the warm figures. Results are recorded in
`docs/benchmarks.md`.

On the reference machine (RTX 3070 Laptop), warm real-time factors are:

| | CUDA | CPU |
|---|---|---|
| XTTS-v2 synthesis | 0.32--0.40 | ~2.7 |
| Whisper `small` transcription | 0.10 | 0.57--0.72 |

A GPU is effectively required. A full conversation turn takes **1.56 s to first
audio** with both models on CUDA, of which synthesis is 84%.

Note that Whisper measured on CPU in isolation looks comfortable, but costs
2.2 s more per turn once XTTS is resident alongside it --- so both models
belong on the GPU. Benchmarks that load one model at a time can mislead.

## Layout

```text
app/
  api/        HTTP and WebSocket interfaces, and the service they wrap
    web/      the browser client, one self-contained page
  assistant/  runtime, sessions, conversation, handlers, intent router
  audio/      audio buffers, capture and playback devices
  client/     a microphone and speaker for a remote assistant
  automation/ home automation providers and device matching
  config/     settings schema and loading
  events/     event definitions and the in-process event bus
  intelligence/ language model providers
  speech/     speech interfaces, voice library, device selection, prompts
    models/   provider implementations (xtts, tone, faster-whisper, scripted)
  tools/      the tool interface and the home automation tool
  main.py     entry point
config/       settings.example.json (settings.json is git-ignored)
docs/         benchmarks
models/       downloaded models --- never committed
scripts/      benchmarking
voices/       voice samples and embeddings --- never committed
tests/
  data/       fixed utterance used by the transcription test
```

Directories for the assistant runtime, intelligence, automation and tools
arrive with the stages that need them, per `PLAN.md`.

Latency measurements live in `docs/benchmarks.md`.

## Licence

The code here is MIT — see `LICENSE`.

That covers the code and nothing else. **No model weights are distributed in
this repository**; every model downloads at runtime from whoever publishes it,
under that publisher's terms. `models/` and `voices/` hold only placeholders,
and the `.gitignore` keeps weights out.

Of everything the project can download, **XTTS-v2 is the only component with a
non-commercial restriction** (Coqui Public Model Licence), and it applies to
its audio output as well as the model. Whisper, Silero VAD and openWakeWord are
all permissive. Selecting a different `textToSpeech.provider` avoids the CPML
entirely.

`NOTICE.md` sets out the full position, package by package.
