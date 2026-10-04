# Benchmarks

Latency is a feature of this project, not an afterthought, so each stage
records what it costs. Numbers are appended at stage boundaries and never
overwritten --- a regression should be visible.

Real-time factor (RTF) is synthesis time divided by the duration of audio
produced. **Below 1.0 means faster than real time**, which is the precondition
for streaming speech in Stage 4.

Reproduce with:

```powershell
$env:COQUI_TOS_AGREED = "1"
python scripts\benchmarkTts.py --provider xtts --device cuda
python scripts\benchmarkTts.py --provider xtts --device cpu --runs 2
```

The script discards the first synthesis after loading, which pays for CUDA
kernel compilation and cuDNN autotuning, and reports it separately.

## Reference machine

| | |
|---|---|
| CPU | 12th Gen Intel Core i7-12700H |
| GPU | NVIDIA GeForce RTX 3070 Laptop, 8 GiB VRAM, driver 576.88 |
| OS | Windows 11 |
| Python | 3.13.0 |
| torch | 2.13.0+cu126 |
| coqui-tts | 0.27.5 |

## Stage 1 --- Text to speech

### xtts-v2 (measured 2026-08-14)

Voice: built-in speaker "Claribel Dervla". Model resident throughout; the
figures below are for a single loaded instance.

| Metric | CUDA | CPU |
|---|---|---|
| Cold load (first run, includes 1.87 GB download) | 181.9 s | --- |
| Warm load (model cached on disk) | 26.6 s | 25.1 s |
| First utterance after load | 1.48 s / 1.15 s audio (RTF 1.28) | 3.84 s / 1.29 s audio (RTF 2.97) |
| Short utterance, warm (~1.4 s audio) | 0.53 s mean, 0.45 s best | 3.91 s mean, 3.87 s best |
| Long utterance, warm (~14 s audio) | 4.51 s mean, 4.20 s best | 36.21 s mean, 34.89 s best |
| **RTF, short** | **0.40** | **2.74** |
| **RTF, long** | **0.32** | **2.70** |
| VRAM after load | 1.78 GiB allocated, 1.79 GiB reserved | --- |
| VRAM after synthesis | 1.79 GiB allocated, 2.77 GiB reserved | --- |

Speaker embedding timings are not listed because built-in speakers ship with
their conditioning latents precomputed. The compute-once-and-cache path applies
to cloned voices and is still to be measured against a real sample.

### Conclusions

**CUDA is comfortably viable.** At RTF 0.32--0.40 there is roughly a 2.5x
margin over real time, which is what Stage 4 needs to begin speaking before the
language model has finished generating.

**CPU is not viable for conversation.** At RTF ~2.7, a ten-second reply takes
twenty-seven seconds to synthesise. This is the outcome PLAN.md Stage 1
identified as a decision point, so the contingency now applies: the CPU
fallback path should be backed by a lighter model (Piper is the leading
candidate --- permissive licence, no cloning) rather than by XTTS on CPU. XTTS
on CPU remains usable for offline batch generation, not for live interaction.

**The first utterance costs about 3x the warm rate on CUDA.** Stage 3 should
synthesise a throwaway phrase at startup so the first real interaction of a
session is not the slowest one.

**2.77 GiB reserved tightens the Stage 4 VRAM budget.** Reserved memory, not
allocated, is what blocks other models. With 8 GiB total, XTTS plus
faster-whisper leaves roughly 4 GiB, which strengthens the case for running the
language model on CPU and keeping the GPU for speech.

### tone (synthetic reference provider)

Not speech, so quality is meaningless --- it exists as a floor for the
non-model parts of the path (buffer handling, WAV encoding, playback).

| Metric | Value |
|---|---|
| Load time | 0.0 s |
| "Turn the kitchen light off." | 0.02 s for 1.72 s of audio |
| RTF | 0.01 |

## Stage 2 --- Speech recognition

Reproduce with:

```powershell
python scripts\benchmarkStt.py --device cuda
python scripts\benchmarkStt.py --device cpu
python scripts\benchmarkStt.py --check-microphone
```

### faster-whisper, model `small` (measured 2026-08-14)

Input is `tests/data/utterance.wav`, 2.59 s of synthesised speech reading
*"Turn the kitchen light off and set a timer for ten minutes."*

| Metric | CUDA (float16) | CPU (int8) |
|---|---|---|
| Cold load (first run, includes 480 MB download) | 46.8 s | --- |
| Warm load (model cached on disk) | 2.1 s | 3.9 s |
| First transcription after load | 0.44 s (RTF 0.17) | 1.55 s (RTF 0.60) |
| Warm transcription | 0.25 s mean, 0.23 s best | 1.47 s mean, 1.45 s best |
| **RTF, warm** | **0.10** | **0.57** |
| Word error rate vs reference | 0.0 | 0.0 |
| GPU memory added | +0.47 GiB | --- |

Both devices transcribed the reference exactly. Whisper writes "10" where the
reference says "ten", which the word-error-rate comparison normalises away ---
a formatting choice, not a recognition error.

### Conclusions

**Whisper is cheap compared with XTTS, on both devices.** At RTF 0.10 on CUDA
it is effectively free within the latency budget, and at RTF 0.57 on CPU it is
still faster than real time.

**This changes the Stage 4 plan.** Stage 1 concluded that the language model
should run on CPU because XTTS needs the GPU. Whisper being viable on CPU at
RTF 0.57 offers a better split: **Whisper on CPU, XTTS on GPU**, freeing about
5.5 GiB of the 8 GiB for a quantised language model. Speech recognition happens
while the user is still finishing their sentence, so its latency is the easiest
to hide; synthesis latency is the hardest, and that is what the GPU should
protect. Decide with the Stage 3 round-trip figures.

**Whisper loads far faster than XTTS** (2.1 s against 26.6 s), so the startup
cost of the pair is dominated by synthesis.

**The VAD filter suppresses hallucination on silence.** Without it Whisper
invents plausible text from nothing; an integration test covers this, because
a push-to-talk capture that catches only silence is an ordinary occurrence.

### Microphone note for the reference machine

The default input device reports digital silence (peak 0.0000) --- it resolves
to a virtual Steam device. Device 2, *Microphone (Realtek(R) Audio)* under MME,
captures normally (peak 0.0207 of ambient room noise). Set `audio.inputDevice`
to 2. The DirectSound variant, device 11, is also silent.

## Stage 3 --- Conversation loop

Reproduce with:

```powershell
python scripts\benchmarkTurn.py
python scripts\benchmarkTurn.py --no-warm-up
python scripts\benchmarkTurn.py --stt-device cpu --tts-device cuda
```

The headline figure is **time to first audio**: the gap between the user
finishing and the assistant starting to speak. Playback duration is excluded,
because a four-second reply does not feel like a four-second wait.

### Full turn, both models on CUDA (measured 2026-08-14)

Input is the 2.59 s fixture; the rule handler echoes it, producing 4.03 s of
reply audio.

| Stage | Warm turn |
|---|---|
| Capture | 0.00 s (file input) |
| Transcription | 0.25 s |
| Handler | 0.000 s |
| Synthesis | 1.32 s |
| **Time to first audio** | **1.56 s mean, 1.48 s best** |

State sequence observed: `listening -> thinking -> speaking -> idle`.

### What warm-up buys

| | First turn | Warm turns |
|---|---|---|
| With warm-up | 1.67 s | 1.36--1.56 s |
| Without warm-up | 2.75 s | 1.36 s |

Warm-up removes about **1.1 s from the first turn** of a session and costs a
few seconds at startup, where nobody is waiting on a reply. Startup is
25--28 s either way, dominated by loading XTTS.

### Conclusions

**Synthesis is 84% of the wait.** Transcription is 0.25 s and the deterministic
handler is immeasurable; XTTS accounts for 1.32 s of a 1.56 s turn. Any effort
spent reducing round-trip latency belongs there, which makes streaming
synthesis the highest-value item in Stage 4 rather than a nice-to-have.

**The deterministic path costs nothing.** The rule handler returns in under a
millisecond, confirming §12's premise that Layer 1 commands need no model. This
is the baseline Stage 5 should compare tool calls against.

**Whisper belongs on the GPU after all --- this reverses the Stage 2
conclusion.** Measured in isolation, Whisper on CPU looked comfortable at
RTF 0.57. Measured inside a real turn with XTTS resident, the same
transcription takes 2.46 s rather than 0.25 s, pushing time to first audio from
1.56 s to 3.79 s. Some of that is ordinary variance (an isolated re-run gave
1.87 s, not 1.47 s) and some is contention for CPU threads with torch. Either
way, moving Whisper off the GPU costs more than two seconds per turn to save
0.47 GiB, which is a bad trade.

Revised VRAM position for Stage 4: **both speech models on GPU** (about 3.2 GiB
of 8 GiB reserved between them), leaving roughly 4.5 GiB for a language model.
That fits a 3--4B model at Q4 comfortably, or a 7--8B at Q4 very tightly. If a
larger model is wanted, the language model --- not Whisper --- should be the
thing that moves to CPU, because streaming hides token latency in a way it
cannot hide transcription latency.

**Measure in situ.** The Stage 2 conclusion was drawn from a benchmark that
loaded one model at a time, and it did not survive contact with a real turn.

## Stage 4 --- Language model and streaming

Reproduce with:

```powershell
python scripts\benchmarkStreaming.py
python scripts\benchmarkStreaming.py --llm-provider ollama --model qwen2.5:3b
```

### Streaming against complete generation (measured 2026-08-14)

Real XTTS on CUDA. The language model is simulated at 25 ms per word (about
40 tokens/second, plausible for a small quantised model on this GPU), so that
the comparison isolates the effect of overlapping synthesis with generation.
The reply is three sentences, roughly 45 words.

| | Time to first audio |
|---|---|
| Without streaming | 5.93 s mean, 4.26 s best |
| **With streaming** | **1.70 s mean, 1.46 s best** (3 pieces) |
| **Improvement** | **4.23 s faster, 71%** |

With the tone provider substituted for XTTS, isolating the overlap of
generation with synthesis alone: 0.91 s against 0.37 s, the same 60% share.

### Conclusions

**Streaming is worth more than every other optimisation in the project so
far.** Waiting for a complete three-sentence reply costs 5.93 s before any
sound; speaking the first sentence while the rest is still being written costs
1.70 s. The saving grows with reply length, because the non-streaming path
waits for generation *and* synthesis of the whole thing while the streaming
path waits only for the first sentence of each.

**It also holds round-trip latency roughly flat against Stage 3.** Stage 3
reached first audio in 1.56 s with an instant rule-based reply. Stage 4 adds a
language model and still reaches it in 1.70 s. Without streaming, adding the
model would have taken the turn to nearly six seconds --- worse than every
gain made in Stages 1 to 3 combined.

**A note on how these numbers were arrived at.** The first version of this
benchmark reported a 2.48 s improvement, because the simulated model returned
complete replies instantly while streaming them word by word. The
non-streaming baseline therefore paid nothing for generation, and the
comparison understated the benefit. `ScriptedLlmProvider.generate` now sleeps
for the same total time it would take to stream, and a test pins that. A
benchmark whose control is faster than reality is worse than no benchmark.

**Sentence length is the tuning knob.** Pieces below about 20 characters sound
clipped, and XTTS gives them poor prosody; pieces above about 200 characters
give back the latency streaming saves. Those are the defaults in
`app/assistant/sentences.py`.

### Against a real model (qwen2.5:3b via Ollama, measured 2026-08-14)

| Metric | Value |
|---|---|
| Warm generation rate | 47--68 tokens/second |
| First stream fragment | 0.17 s |
| Time to first audio, without streaming | 2.45 s mean, 1.41 s best |
| **Time to first audio, with streaming** | **1.72 s mean, 1.23 s best** |
| Improvement | 0.73 s faster, 30% |

The 30% here against 71% with the simulated model is not a contradiction. The
simulation produced a fixed three-sentence reply at 40 tokens/second; qwen2.5
runs at 68 and, told to be brief, answered in **one sentence**. With a single
sentence there is nothing to pipeline, so streaming only overlaps generation
with synthesis rather than overlapping sentences with each other.

That is worth stating plainly: **the system prompt asking for brevity, which is
right for a voice assistant, also reduces what streaming can save.** Streaming
still helps, and helps more as replies grow, but a fast model giving short
answers is already close to the floor.

## Stage 5 --- Tools and home automation

Reproduce with:

```powershell
python scripts\benchmarkRouting.py
```

### Routing layers (measured 2026-08-14, qwen2.5:3b, simulated devices)

| Layer | Utterance | Time | Outcome |
|---|---|---|---|
| 1, deterministic | "turn off the bedroom light" | **0.000 s** | acted |
| 2/3, via model | "it is dark in the bedroom" | 0.49 s | varies, see below |
| 3, conversation | "what is the capital of Australia" | 0.26 s | answered |

Layer 1 is roughly **2000x faster** than reaching the model, and its cost is
too small to measure at millisecond resolution. That is the §12 argument
confirmed: obvious commands should not pay for a model round trip.

### Conclusions

**Deterministic routing is effectively free.** A command matching a pattern
costs microseconds and never puts model output in the path of something that
changes the physical environment. Every command a household actually repeats
should end up in Layer 1.

**Small-model tool selection is not deterministic.** Across repeated runs of
the same phrase, qwen2.5:3b variously called the tool, asked a clarifying
question, and returned neither text nor a tool call. All three are defensible
responses to an indirect request, but the last one would have left the
assistant silent, so an explicit fallback now speaks when the model produces
nothing.

**A small model will occasionally claim to have acted without acting.** In one
run it replied "I've turned the lounge lamp on" having called no tool at all.
The system prompt used for tool turns now forbids this specifically. It reduces
the behaviour rather than eliminating it, which is an argument for keeping
common commands in Layer 1 where the question does not arise.

## Stage 6 --- Wake word and voice activity

Reproduce with:

```powershell
python scripts\benchmarkWake.py --idle-seconds 20
python scripts\benchmarkWake.py --audio tests\data\wakeWord.wav --expect 3
```

### Cost of listening (measured 2026-08-14)

The figure that decides whether a device can be left running in a room.

| Metric | Value |
|---|---|
| Silero VAD, per 32 ms frame | 0.52 ms |
| openWakeWord, per 80 ms frame | 1.81 ms |
| **Idle load, both running** | **3.4% of one core** |
| Load time, first run (includes model download) | 54.6 s |
| Load time, warm | ~2 s |

Neither model touches the GPU, so this costs nothing against the 8 GiB the
speech models and language model are sharing.

### Wake-word accuracy

| Metric | Value |
|---|---|
| Detections | 3 of 3 |
| Missed | 0 |
| Spurious | 0 |
| Confidence at each detection | 0.998--1.000 |

**This is not a real-world accuracy figure.** The test audio is XTTS-synthesised
speech, and openWakeWord was trained on human recordings. It demonstrates that
the pipeline works end to end; it says little about how the wake word behaves
across a room, with background noise, in a real voice. That needs a human
recording to measure.

The three distractor phrases in the same file, one of which begins "Hey there",
produced no false accepts.

### Conclusions

**Two bugs found by measuring rather than by testing.** Both are recorded here
because in each case the first measurement looked like a model problem and was
not.

The wake-word refractory period was measured against the **wall clock**. Live
that is indistinguishable from audio time, but replaying an 18-second recording
takes a fraction of a second, so every detection after the first was suppressed.
The first accuracy run reported 1 of 3 and looked like a model failing on
synthetic speech; the model had in fact detected all three at 0.999 confidence.
It now counts audio time, accumulated from the frames.

`audio.inputDevice` set to `"2"` selected the **wrong device**. PortAudio reads
an integer as an index and a string as a name to match, and both environment
variables and JSON deliver `"2"` as a string, so it matched a WDM-KS device and
failed to open. Digit-only values are now converted to integers.

**A pre-roll buffer is not optional.** Speech is always recognised a fraction
of a second after it starts, so without keeping the preceding frames the first
word is clipped. Ten frames, about 320 ms, is the default.

**A minimum utterance length is doing real work.** Without it a cough or a
door becomes a transcription request. 0.3 seconds is the default.

## Stage 7 --- The API

### Measured against a live server (2026-08-15)

Both speech models on CUDA, qwen2.5:3b through Ollama, simulated devices.

| Request | Time | Notes |
|---|---|---|
| `GET /status` | 0 ms | |
| `GET /automation/devices` | 4 ms | cached device list |
| `POST /automation/execute` | 3 ms | permitted action |
| `POST /automation/execute` | 2 ms | refused, 403 |
| `POST /assistant/message`, deterministic | 1108 ms | Layer 1, `usedLlm: false` |
| `POST /assistant/message`, conversational | 8532 ms | see below |
| `POST /speech/synthesise` | 1446 ms | |
| `POST /speech/transcribe` | 948 ms | 2.59 s of audio |

Startup is about 80 seconds, almost all of it loading XTTS and Whisper. After
that the models stay resident: `/status` reported the same providers and a
rising uptime across every request.

### Conclusions

**The language model is the slowest part of a conversational turn**, at 7.4 s of
an 8.5 s request against 0.3 s when measured alone in Stage 5.

> **This was diagnosed as VRAM contention, and that was wrong.** It was a cold
> model load. See the Stage 10 section below for the measurements that
> settled it and the fix, which brought the same request to 0.9 s.

**Deterministic commands are unaffected**, at 1.1 s end to end including
synthesis, because they never reach the model.

**A busy port used to cost eighty seconds to discover.** uvicorn binds after
startup completes, so the port conflict surfaced only once both models were
resident. The address is now probed before anything loads, and the error is
immediate.

**The response event was being published after the audio finished playing.** A
client would have shown the reply text only once it had stopped being spoken.
It is now announced before synthesis begins, which the WebSocket test pins.

## Stage 8 --- Remote audio clients

Reproduce with:

```powershell
python scripts\benchmarkRemote.py --server http://192.168.1.10:8000 --token ...
```

### Over the network (measured 2026-08-15)

Server bound to the LAN address, client connecting to it over the network
stack. Utterance is the 2.59 s fixture; the assistant runs the router with
qwen2.5:3b and simulated devices.

| Metric | Value |
|---|---|
| Utterance as pcm16 | 81 KB for 2.59 s |
| Upload | **0.004 s** |
| Full round trip | 2.14 s mean, 1.40 s best |
| Of which the assistant itself | 2.12 s |
| **Added by the network** | **0.015 s** |
| Reply audio returned | 218 KB |

### Conclusions

**The network costs 15 ms.** Everything else in the round trip is the assistant
thinking. The variance between the mean and the best run is the language model,
not the link. A remote client is, on a home network, indistinguishable from a
local one.

**Raw PCM is the right choice, and Opus would be a waste.** 16 kHz mono is
32 KB/s, and because the client runs its own wake word and voice activity
detection, it only transmits during an utterance --- 81 KB for this request,
uploaded in four milliseconds. An encoder, a decoder, a dependency and the
latency of both would buy nothing measurable here. The format is announced in
the control message rather than assumed, so Opus can be added if this ever
crosses the internet, where the calculation is different.

**Server-side timing is reported to the client.** The reply header carries
`serverSeconds`, so a client can subtract it from its own round trip. Without
that, a slow model and a slow link are indistinguishable from the client's
side, which is exactly the question anyone debugging a remote device asks
first.

### Caveat on the measurement

Client and server ran on the same machine, communicating over the LAN
interface rather than loopback. That exercises the real network stack,
serialisation and framing, but not a physical hop: no switch, no Wi-Fi
contention, no second machine's scheduler. **A genuine second device will add
more than 15 ms**, though on a home network the difference should stay small
against a two-second turn.

## Stage 9 --- The web client

### Why no C++

The brief permits native code "only where profiling or deployment requirements
justify it". Collecting what eight stages measured:

| Component | Cost | Bottleneck? |
|---|---|---|
| Voice activity detection | 0.52 ms per 32 ms frame | no |
| Wake word | 1.81 ms per 80 ms frame | no |
| Idle listening, both | 3.4% of one core | no |
| Network, remote client | 15 ms per turn | no |
| Speech recognition | 0.25 s per turn | no |
| **Synthesis** | **1.32 s, 84% of a rules turn** | **yes** |
| **Language model** | **7.4 s of an 8.5 s turn** | **yes** |

Neither bottleneck is Python. Synthesis is XTTS on the GPU, and the model runs
in another process entirely; rewriting the orchestration around them in C++
would change nothing measurable. The decision is to leave it, and revisit if
this ever targets hardware smaller than a Raspberry Pi.

### Verified in a real browser

Driven with headless Chromium, using a WAV file as a fake microphone so the
capture path is exercised rather than assumed.

| Path | Result |
|---|---|
| Page load, connect, session assigned | works |
| Typed message, deterministic command | 0.01 s |
| Typed message, conversational | 0.52 s |
| Push-to-talk via `getUserMedia` and `AudioWorklet` | transcribed correctly |
| Ambiguous spoken command | asked which light, as Stage 5 intends |
| Device list and controls | works |
| Console errors | none |

### Three defects the screenshot caught and the tests did not

All three passed every assertion written about them, and all three were obvious
the moment the page was looked at.

**"Hallway Hallway light".** The page rebuilt a display name by joining area and
name, duplicating a rule the server already had --- and reintroducing the exact
bug Stage 5 had already fixed once. The API now returns `describedName`, so the
rule lives in one place.

**Illegible timings.** The per-turn timing used the muted grey that works on the
page background, inside a bubble filled with the accent colour. It rendered as
dark green on green.

**Ambiguous device buttons.** A chip reading "Hallway light --- On" gives no way
to tell whether "On" is the state or the action offered. The buttons now say
"Turn on" and "Turn off".

The lesson is not that the tests were poor --- they caught real things --- but
that no assertion about markup tells you what a page looks like. One screenshot
found three defects immediately.

## Stage 10 --- The language model bottleneck

Reproduce with:

```powershell
python scripts\diagnoseLlm.py
python scripts\diagnoseLlm.py --load-speech
```

### Correcting the Stage 7 diagnosis

Stage 7 measured a conversational turn at 7.4 s and attributed it to VRAM
contention: XTTS and Whisper hold about 3.2 GiB, so Ollama was assumed to be
evicting and reloading around them. **That was wrong**, and the conclusion drawn
from it --- that the language model should move to CPU --- was wrong with it.

Three candidate explanations were measured separately.

| Candidate | Measurement | Verdict |
|---|---|---|
| Prompt size (tool schemas) | 1181 extra characters costs **+0.01 s** | not the cause |
| VRAM contention | 0.21 s alone, **0.23 s** with speech models resident, 100% GPU either way | not the cause |
| Cold model load | **3.31 s** cold against **0.20 s** warm | the cause |

Ollama reported `100% GPU` in every configuration, including when the speech
models were loaded first and it had to fit around them. Total use peaked at
4984 MiB of 8192. There was never any contention to find.

The 7.4 s was the first language-model request of the process paying to load
qwen2.5 from disk. The 3.31 s measured here is a reload with the file still in
the operating system's page cache; a genuinely cold one is slower, which is why
Stage 7 saw more.

### The fix

The runtime had warmed XTTS and Whisper since Stage 3, and the reason it did
not warm the language model is that the model is not the runtime's to load:
Ollama loads on first request, not when its service starts. The only way to pay
that cost early is to make a request. `ResponseHandler.warmUp` does, alongside
the existing speech warm-up.

| | Before | After |
|---|---|---|
| First conversational request after startup | 7.37 s | **0.89 s** |
| Subsequent requests | 0.3 s | 0.26--0.31 s |
| Added to startup | --- | 4.8 s, in parallel with a 25 s model load |

The startup cost is free in practice: it runs concurrently with loading the
speech models, which take five times longer.

### Residency, which mattered more than the first turn

Ollama's default is to unload a model five minutes after its last use. An
assistant spoken to every ten minutes therefore paid the cold load **on every
single turn**, not just the first --- a worse problem than the one being
investigated, and invisible in any benchmark that runs its requests
back to back.

`llm.keepAlive` now defaults to `30m`, which keeps ordinary household use warm
while still releasing 2.2 GiB on a machine that is also used for other things.
Set it to `-1` on a dedicated assistant machine to keep the model resident
indefinitely, which is what §18 of the brief asks for.

### Where the time goes now

A full spoken turn through the API, measured over the network:

| Stage | Time |
|---|---|
| Upload | 0.005 s |
| Transcription | ~0.25 s |
| Language model | 0.26--0.31 s |
| **Synthesis** | **~1.3 s** |
| Network | 0.014 s |
| **Total** | **2.25 s** |

**Synthesis is the bottleneck again**, as it was in Stage 3, and the language
model is no longer close. Deterministic commands remain immeasurably fast at
0.000 s, roughly 2000x quicker than reaching the model.

### The lesson

The Stage 7 explanation was plausible, fitted the evidence available, and was
wrong. VRAM contention was a real thing that could have been happening, on a
card that was genuinely close to full, in a system that genuinely had three
models on it. What it lacked was a measurement that distinguished it from the
alternatives --- and the measurement took one script and ten minutes.

## Stage 11 --- The synthesis bottleneck

Reproduce with:

```powershell
python scripts\diagnoseTts.py
```

### Where synthesis time went

| Reply | Length | Time | Audio | RTF |
|---|---|---|---|---|
| Confirmation | 34 chars | 0.57 s | 1.48 s | 0.38 |
| Short answer | 37 chars | 0.93 s | 2.08 s | 0.45 |
| Two sentences | 65 chars | 1.37 s | 3.48 s | 0.39 |
| Conversational | 148 chars | 2.76 s | 8.12 s | 0.34 |

Fitted, this is about **19 ms per character with no meaningful fixed cost per
call**. There is no per-call overhead to amortise, which is what makes cutting
an utterance into smaller pieces viable rather than wasteful.

The sentence-level streaming built in Stage 4 could not help here, because it
has nothing to pipeline when the reply is one sentence --- and most of what a
home assistant says is one sentence.

### Chunk-level streaming

XTTS decodes autoregressively, so the beginning of an utterance exists long
before the end. `inference_stream` exposes that. Sweeping the chunk size:

| Reply | Whole | size 20 | size 10 | size 5 | size 3 |
|---|---|---|---|---|---|
| Short (34 chars) | 0.54 s | 0.35 s | **0.19 s** | 0.12 s | 0.09 s |
| One sentence (105) | 1.72 s | 0.35 s | **0.17 s** | 0.13 s | 0.05 s |
| Long (149 chars) | 2.48 s | 0.33 s | **0.38 s** | 0.11 s | 0.06 s |

Smaller chunks reach sound sooner but cost more in total: the long reply takes
2.48 s whole, 3.57 s at size 20 and 6.23 s at size 3. What matters is that
generation stays ahead of playback, and at size 3 a 7.62 s reply takes 6.23 s
to make --- close enough to real time to risk stuttering. **10 is the default**:
first sound in about 0.2 s, generation comfortably ahead.

### Measured on the running assistant

Same server, same replies, changing only `textToSpeech.streamChunkSize`.

| Reply | Streaming off | Streaming on | |
|---|---|---|---|
| "Hello. How can I help?" | 0.653 s | **0.257 s** | 2.5x |
| A longer capability answer | 2.674 s | **0.139 s** | **19x** |

Total synthesis rises slightly --- 2.61 s to 3.55 s on the longer reply --- and
that is the trade being made: more total work for far less waiting.

Remote clients benefit too, because `AudioOutput` already had the right shape.
The runtime calls `play` once per piece, so an output that forwards to a
WebSocket streams to it with no new plumbing. A remote client now hears audio
at **0.95 s** rather than waiting 3.47 s for a complete reply.

### A bug the timing sweep could not have caught

`inference` returns a numpy array; `inference_stream` yields tensors **still on
the GPU**. The sweep measured the generator without converting its output, so
it ran clean --- and the first real turn failed with "can't convert cuda:0
device type tensor to numpy". Anything tensor-shaped is now brought back to the
host before conversion, with a test for it.

A benchmark that exercises only the expensive part of a path will not find the
defects in the cheap part.

### The playback bug this introduced

Streaming synthesis made a reply arrive as a dozen pieces instead of one, and
`SoundDeviceOutput.play` opened a device for each. Reported as speech being
"jittery, only saying the first part of each word".

| Playing 9 pieces totalling 3.60 s of audio | Wall clock | Ratio |
|---|---|---|
| A fresh `sounddevice.play` per piece | 7.13 s | 1.98 |
| **One stream, written into** | **3.75 s** | **1.04** |

About 0.4 s of device open and close between every piece, so the reply played
at half speed with silences through the middle of words. Audio was **never
lost** --- the check that ruled that out compared the streamed pieces against a
whole-utterance synthesis and found matching content, and compared the tail of
each piece against the head of the next to rule out duplication too. The
content was always correct; only the delivery was broken.

`SoundDeviceOutput` now keeps one `OutputStream` open and writes into it.
Blocking on the write also gives backpressure, so synthesis is paced by
playback instead of racing ahead of it --- which is why `synthesisSeconds` now
covers speaking as well as generating, and `firstAudioSeconds` is the figure
that describes what a listener experiences.

**Every benchmark in this document had missed it**, because they all use the
WAV-file backend so that they can run without a speaker. A measurement harness
built to avoid hardware will not exercise the hardware path.

### Where the time goes now

| Stage | Time |
|---|---|
| Transcription | ~0.25 s |
| Language model | 0.26--0.52 s |
| Synthesis, to first sound | **0.14--0.26 s** |
| Network, remote | 0.005 s |
| **To first audio, local** | **~0.5 s** |

Against Stage 3's 1.56 s and Stage 7's 8.5 s. No single component now dominates:
the language model and synthesis cost about the same, and transcription is
comparable to both. That is the point at which further work should stop unless
something changes.
