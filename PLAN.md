# Local Voice Assistant --- Staged Implementation Plan

Companion to `local_voice_assistant_project_architecture.md`. That document
describes *what* the system is. This one describes the order we build it in,
what "done" means at each stage, and the decisions we defer.

## Ground Rules

- Each stage ends in something demonstrable that we can run. No stage exists
  only to produce scaffolding for a later stage.
- Abstractions are introduced at the moment a *second* implementation appears,
  or when the stage's own tests need a seam. Not before.
- We do not move to the next stage until the current one's exit criteria pass.
- Provider-specific dependencies are optional extras in `pyproject.toml`, so a
  machine that only runs the API does not need to install XTTS.

## Environment Baseline (verified 2026-08-14)

| Item | Value | Consequence |
|---|---|---|
| Python | 3.13.0 (`.venv` present, empty except pip) | Original `TTS` package (coqui-ai) has **no distribution for 3.13**. Use the maintained `coqui-tts` fork. |
| GPU | RTX 3070 Laptop, 8 GB VRAM | XTTS (~2 GB) + faster-whisper small (~1 GB) fit together. Adding a 7B LLM at Q4 (~4.5 GB) does not comfortably. See Stage 4. |
| OS | Windows 11 | CUDA support for faster-whisper needs cuBLAS/cuDNN 9 DLLs on PATH; simplest route is the `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` pip packages. |
| Shell | PowerShell | Dev scripts should not assume a POSIX shell. |

Decisions this forces in Stage 0/1, rather than discovering them mid-build:
we target `coqui-tts`, we pin torch explicitly, and CPU must be a working
fallback path from day one (the brief already requires this).

---

## Stage 0 --- Project Skeleton and Tooling *(complete)*

**Goal:** a repo that runs, lints, and tests, with nothing in it yet.

Deliverables:
- `git init`, `.gitignore` covering `models/`, `voices/`, `outputs/`, `.venv/`,
  `__pycache__/`, `*.wav`, `.env`.
- `pyproject.toml` with base deps only (`pydantic`, `pydantic-settings`), and
  optional extras declared but not yet populated: `[tts]`, `[stt]`, `[llm]`,
  `[automation]`, `[api]`.
- `app/config/` --- Pydantic settings models, loading from
  `config/settings.json` with environment-variable overlay for secrets.
- `app/events/` --- a minimal in-process event bus (publish/subscribe over
  asyncio), plus the event dataclasses from brief §8. Small enough to be worth
  having early, because every later stage emits into it.
- `tests/unit/` with tests for config loading and the event bus.
- `ruff` + `pytest` configured. Type hints everywhere; `mypy` optional.

**Exit criteria:** `pytest` passes; `python -m app.main` starts and exits
cleanly having loaded config and printed the resolved provider names.

**Note on conventions:** the brief mandates lowerCamelCase method names and
British spelling in identifiers. This is non-idiomatic for Python and will make
`ruff`'s naming rules noisy --- we disable `N802`/`N803` in config and apply it
consistently rather than half-and-half.

---

## Stage 1 --- TTS Proof of Concept (brief Phase 1) *(complete)*

**Goal:** `python -m app.main` gives a REPL that speaks what you type, in a
cloned voice, with the model loaded exactly once.

**Outcome:** met on CUDA. Warm RTF 0.32--0.40 on the RTX 3070, leaving roughly
a 2.5x margin over real time. Full figures in `docs/benchmarks.md`. Three
findings changed later stages:

1. **CPU XTTS is not viable for conversation** (RTF ~2.7). The contingency
   below now applies --- the fallback path needs a lighter model, not XTTS on
   CPU. Deferred until something actually needs to run without a GPU.
2. **The first utterance costs ~3x the warm rate**, so Stage 3 should warm the
   model at startup.
3. **XTTS reserves 2.77 GiB**, not the ~2 GiB assumed when this plan was
   written. Reserved memory is what blocks other models --- see Stage 4.

Built-in speaker support was added beyond the original scope, so the assistant
has a working voice before any recording exists. Cloned-voice quality and the
embedding cache remain unmeasured, pending a voice sample.

Deliverables:
- `app/speech/textToSpeech.py` --- `TextToSpeechProvider` ABC: `synthesise(text,
  voice, language) -> bytes`, plus `load()` / `unload()` lifecycle.
- `app/speech/models/xttsProvider.py` --- XTTS-v2 via `coqui-tts`.
- `app/audio/output.py` --- `AudioOutput` abstraction; `sounddevice` backend,
  with a "write to WAV file" backend used by tests and headless machines.
- A second, trivial provider (`nullProvider` or `pyttsx3`) so the abstraction is
  proven by an actual swap, not by assertion.
- Voice sample loading from `voices/`, speaker embedding cached to disk after
  first computation.
- Device detection: CUDA if available, else CPU, logged at startup.

**Exit criteria:**
1. Model loads once; second and subsequent prompts show no reload in the logs.
2. Switching `textToSpeech.provider` in config changes engine with no code edit.
3. Ctrl-C shuts down cleanly, releasing the audio device.
4. Cold-start and per-utterance latency recorded in `docs/benchmarks.md` for
   both CUDA and CPU. We need these numbers before Stage 3 to know our budget.

**Known risks to handle here, not later:**
- **PyTorch 2.6+ changed `torch.load` to `weights_only=True` by default**, which
  breaks XTTS checkpoint loading. Fix via `torch.serialization.add_safe_globals`
  for the XTTS config classes --- confined to `xttsProvider.py`.
- **XTTS-v2 ships under the Coqui Public Model Licence (CPML), which is
  non-commercial.** Given this is under a company account, flag it now: it is
  fine for a personal/home build, but if this ever becomes commercial we need a
  different model. The provider interface is what protects us --- candidates
  with permissive licences include Piper (fast, no cloning), Kokoro (Apache),
  and Chatterbox (MIT, supports cloning). **Decision needed from you before we
  invest heavily in XTTS-specific voice tooling.**
- XTTS is not fast on CPU. If the CPU fallback measures unusably slow, that is
  an argument for a Piper-backed fallback provider rather than a CPU XTTS one.

---

## Stage 2 --- Speech Recognition (brief Phase 2) *(complete)*

**Goal:** speak into the microphone, see the transcript in the terminal.

**Outcome:** met on both devices. Warm RTF 0.10 on CUDA and 0.57 on CPU, with a
word error rate of 0.0 against the reference utterance. Full figures in
`docs/benchmarks.md`. Three findings:

1. **Whisper is viable on CPU**, unlike XTTS. That reopens the Stage 4 VRAM
   question in a better direction --- see below.
2. **CTranslate2's CUDA libraries needed help being found on Windows.** A CUDA
   build of torch already ships cuBLAS 12 and cuDNN 9, but Windows only
   searches registered directories. The provider registers `torch/lib` itself,
   so CUDA transcription works with no separate cuDNN install.
3. **The reference machine's default microphone is a virtual device** that
   returns digital silence. `audio.inputDevice` should be set to 2.

Whisper's own VAD filter is enabled by default, because it otherwise invents
text during silence and push-to-talk regularly captures some.

Deliverables:
- `SpeechToTextProvider` ABC + `fasterWhisperProvider`, configurable model,
  device, compute type, language.
- `app/audio/input.py` --- `AudioInput` abstraction; `sounddevice` capture at
  16 kHz mono, plus a WAV-file source for tests.
- Push-to-talk / press-enter capture for now. No VAD yet.
- CLI flag to list and select input devices.

**Exit criteria:** transcription of a fixed test WAV matches expected text in an
integration test; live microphone capture works on this machine; CPU fallback
verified by forcing `device=cpu`; word-error-rate spot check on a few utterances
recorded in `docs/benchmarks.md`.

**Risk:** CUDA-accelerated CTranslate2 on Windows fails obscurely if cuDNN 9
isn't discoverable. Detect and produce a clear error naming the missing DLL,
rather than letting it crash --- brief §25 requires useful failures.

---

## Stage 3 --- Conversation Loop (brief Phase 3) *(complete)*

**Goal:** speak, get spoken reply, hands on keyboard but no cloud, no LLM.

**Outcome:** met. Time to first audio is **1.56 s mean** with both models on
CUDA, and the state sequence is exactly `listening -> thinking -> speaking ->
idle`. Full figures in `docs/benchmarks.md`. Three findings:

1. **Synthesis is 84% of the wait** (1.32 s of 1.56 s). Streaming synthesis is
   therefore the highest-value item in Stage 4, not an optimisation to do
   later.
2. **Warm-up saves ~1.1 s on the first turn** and is enabled by default.
3. **Whisper belongs on the GPU --- this reverses the Stage 2 conclusion.**
   See the revised VRAM position under Stage 4.

Deliverables:
- `app/assistant/runtime.py` --- orchestrates capture → STT → handler → TTS,
  emitting the §8 events at each transition.
- `app/assistant/session.py`, `conversation.py` --- session state and turn
  history, in memory only.
- A deterministic handler: echo, plus a handful of hard-coded responses.
- End-to-end latency instrumentation per stage, surfaced on every turn.

**Exit criteria:** a full spoken round-trip works; the event log shows the
correct state sequence (Idle → Listening → Thinking → Speaking → Idle);
total round-trip latency measured and recorded.

This is the stage where we learn whether the latency story works at all. If
round-trip is unpleasant here, we address it before adding an LLM, because
everything after this only adds to it.

---

## Stage 4 --- Local LLM (brief Phase 4) *(complete)*

**Goal:** conversational replies from a local model.

**Outcome:** met, and verified against qwen2.5:3b through Ollama. Streaming
reaches first audio in **1.70 s against 5.93 s** with a simulated model, a 71%
improvement; against the real model it is 1.72 s against 2.45 s, a 30%
improvement. The gap between those two figures is instructive and recorded in
`docs/benchmarks.md`: qwen2.5 is fast and, told to be brief, answers in one
sentence, which leaves streaming nothing to pipeline.

Warm generation runs at 47--68 tokens/second with the first fragment at 0.17 s,
so the model is not the bottleneck on this hardware --- synthesis still is.

Deliverables:
- `LlmProvider` ABC with both `generate()` and `generateStream()`.
- `ollamaProvider` (HTTP, no vendor SDK dependency) and
  `openAiCompatibleProvider`.
- System prompt and conversation-history management with a token/turn budget.
- Streaming: first sentence handed to TTS while the LLM is still generating.
  This is the single biggest perceived-latency win available to us, and the
  brief names it as a long-term objective --- doing it here is cheaper than
  retrofitting it after the API layer exists.

**Exit criteria:** spoken question produces spoken conversational answer;
provider swap between Ollama and an OpenAI-compatible endpoint is a config
change; time-to-first-audio measurably better with streaming than without.

**VRAM position, settled by Stage 3 measurement.** Stage 2 suggested moving
Whisper to CPU to free VRAM. Measured inside a real turn that costs 2.2 s per
turn to save 0.47 GiB, so it is rejected. Both speech models stay on GPU,
reserving about 3.2 GiB of 8 GiB and leaving roughly 4.5 GiB.

*Stage 10 note: all three models on the GPU at once peaks at 4984 MiB of 8192,
with Ollama reporting 100% GPU placement throughout. The contention this
section was written to manage does not occur at these model sizes, and option 3
below --- moving the language model to CPU --- is withdrawn.*

In preference order:

1. **A 3--4B model at Q4, fully on GPU.** Fits comfortably in 4.5 GiB.
2. **A 7--8B model at Q4 on GPU with a reduced context window.** Very tight;
   needs measuring before committing.
3. **The language model on CPU via Ollama**, if a larger model matters more
   than token rate. Streaming hides first-token latency in a way it cannot
   hide transcription latency --- which is exactly why the language model, not
   Whisper, is the thing that should move if something must.
4. Ollama `keep_alive` tuning so the model unloads between turns --- rejected
   unless nothing else works, as reload cost lands directly in the response.

**Streaming is now the priority, not a stretch goal.** Synthesis is 84% of
Stage 3's round trip. Handing the first sentence to XTTS while the language
model is still generating is what keeps the turn from getting worse once a
model is in the loop.

---

## Stage 5 --- Tools and Home Automation (brief Phase 5) *(complete against simulated devices)*

**Goal:** "turn the kitchen light off" actually turns the light off.

**Outcome:** the tool system, allow-list, device matching and the three-layer
router are built and tested. Deterministic commands run in **0.000 s against
0.49 s** through the model, roughly 2000x faster, with no model output in the
path of anything that changes state.

Verified end to end against the simulated house and a live qwen2.5:3b.
**Verification against real Home Assistant hardware is outstanding**, pending an
address and token. MQTT is not built --- see the open questions.

Three findings, all in `docs/benchmarks.md`:

1. **Small-model tool selection is not deterministic.** The same phrase
   variously produced a tool call, a clarifying question, and nothing at all.
2. **A small model will occasionally claim to have acted without acting.** The
   tool-turn system prompt now forbids this explicitly, which reduces rather
   than eliminates it --- an argument for keeping common commands in Layer 1.
3. **A spoken domain word does not reliably imply an entity domain.** A smart
   plug driving a fan is a `switch`, so domain words bias the match rather than
   filtering it.

**Deviation from the deliverables below:** the action allow-list is enforced in
the automation tool rather than the tool registry. Putting "light.turnOff is
permitted" in a generic registry would require the registry to understand home
automation vocabulary, which is the coupling §10 exists to prevent. The registry
enforces tool-level policy; the automation layer enforces action-level policy.
Both run before execution, which is what §17 requires.

Deliverables:
- `app/tools/base.py` --- `Tool` with name, description, Pydantic input schema,
  `execute()`. Explicit `readOnly` vs `mutatesPhysicalState` classification
  (brief §17).
- `app/assistant/toolRegistry.py` --- registration, lookup, schema export for
  LLM function-calling.
- `AutomationProvider` ABC + `homeAssistant` (REST/WebSocket API) and `mqtt`.
- `app/tools/homeAutomation.py` --- translates the generic
  `{action, target}` form into provider calls.
- **Action allow-list**, enforced in the registry, not in the tool. Every
  invocation validated against its schema before execution. LLM output treated
  as untrusted throughout.
- Device/entity discovery and a name-matching layer ("kitchen light" →
  `light.kitchen_ceiling`).
- `app/assistant/intentRouter.py` --- Layer 1 deterministic patterns first,
  falling back to LLM tool-calling (Layer 2) only on no match.

**Exit criteria:** deterministic commands execute with no LLM round trip and are
demonstrably faster; unstructured phrasing routes correctly via the LLM; a fake
`AutomationProvider` covers the tool tests so CI needs no Home Assistant; an
LLM-proposed action outside the allow-list is refused and logged.

**Needs from you:** Home Assistant URL, a long-lived access token (env var,
never committed), and whether MQTT is in scope for this stage or deferred.

---

## Stage 6 --- Wake Word and VAD (brief Phase 6) *(complete; accuracy unverified on real speech)*

**Goal:** hands-free.

**Outcome:** `python -m app.main --mode wake` sits listening for "hey jarvis",
captures until the speaker stops, answers, and returns to waiting. Idle cost is
**3.4% of one core**, and neither model touches the GPU.

Wake-word accuracy on the synthesised test file is 3 of 3 with no false
accepts, but that is a pipeline check rather than an accuracy measurement:
openWakeWord was trained on human speech. **Real accuracy needs you to speak to
it.**

Two bugs found by measuring, both recorded in `docs/benchmarks.md`:

1. **The refractory period used the wall clock**, so replaying a recording
   suppressed every detection after the first. It looked exactly like a model
   failing on synthetic speech. It counts audio time now.
2. **`audio.inputDevice` set to `"2"` selected the wrong device.** PortAudio
   reads an integer as an index and a string as a name, and both environment
   variables and JSON deliver `"2"` as a string. Digit-only values are now
   converted.

`AudioInput` gained a `frames()` primitive and `record()` is now built on it,
because voice activity and wake-word models both need small fixed frames and
neither can work with a single blocking recording.

Deliverables:
- `VadProvider` ABC + Silero VAD; utterance segmentation with configurable
  silence threshold and max duration.
- `WakeWordProvider` ABC + one local engine (openWakeWord is the leading
  candidate --- permissive licence, no account required, custom words trainable;
  Porcupine is the fallback but requires an access key).
- Barge-in: speaking is interrupted when the wake word is detected again.
- Continuous-listen state machine replacing push-to-talk.

**Exit criteria:** assistant sits idle at low CPU, wakes reliably, ends capture
on natural silence rather than a timer, and false-accept/false-reject rates over
a short bench test are recorded.

---

## Stage 7 --- FastAPI Service (brief Phase 7) *(complete)*

**Goal:** the assistant becomes a service; the CLI becomes just one client.

**Outcome:** met, and verified against a live server. Every §9 endpoint works,
two WebSocket clients observe the same turn, unauthenticated requests are
refused, and models stay resident across requests.

Three findings, in `docs/benchmarks.md`:

1. **The language model is the slowest part of a conversational turn** ---
   7.4 s of an 8.5 s request, against 0.3 s measured alone in Stage 5.
   *This was attributed to VRAM contention, and that was wrong: it was a cold
   model load. Corrected in Stage 10 below.*
2. **A busy port cost eighty seconds to discover**, because uvicorn binds after
   startup. The address is probed before anything loads now.
3. **The response event fired after the audio finished.** A client would have
   shown the reply only once it had stopped playing.

**Deviation:** there is one assistant and one session, not one per client.
That is the model §26 describes --- several clients, one local service --- and
it matches the hardware, since there is one GPU and one speaker. Turns are
serialised behind a lock rather than run concurrently, because two replies
spoken over each other would be worse than one waiting. `SessionRegistry`
exists for when Stage 8 gives remote devices their own context.

Deliverables:
- REST endpoints per brief §9; WebSocket `/ws/assistant` carrying the §8 events.
- **Binds to 127.0.0.1 by default.** Non-loopback binding requires an API token
  to be configured, and refuses to start without one.
- Auth middleware (bearer token), request validation, structured logging that
  excludes audio and secrets.
- The existing CLI refactored to talk to the runtime through the same interface
  the API uses, proving the seam is real.

**Exit criteria:** two clients can connect concurrently and both observe state
events; `curl` round-trips text; unauthenticated remote requests are rejected;
models stay loaded across requests.

---

## Stage 8 --- Remote Audio Clients (brief Phase 8) *(complete; one machine, not two)*

**Goal:** microphone and speaker no longer need to live on the inference host.

**Outcome:** `--mode client` is a microphone and a speaker for an assistant
running elsewhere. It loads no speech models and needs no GPU. **The network
adds 15 ms** to a 2.14 s round trip; everything else is the assistant thinking.

**Codec decided by measurement: raw PCM, not Opus.** 81 KB uploaded in four
milliseconds. Because the client runs its own wake word and voice activity
detection it only transmits during an utterance, so an encoder, a decoder and
the latency of both would buy nothing. The format is announced in the control
message rather than assumed, so Opus can be added when this crosses the
internet.

**Per-client sessions were built**, which required moving assistant *state*
out of `Session` and onto the runtime. Conversations are per-client;
state is not, because there is one speaker and one GPU.

**Caveat:** client and server ran on the same machine over the LAN interface,
not on two devices. That exercises the real network stack, serialisation and
framing, but not a physical hop. A genuine second device will add more than
15 ms.

Deliverables:
- Bidirectional audio streaming over the WebSocket, with a defined framing and
  codec choice (Opus if bandwidth matters, raw PCM if it doesn't --- decide with
  measurements).
- A thin Python client acting purely as microphone + speaker.
- Per-client session isolation in the runtime.
- VAD/wake-word running client-side so we don't stream silence continuously.

**Exit criteria:** a second machine on the LAN drives the assistant end to end;
added latency over local operation is measured and acceptable.

---

## Stage 9 --- Clients and Native Components (brief Phases 9, 20, 21) *(web client complete)*

Deliberately last, and deliberately vague --- scope here depends on what the
earlier stages teach us.

**What they taught us, and what was built.**

**No C++.** The brief permits native code "only where profiling or deployment
requirements justify it", and nothing in eight stages of measurement justifies
it. Idle listening costs 3.4% of one core; wake-word inference is 1.81 ms per
80 ms frame; voice activity detection 0.52 ms per 32 ms frame; the network
15 ms. The two real bottlenecks are the language model (7.4 s of an 8.5 s turn
under VRAM contention) and synthesis (84% of a rules-based turn), and C++ would
not move either. Reconsider if this ever targets hardware smaller than a
Raspberry Pi.

**A web client**, served by the API built in Stage 7. It needs no toolchain, no
install and no build step, works on a phone and a desktop from the same URL,
and reaches the mobile case of §22 at a fraction of the cost of a native app.
The token-in-query-string support already existed because browsers cannot set
WebSocket handshake headers --- that was built for exactly this.

Verified in a real browser: push-to-talk through `getUserMedia` and an
`AudioWorklet`, PCM conversion, binary frames, transcription, reply audio, and
device controls. Three defects showed up in the screenshot that no test caught
--- see `docs/benchmarks.md`.

**Still outstanding**, in rough order of usefulness:

- **Reconnection and backoff** in `--mode client`, and service files for
  Windows and Linux, if this is to run daily.
- A **C# tray client** (§21), for a nicer Windows desktop experience than a
  browser tab.
- **MQTT**, still unbuilt from Stage 5.
- **Home Assistant** against real hardware, still verified only against a mock.

---

## Stage 10 --- The Language Model Bottleneck *(complete)*

Not in the original plan. Added because Stage 7 measured a conversational turn
at 7.4 s and the explanation given for it turned out to be wrong.

**Diagnosis.** Three candidates were measured separately rather than reasoned
about. Prompt size cost +0.01 s. VRAM contention cost +0.02 s, with Ollama
reporting `100% GPU` even when the speech models were loaded first and it had
to fit around them. The cause was a **cold model load**: 3.31 s against 0.20 s
warm.

**Fix.** The runtime had warmed XTTS and Whisper since Stage 3 but not the
language model, because the model is not the runtime's to load --- Ollama loads
on first request, not when its service starts. `ResponseHandler.warmUp` now
makes one, alongside the existing warm-up. **The first conversational request
fell from 7.37 s to 0.89 s**, at a startup cost of 4.8 s that runs concurrently
with a 25 s model load and so costs nothing.

**A worse problem found on the way.** Ollama unloads a model five minutes after
its last use, so an assistant spoken to every ten minutes paid the cold load on
*every* turn --- invisible to any benchmark that runs its requests back to
back. `llm.keepAlive` now defaults to `30m`; `-1` keeps it resident
indefinitely on a dedicated machine.

**Synthesis is the bottleneck again**, at about 1.3 s of a 2.25 s turn, exactly
as in Stage 3. The language model is no longer close, and the Stage 4 note about
moving it to CPU is withdrawn --- there was never any contention to relieve.
*Stage 11 addressed the synthesis cost.*

---

## Stage 11 --- The Synthesis Bottleneck *(complete)*

**Diagnosis.** Synthesis costs about 19 ms per character with no meaningful
fixed cost per call, so there is no per-call overhead to amortise and cutting
an utterance into pieces is viable. The sentence-level streaming from Stage 4
could not help, because it has nothing to pipeline when the reply is one
sentence --- and most of what a home assistant says is one sentence.

**Fix.** XTTS decodes autoregressively, so the beginning of an utterance exists
long before the end, and `inference_stream` exposes it. `TextToSpeechProvider`
gained `synthesiseStream`, defaulting to yielding the whole utterance so any
provider still works. `textToSpeech.streamChunkSize` defaults to 10: first
sound in about 0.2 s while generation stays comfortably ahead of playback.
Below 5 it approaches real time and risks stuttering.

Measured on the running assistant, same server, changing only the chunk size:
**0.653 s to 0.257 s** on a short reply and **2.674 s to 0.139 s** on a longer
one. Total synthesis rises slightly, which is the trade: more total work for
far less waiting.

**Remote clients benefit without new plumbing**, because `AudioOutput` already
had the right shape --- the runtime calls `play` once per piece, so an output
that forwards to a WebSocket streams to it. A remote client hears audio at
0.95 s rather than waiting 3.47 s for a complete reply.

**A bug the sweep could not catch.** `inference` returns numpy but
`inference_stream` yields tensors still on the GPU. The chunk-size sweep timed
the generator without converting its output, so it ran clean and the first real
turn failed. A benchmark that exercises only the expensive part of a path will
not find the defects in the cheap part.

**Nothing dominates any more.** Transcription, the language model and synthesis
now cost roughly the same, at about 0.5 s to first audio locally against Stage
3's 1.56 s. Further latency work should wait for something to change.

---

## Cross-Cutting Work

Carried through every stage rather than scheduled as its own:

- **Tests.** Unit tests for anything not requiring a model (config, registry,
  intent routing, event bus, name matching). Fake providers for each ABC.
  Integration tests marked so they can be skipped without models present.
- **Benchmarks.** `docs/benchmarks.md` updated at each stage boundary. Latency
  is a feature here; we should never be guessing at it.
- **Security.** Applied at the stage that introduces the surface --- allow-list
  in Stage 5, auth in Stage 7 --- not as a hardening pass at the end.
- **Docs.** README updated per stage with how to run what now exists.

## Open Questions

1. **XTTS licence (CPML, non-commercial)** --- personal project, or does this
   need a commercially usable voice model? Affects Stage 1's model choice.
2. **Whose voice** is being cloned, and do we have clean 6--30 s samples?
3. **Home Assistant** --- already running? Which entities are in scope first?
4. **MQTT** --- in scope for Stage 5, or deferred?
5. **Wake word** --- any preference on the wake phrase, and is a Porcupine
   access key acceptable if openWakeWord underperforms?
6. **Python 3.13** --- happy to stay on it (`coqui-tts` supports it), or would
   you rather drop to 3.12 for the wider ecosystem compatibility?
