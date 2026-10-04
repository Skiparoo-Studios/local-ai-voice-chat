# A Raspberry Pi as a voice client

A Pi in the kitchen, one in the bedroom, one in the shed, all talking to the
single machine with the GPU. This is §26 of the brief — several clients, one
service — and `--mode client` already implements it. Nothing new has to be
written; this is what to install and configure.

## What runs where

The Pi is a microphone and a speaker with just enough intelligence to know when
to open the connection. Everything expensive stays on the server.

| | Raspberry Pi | Server |
|---|---|---|
| Capture and playback | yes | — |
| Wake word | yes | — |
| Voice activity detection | yes | — |
| Speech recognition (Whisper) | — | yes |
| Language model | — | yes |
| Speech synthesis (XTTS) | — | yes |
| Home automation | — | yes |

The Pi does the wake word and voice activity detection locally on purpose:
**it only transmits while someone is actually speaking**, so an idle Pi costs
nothing on the network and the server never sees the room's background noise.

Audio crosses as raw 16 kHz mono PCM over a WebSocket. Measured, an utterance
of 2.59 s is 81 KB and uploads in 4 ms; the network adds **15 ms** to a
round trip that is otherwise all model time.

## Hardware

| Part | Recommendation |
|---|---|
| Board | Pi 4 (2 GB) or Pi 5. A Pi Zero 2 W works with `always-awake` but is tight for openWakeWord. Anything 32-bit — a Pi 2 v1.1, a Pi 1, a Zero W — has no wake word available at all; see the Pi 2 section below. |
| Microphone | Any USB microphone. A ReSpeaker 2-Mic HAT or a USB conference mic is much better than a cheap capsule. |
| Speaker | USB or 3.5 mm. On a Pi 5 there is no headphone jack, so USB or HDMI audio. |
| Network | Ethernet if you can. Wi-Fi is fine — the traffic is tiny — but adds jitter. |
| Storage | 8 GB is plenty. Nothing large is installed, and no models are downloaded unless you use a wake word. |

**Headphones or a directional microphone are worth it.** The assistant has an
echo guard (`audio.echoGuardSeconds`) that stops it hearing its own reply, but
that is suppression, not acoustic echo cancellation. A speaker pointed straight
at the microphone in a bare room can still get through.

## Before you start: the Python version

This is the one thing likely to stop you.

```toml
requires-python = ">=3.12"
```

**Raspberry Pi OS Bookworm ships Python 3.11**, which will not install this
project. Your options, best first:

1. **Raspberry Pi OS Trixie** (Debian 13) ships Python 3.13. Simplest.
2. **`pyenv`** to build 3.12 alongside the system Python. Takes about 20
   minutes to compile on a Pi 4.
3. **A Debian container** with a newer Python, if you already run Docker.

Check what you have:

```bash
python3 --version
```

## Install

```bash
sudo apt update
sudo apt install -y python3-venv libportaudio2 git

git clone https://github.com/Skiparoo-Studios/local-ai-voice-chat.git voicebox
cd voicebox

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[client]"
```

`libportaudio2` is the system library `sounddevice` binds to. Without it the
import fails with an `OSError` that does not mention PortAudio at all.

### What the `client` extra deliberately leaves out

It is **not** the `wake` extra. `silero-vad` depends on `torch` and
`torchaudio` unconditionally — not as an optional extra — and PyTorch on a Pi
is hundreds of megabytes and a slow install, to do a job the energy detector
does for nothing. Use `vad.provider: "energy"` on a Pi.

Neither Whisper nor XTTS is installed either. The server does all of that.

If you do want a wake word, `openwakeword` is in the extra, but note it pulls
`scipy`, `scikit-learn` and — on Linux — `tflite-runtime`, whose ARM wheels
lag new Python releases. If `pip install` fails there, use `always-awake` and
a headset, or accept the extra work of finding a wheel.

## Configure the Pi

Create `config/settings.json` on the **Pi** (it is git-ignored, so it stays
local):

```json
{
    "client": {
        "serverUrl": "http://192.168.1.10:8000"
    },
    "wakeWord": {
        "provider": "openwakeword",
        "word": "hey_jarvis",
        "threshold": 0.5
    },
    "vad": {
        "provider": "energy",
        "energyThreshold": 0.02,
        "silenceFrames": 25
    },
    "audio": {
        "inputDevice": null,
        "outputDevice": null,
        "sampleRate": 16000,
        "echoGuardSeconds": 0.4
    }
}
```

The token never goes in this file:

```bash
export VOICE_CLIENT__TOKEN="the-same-token-the-server-has"
```

### On the server

The server has to accept connections from outside its own machine, which the
configuration refuses to do without a token — that invariant is enforced in
`ApiSection`, not left to the API layer:

```json
{ "api": { "host": "0.0.0.0", "port": 8000 } }
```

```powershell
$env:VOICE_API__AUTHTOKEN = "a-long-random-string"
python -m app.main --mode serve
```

Set `host` to `0.0.0.0` and forget the token and the assistant **will not
start** — by design, since it can act on the physical environment.

## Find the audio devices

```bash
python -m app.main --list-input-devices
python -m app.main --list-devices
```

Then set `audio.inputDevice` to the index or a substring of the name. A number
selects by index; a string selects by name. Beware that a digit in quotes
(`"2"`) is read as an index, not a name — there is a validator for exactly that
mistake.

Test capture before involving the server:

```bash
python scripts/benchmarkStt.py --check-microphone
```

## Run it

```bash
source .venv/bin/activate
python -m app.main --mode client
```

Send one message without a microphone, to prove the link works:

```bash
python -m app.main --mode client --send "turn off the hallway light"
```

### As a service

`--mode client` has **no reconnection or backoff** — if the server restarts or
the Wi-Fi drops, the client exits. For a Pi left running unattended that has to
be handled outside the process, and systemd does it properly anyway.

`/etc/systemd/system/voicebox-client.service`:

```ini
[Unit]
Description=Voicebox client
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/voicebox
Environment="VOICE_CLIENT__TOKEN=the-same-token-the-server-has"
ExecStart=/home/pi/voicebox/.venv/bin/python -m app.main --mode client
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now voicebox-client
journalctl -u voicebox-client -f
```

`Restart=always` with `RestartSec=5` is what covers the missing backoff. The
user must be in the `audio` group — `pi` already is.

## How many Pis can run at once

Two different numbers, and conflating them is the mistake.

### Connected at once: no fixed limit

Nothing in the code caps connections. Each client costs one `Session` (its own
conversation) and a 64-event queue. Ten clients connected and all issuing turns
at the same instant behaves correctly:

```text
10 clients, 10 simultaneous turns
peak concurrent turns : 1
sessions held         : 10
conversations separate: True
```

Memory is flat in the number of clients — Whisper and XTTS are loaded once at
startup and shared, so a tenth Pi costs no VRAM.

### Talking at once: exactly one

`AssistantService` holds an `asyncio.Lock` around every turn. This is
deliberate, and the reason is in the code:

> Turns are therefore serialised behind a lock rather than run concurrently:
> two replies spoken over each other would be worse than one waiting.

There is one GPU. A second Pi that speaks mid-turn is not refused, it **waits**
— its audio is already captured and buffered, so nothing is lost, but its reply
is delayed by however much of the first turn remains.

### So what does that mean in practice

A turn measured end to end is **2.14 s mean** (qwen2.5:3b and XTTS on CUDA), of
which 2.12 s is the assistant thinking. That is a hard ceiling of roughly
**28 turns per minute** across all Pis put together.

The useful question is how often anyone actually speaks. Keeping the server
under about 30% busy so waits stay rare:

| Each Pi used once every… | Comfortable number of Pis |
|---|---|
| 30 seconds (active conversation) | ~4 |
| 1 minute | ~8 |
| 5 minutes (normal household use) | ~40 |

**For a house, the answer is that the client count is not your constraint.**
Five or six Pis in different rooms will essentially never collide, because each
one is idle almost all the time and a turn is two seconds. If several people
genuinely hold continuous conversations at once, the fix is a faster turn — a
smaller model, or synthesis on a second GPU — not more clients.

Bandwidth is not a constraint either: 32 KB/s per Pi *while someone is
speaking*, and zero otherwise. Eight Pis all talking simultaneously is about
2 Mbps.

## Things to know before you deploy several

- **Conversations are per-client, but the assistant's state is not.** Each Pi
  gets its own history, so two people do not share context. The
  `AssistantState` (idle, thinking, speaking) is global and every client sees
  it — a client can tell the assistant is busy, but not for whom.

- **`assistant.interrupt` is global.** A client sending it stops whatever the
  assistant is doing, including another client's turn. Wake-word barge-in is
  handled locally on each Pi and does not have this problem.

- **Toddler mode state is shared.** The child's name, which questions have been
  asked and the per-topic difficulty live on the handler, not the session. Two
  Pis in toddler mode are talking to the same game. Fine for one child; not a
  design for two.

- **A client that falls 64 events behind is dropped**, with a warning in the
  server log. That is backpressure protecting the assistant from a stalled
  client, not a limit you are likely to hit — a turn publishes six events, so a
  client has to be about ten turns behind before it is cut off.

- **The reply header carries `serverSeconds`.** Subtract it from your own round
  trip to tell a slow model from a slow link, which is the first question worth
  asking when a remote device feels sluggish.

## Older boards: the Pi 2 Model B v1.1

Short answer: **it will work, but as a headset or push-to-button device rather
than something you leave listening in a room.** The board is fast enough. The
problem is entirely that it is 32-bit.

The v1.1 has a BCM2836 — quad-core Cortex-A7 at 900 MHz, **ARMv7, 32-bit
only**. (The later v1.2 revision quietly swapped in the BCM2837, which is
64-bit capable. Check with `cat /proc/cpuinfo`; the v1.1 reports `ARMv7
Processor rev 5 (v7l)`.) It also has 1 GB of RAM, no onboard Wi-Fi, and
100 Mbit Ethernet hanging off the USB bus.

### It is not short of CPU

The idle client loop does very little: an energy threshold on each 32 ms frame,
and resampling if the microphone is not natively 16 kHz. Measured on a desktop
core:

| Per 32 ms frame | Cost | Share of one core |
|---|---|---|
| Energy VAD | 0.032 ms | 0.10% |
| soxr 48 kHz → 16 kHz | 0.056 ms | 0.17% |
| **Combined** | **0.088 ms** | **0.28%** |

A Cortex-A7 at 900 MHz is a narrow in-order core; call it 20–30× slower than
that per thread. That still lands around **6–9% of one of its four cores**.
This is an extrapolation rather than a measurement on the hardware, but the
margin is wide enough that CPU is not what will stop you. 1 GB of RAM is
likewise ample — no model is loaded on the client.

### What actually blocks it

**1. openWakeWord cannot be installed.** This is the real one. It needs
`onnxruntime`, which publishes **no armv7 wheels at all**, and on Linux it
wants `tflite-runtime`, whose armv7 wheels stop at **cp39** — Python 3.9, three
versions below this project's floor. Building onnxruntime from source on a Pi 2
is a multi-hour exercise you should not attempt.

So there is no wake word on this board. See the options below.

**2. Python 3.12+ on 32-bit.** Bookworm armhf ships 3.11. Raspberry Pi OS
Trixie has a 32-bit build with Python 3.13, which is the path of least
resistance. `pyenv` also works but expect an hour or more to compile on a
900 MHz A7.

**3. `soxr` has no 32-bit wheel for Python 3.13.** piwheels has `numpy 2.0.1`
for `cp313-linux_armv7l`, and `websockets` and `sounddevice` are pure Python,
so those are fine. `soxr` you will have to build:

```bash
sudo apt install -y libsoxr-dev build-essential python3-dev
```

Or avoid it entirely: `soxr` is imported lazily and **only when the capture
rate differs from 16 kHz**. A USB microphone that offers 16 kHz natively never
touches it. Check with `--list-input-devices` and the microphone check.

### Without a wake word, what are the options

| Option | Suits |
|---|---|
| `always-awake` + a **headset** | Good. The energy detector only sees your voice, close and loud. |
| `always-awake` + a room microphone | Poor. Every noise above the threshold becomes an utterance sent to the server. |
| A **physical button** | Best for a fixed device, but needs a little code — see below. |
| Let the server do the wake word | Not supported. Clients wake themselves; that is what keeps the network quiet. |

The `manual` wake-word provider is built for exactly this — its docstring says
"intended for tests and for a push-to-talk button" — and it is selectable as
`wakeWord.provider: "manual"`. **But nothing in `--mode client` calls its
`trigger()` method**, so configured as-is the Pi would sit silent forever.
Wiring a GPIO button to it is genuinely small:

```python
# pi2Button.py --- run this instead of --mode client
import asyncio
from gpiozero import Button
from app.client.remoteClient import RemoteClient
from app.config import loadSettings

async def main() -> None:
    client = RemoteClient.fromSettings(loadSettings(None))
    Button(17).when_pressed = client._wakeWord.trigger   # wakeWord.provider: manual
    await client.start()
    await client.run()

asyncio.run(main())
```

That reaches into a private attribute, which is the sign the client should
expose its wake-word provider properly. Say the word and I will add the
accessor and a `--button <pin>` option rather than leaving it as a snippet.

### A hardware switch on the microphone

This is the best answer for a 32-bit board, and arguably better than a wake
word on any board: a physical cut is the only privacy control that cannot be
subverted by software. Leave the client in `always-awake` and let the switch
decide when it can hear.

**The kind of switch matters, and it is not obvious.**

| Switch | What the client sees |
|---|---|
| Mutes the *signal* — an inline switch on the analogue side, or a mute button on a USB microphone | The device stays on the bus and keeps streaming frames of silence. The energy detector sees nothing and transmits nothing. Perfect. |
| Cuts the *connection* — a switch in the USB cable, or unplugging it | The device leaves the bus and the audio callback stops firing. |

The second case used to be a silent failure: the frame queue simply never
filled again, so the client stayed connected and alive but permanently deaf,
and `Restart=always` never fired because nothing had crashed. Capture now gives
up after `audio.captureStallSeconds` (5 by default) and exits with a message,
so systemd restarts it and it recovers when the switch goes back on.

A quiet room never triggers this — silence still arrives as frames. Only a
device that has actually gone does.

Prefer a signal mute if you have the choice: it keeps the stream open, so
switching back on is instant rather than costing a restart and a reconnect.

### Settings for a Pi 2

```json
{
    "client":   { "serverUrl": "http://192.168.1.10:8000" },
    "wakeWord": { "provider": "always-awake" },
    "vad":      { "provider": "energy", "energyThreshold": 0.05 },
    "audio":    {
        "sampleRate": 16000,
        "echoGuardSeconds": 0.5,
        "captureStallSeconds": 5.0
    }
}
```

A higher `energyThreshold` than the 0.02 default, because with no wake word it
is the only thing standing between a passing lorry and a transcription.

Note that the Pi's 3.5 mm jack is **output only** — no Pi has ever had analogue
microphone input — so the microphone is USB, or a USB sound card with an
analogue mute switch on its input.

### Is it worth it

If you have the board already, yes — as a bedside or desk device with a headset
or a button. If you are buying, a Pi 4 or a Pi Zero 2 W is 64-bit, runs
openWakeWord, and skips every problem above. The Pi 2 v1.1 is a 2015 board
whose only real handicap here is its instruction set.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `OSError` on import of `sounddevice` | `libportaudio2` not installed |
| `pip install` rejects the project | Python older than 3.12 |
| Connects, but nothing is heard | Wrong `audio.inputDevice`; check `--list-input-devices` |
| Server refuses to start on `0.0.0.0` | `VOICE_API__AUTHTOKEN` not set — this is deliberate |
| Client exits when the server restarts | No reconnection; use systemd `Restart=always` |
| "No audio from the microphone for 5.0s" | The device left the bus — unplugged, or a switch that cuts the connection rather than muting the signal |
| It answers itself | Raise `audio.echoGuardSeconds`, or use headphones |
| Cuts you off mid-sentence | Raise `vad.silenceFrames` (each is 32 ms) |
| Triggers on background noise | Raise `vad.energyThreshold`, or use a wake word |
| `tflite-runtime` will not install | 32-bit board; use `always-awake` — see the Pi 2 section |
| `onnxruntime` has no matching wheel | 32-bit board; openWakeWord is not available |
| `soxr` fails to build | `sudo apt install libsoxr-dev build-essential python3-dev`, or use a 16 kHz microphone |

See `docs/benchmarks.md` for the measurements quoted here, and the README for
the wake word and voice activity tuning tables.
