# Local Voice Assistant --- Project & Architecture Brief

## 1. Project Overview

Build a local-first voice assistant application that can:

-   Listen for spoken commands.
-   Convert speech to text locally.
-   Interpret commands and conversational requests.
-   Control home automation devices.
-   Generate natural-language responses.
-   Convert responses to speech using a locally hosted cloned voice.
-   Expose the assistant through an API so multiple clients can use it.
-   Avoid dependence on commercial voice services such as ElevenLabs
    where practical.
-   Remain modular so speech, language-model, automation, and client
    technologies can be replaced independently.

The initial implementation should be Python-first. C++ or C# components
can be introduced later where they provide a clear performance,
platform-integration, or UI benefit.

## 2. Primary Goals

1.  Run speech recognition and TTS locally.
2.  Support a cloned/custom voice for TTS.
3.  Provide low-latency conversational interaction.
4.  Integrate with home automation systems.
5.  Keep individual AI models behind abstractions so models can be
    replaced without rewriting the application.
6.  Provide HTTP and WebSocket interfaces for external clients.
7.  Support future desktop, mobile, embedded, and web clients.
8.  Allow local LLM integration without requiring cloud services.
9.  Permit optional cloud services without making them architectural
    dependencies.
10. Keep the system testable by separating hardware, models, assistant
    logic, and integrations.

## 3. Proposed Technology Stack

### Core

-   Python 3.12+
-   FastAPI
-   WebSockets
-   Pydantic
-   asyncio

### Text-to-Speech

Initial candidate:

-   Coqui XTTS / XTTS-v2

TTS must be implemented behind a generic interface so XTTS can be
replaced by another model later.

### Speech-to-Text

Initial candidates:

-   faster-whisper
-   Whisper

Prefer `faster-whisper` initially where supported because of its
performance characteristics.

### Voice Activity Detection

Candidate:

-   Silero VAD

The VAD implementation must also sit behind an interface.

### Wake Word

Do not tightly couple the application to a specific wake-word engine.

Provide a `WakeWordProvider` abstraction that can later support local
engines.

### LLM / Intelligence

The assistant should support interchangeable providers:

-   Local LLM
-   Ollama
-   llama.cpp-compatible services
-   OpenAI-compatible APIs
-   Other remote providers

The rest of the application must not depend directly on a particular LLM
SDK.

### Home Automation

Initial integrations:

-   Home Assistant
-   MQTT
-   HTTP/REST devices

Home Assistant should preferably be accessed through its supported API
rather than directly manipulating its internal state.

## 4. High-Level Architecture

``` text
                         ┌─────────────────────┐
                         │      Clients        │
                         │                     │
                         │ Web / Mobile / C#   │
                         │ C++ / Embedded      │
                         └──────────┬──────────┘
                                    │
                            HTTP / WebSocket
                                    │
                         ┌──────────▼──────────┐
                         │     FastAPI API     │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │ Assistant Runtime   │
                         │                     │
                         │ Session             │
                         │ Conversation        │
                         │ Intent Routing      │
                         │ Tool Execution      │
                         └──────┬───────┬──────┘
                                │       │
                    ┌───────────┘       └───────────┐
                    │                               │
             ┌──────▼──────┐                 ┌──────▼──────┐
             │     LLM     │                 │ Automation  │
             │  Providers  │                 │   Layer     │
             └─────────────┘                 ├─────────────┤
                                             │ Home Assist │
                                             │ MQTT        │
                                             │ REST        │
                                             └─────────────┘

Microphone
    │
    ▼
┌────────────┐
│ Audio Input│
└─────┬──────┘
      ▼
┌────────────┐
│    VAD     │
└─────┬──────┘
      ▼
┌────────────┐
│ Wake Word  │
└─────┬──────┘
      ▼
┌────────────┐
│    STT     │
│  Whisper   │
└─────┬──────┘
      ▼
Assistant Runtime
      │
      ▼
┌────────────┐
│    TTS     │
│    XTTS    │
└─────┬──────┘
      ▼
┌────────────┐
│Audio Output│
└────────────┘
```

## 5. Voice Processing Pipeline

The normal interaction pipeline should be:

``` text
Microphone
    ↓
Audio frames
    ↓
Voice Activity Detection
    ↓
Wake-word detection
    ↓
Capture utterance
    ↓
Speech-to-text
    ↓
Intent classification / assistant processing
    ↓
Tool execution if required
    ↓
LLM response generation if required
    ↓
Text response
    ↓
Text-to-speech
    ↓
Audio output
```

Not every request needs an LLM.

For example:

> "Turn off the kitchen lights"

should be capable of being resolved by deterministic intent/tool routing
without requiring an LLM round trip once the intent is understood.

The LLM can still be used to interpret less structured language where
useful.

## 6. Suggested Project Structure

``` text
voice-assistant/
├── app/
│   ├── main.py
│   │
│   ├── api/
│   │   ├── routes.py
│   │   ├── websocket.py
│   │   └── models.py
│   │
│   ├── audio/
│   │   ├── input.py
│   │   ├── output.py
│   │   ├── vad.py
│   │   └── wakeWord.py
│   │
│   ├── speech/
│   │   ├── speechToText.py
│   │   ├── textToSpeech.py
│   │   └── models/
│   │       ├── whisperProvider.py
│   │       └── xttsProvider.py
│   │
│   ├── assistant/
│   │   ├── runtime.py
│   │   ├── session.py
│   │   ├── conversation.py
│   │   ├── intentRouter.py
│   │   └── toolRegistry.py
│   │
│   ├── intelligence/
│   │   ├── llmProvider.py
│   │   └── providers/
│   │       ├── ollamaProvider.py
│   │       └── openAiCompatibleProvider.py
│   │
│   ├── automation/
│   │   ├── automationProvider.py
│   │   ├── homeAssistant.py
│   │   └── mqtt.py
│   │
│   ├── tools/
│   │   ├── base.py
│   │   └── homeAutomation.py
│   │
│   ├── config/
│   │   ├── settings.py
│   │   └── models.py
│   │
│   └── events/
│       ├── eventBus.py
│       └── events.py
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── config/
│   └── settings.example.json
│
├── voices/
│   └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── pyproject.toml
├── README.md
└── .gitignore
```

The exact structure can evolve as implementation requirements become
clearer. Avoid unnecessary abstractions before they provide value.

## 7. Core Interfaces

Model-specific code should not leak into the assistant runtime.

Conceptually, use interfaces similar to:

``` python
from abc import ABC, abstractmethod


class SpeechToTextProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio: bytes) -> str:
        raise NotImplementedError


class TextToSpeechProvider(ABC):
    @abstractmethod
    async def synthesise(self, text: str) -> bytes:
        raise NotImplementedError


class LlmProvider(ABC):
    @abstractmethod
    async def generate(self, messages: list[dict]) -> str:
        raise NotImplementedError


class AutomationProvider(ABC):
    @abstractmethod
    async def execute(self, action: str, parameters: dict) -> dict:
        raise NotImplementedError
```

Provider implementations should be swappable through configuration or
dependency injection.

## 8. Event-Driven Runtime

Audio and AI processing should not unnecessarily block one another.

Use `asyncio` for orchestration and introduce an internal event bus
where useful.

Possible events include:

``` text
AudioDetected
WakeWordDetected
SpeechStarted
SpeechEnded
TranscriptionStarted
TranscriptionCompleted
IntentDetected
ToolExecutionStarted
ToolExecutionCompleted
ResponseGenerated
SpeechGenerationStarted
SpeechGenerationCompleted
AssistantError
```

This will make it easier for UI clients to display assistant state such
as:

``` text
Idle
Listening
Thinking
Controlling device
Speaking
```

## 9. API

FastAPI should expose both REST and WebSocket interfaces.

### REST

Possible endpoints:

``` text
GET  /health
GET  /status

POST /assistant/message
POST /speech/transcribe
POST /speech/synthesise

GET  /automation/devices
POST /automation/execute
```

### WebSocket

``` text
WS /ws/assistant
```

The WebSocket connection can carry events such as:

``` json
{
    "type": "assistant.state",
    "state": "listening"
}
```

``` json
{
    "type": "assistant.transcription",
    "text": "turn the lounge room light off"
}
```

``` json
{
    "type": "assistant.response",
    "text": "I've turned the lounge room light off."
}
```

Eventually audio could also be streamed through WebSockets rather than
requiring the microphone to be physically attached to the assistant
host.

## 10. Home Automation Architecture

Do not put Home Assistant-specific behaviour directly inside assistant
logic.

Use a generic automation layer:

``` text
Assistant
    ↓
Tool Registry
    ↓
Home Automation Tool
    ↓
Automation Provider
    ├── Home Assistant
    ├── MQTT
    └── REST
```

An assistant action should use a generic representation such as:

``` json
{
    "action": "light.turnOff",
    "target": "kitchen"
}
```

The provider translates that action into the protocol required by the
actual automation platform.

## 11. Tool System

The assistant should have a tool registry.

Tools may eventually include:

``` text
Home automation
Timers
Alarms
Weather
Calendar
Music
Reminders
Search
Computer control
Custom scripts
Game/server status
```

Each tool should expose:

-   Name
-   Description
-   Input schema
-   Execute method

This allows deterministic code and LLMs to invoke the same underlying
capabilities.

## 12. Conversation and Intent Processing

Use a layered approach.

### Layer 1 --- Deterministic Commands

Handle obvious commands without an LLM where practical.

Examples:

``` text
Volume up
Stop
Cancel
Turn the bedroom light off
Set a timer for ten minutes
```

### Layer 2 --- Intent Interpretation

Use lightweight parsing or an LLM to turn natural language into
structured actions.

Example:

``` text
"It's pretty dark in the kitchen."
```

could potentially become:

``` json
{
    "tool": "homeAutomation",
    "action": "light.turnOn",
    "target": "kitchen"
}
```

Actions affecting the physical environment should remain constrained to
explicitly supported tools and validated arguments.

### Layer 3 --- Conversation

General questions and conversational interactions can be passed to the
configured LLM.

## 13. TTS Design

The first implementation should support XTTS.

The TTS interface should accept at least:

``` text
text
voice
language
```

Future options may include:

``` text
speed
emotion
style
streaming
```

Voice samples and generated speaker embeddings should be stored
separately from application code.

Do not commit personal voice recordings or generated embeddings to
source control.

The architecture must allow XTTS to be replaced without changing
assistant logic.

## 14. STT Design

Start with `faster-whisper`.

Support configuration of:

``` text
model
device
compute type
language
```

Example configuration:

``` json
{
    "speechToText": {
        "provider": "faster-whisper",
        "model": "small",
        "device": "auto",
        "computeType": "auto",
        "language": "en"
    }
}
```

Do not assume CUDA is available.

The system should detect available acceleration and fall back to CPU
where necessary.

## 15. Configuration

Keep configuration outside implementation code.

Example:

``` json
{
    "assistant": {
        "name": "Assistant"
    },
    "speechToText": {
        "provider": "faster-whisper",
        "model": "small"
    },
    "textToSpeech": {
        "provider": "xtts",
        "voice": "default"
    },
    "llm": {
        "provider": "ollama",
        "model": ""
    },
    "automation": {
        "provider": "home-assistant"
    }
}
```

Secrets must come from environment variables or a secret store rather
than committed configuration.

## 16. Local-First Requirement

The application should be capable of operating locally for its core
functions:

``` text
Microphone
Speech recognition
Intent processing
Home automation
LLM
Text-to-speech
Speaker
```

Internet connectivity should not be a fundamental requirement.

Cloud providers may be supported as optional plugins/providers.

## 17. Security

Home automation gives the assistant access to the physical environment,
so security boundaries should be explicit.

Requirements:

-   Never expose the API publicly by default.
-   Bind to localhost by default.
-   Require authentication before allowing remote clients.
-   Validate every tool invocation.
-   Maintain an allow-list of available automation actions.
-   Do not permit arbitrary shell execution through LLM-generated tool
    calls.
-   Treat LLM output as untrusted input.
-   Keep API tokens outside source control.
-   Avoid logging secrets.
-   Avoid logging raw microphone audio by default.
-   Make persistent conversation storage opt-in.
-   Separate read-only tools from tools that modify physical state.
-   Consider confirmation requirements for sensitive actions.

## 18. Performance Considerations

The architecture should eventually permit models to remain loaded in
memory.

Avoid repeatedly loading:

``` text
Whisper
XTTS
LLM
voice embeddings
```

for individual requests.

Long-running provider instances should own their models.

Where supported, investigate streaming:

``` text
Microphone → STT
LLM → response tokens
TTS → audio
```

The long-term objective is to begin speaking before the complete
response has been generated.

## 19. Hardware Independence

Do not assume a particular microphone, speaker, GPU, or operating
system.

Define audio input/output abstractions so the assistant can eventually
run on:

``` text
Windows PC
macOS
Linux server
Raspberry Pi-class device
Dedicated assistant hardware
```

A lightweight device should also be able to operate purely as:

``` text
Microphone + speaker client
            ↓
        Local network
            ↓
      Assistant server
```

This permits expensive AI inference to happen on a more powerful
machine.

## 20. Future C++ Components

Do not begin by rewriting AI components in C++.

Introduce C++ only where profiling or deployment requirements justify
it.

Potential uses:

-   Low-latency audio capture.
-   Audio DSP.
-   Wake-word processing.
-   Embedded clients.
-   ONNX Runtime inference.
-   GPU-specific inference.
-   Native integrations.
-   Integration with other C++ applications.

Expose native components to Python through a narrow boundary rather than
coupling the entire application to them.

## 21. Future C# Client

A C# application may later provide a Windows-native client.

Potential responsibilities:

``` text
System tray
Settings
Microphone selection
Speaker selection
Push-to-talk
Conversation history
Assistant status
Notifications
Windows integration
```

The C# client should communicate with the Python assistant through
HTTP/WebSockets.

It should not need to host the AI models itself.

## 22. Mobile Clients

Future iOS and Android applications should be treated as clients rather
than rewrites of the assistant.

Possible flow:

``` text
Phone microphone
      ↓
Audio stream
      ↓
Local assistant server
      ↓
STT → Assistant → TTS
      ↓
Audio stream
      ↓
Phone speaker
```

This keeps the model stack centralised while permitting multiple
assistant devices around the home.

## 23. Development Phases

### Phase 1 --- TTS Proof of Concept

Build a CLI application:

``` text
Text input
    ↓
XTTS
    ↓
Audio output
```

Requirements:

-   Load a voice sample.
-   Generate speech.
-   Play generated audio.
-   Keep the model loaded between requests.

### Phase 2 --- Speech Recognition

Add:

``` text
Microphone
    ↓
faster-whisper
    ↓
Terminal transcription
```

### Phase 3 --- Voice Conversation Loop

Combine STT and TTS:

``` text
Speak
 ↓
STT
 ↓
Simple assistant
 ↓
TTS
 ↓
Hear response
```

Initially the assistant can simply echo or use deterministic responses.

### Phase 4 --- Local LLM

Introduce the `LlmProvider` interface and connect a local model.

Target:

``` text
Speak → STT → LLM → TTS
```

### Phase 5 --- Home Automation

Add Home Assistant and/or MQTT.

Target:

``` text
"Turn the kitchen light off"
              ↓
             STT
              ↓
        Intent Router
              ↓
      Home Automation Tool
              ↓
       Home Assistant
```

### Phase 6 --- Wake Word and VAD

Make interaction hands-free.

Target:

``` text
Idle
 ↓
Wake word
 ↓
Listen
 ↓
Process
 ↓
Respond
 ↓
Idle
```

### Phase 7 --- FastAPI

Expose assistant capabilities over the local network.

### Phase 8 --- Remote Audio Clients

Allow another computer or phone to provide microphone audio and receive
generated audio.

### Phase 9 --- Desktop/Mobile UI

Build clients independently of the Python AI runtime.

## 24. Initial Codex Task

Do not attempt to build the entire assistant in the first
implementation.

Start with Phase 1.

Create the initial Python project structure and a working TTS proof of
concept.

The first milestone is:

``` text
python -m app.main
```

The program should:

1.  Load configuration.
2.  Initialise the configured TTS provider.
3.  Load XTTS once.
4.  Accept text repeatedly from the terminal.
5.  Synthesise the text using a configured voice.
6.  Play the resulting audio.
7.  Continue accepting prompts without reloading the model.
8.  Shut down cleanly.

Keep XTTS-specific code behind `TextToSpeechProvider`.

Do not introduce Home Assistant, Whisper, FastAPI, wake-word detection,
an LLM, or unnecessary infrastructure during this milestone.

## 25. Engineering Guidelines

-   Prefer simple modules over premature frameworks.
-   Use Python type hints.
-   Use `async` where operations are naturally asynchronous, but do not
    make everything asynchronous unnecessarily.
-   Use lower camel case for method names.
-   Use British English spelling in identifiers and documentation where
    practical.
-   Keep model/provider-specific dependencies isolated.
-   Keep hardware-specific behaviour behind interfaces.
-   Avoid global mutable state.
-   Use dependency injection where it improves testability.
-   Add unit tests for logic that does not require loading AI models.
-   Keep model downloads, generated audio, voice samples, caches, and
    secrets out of Git.
-   Fail with useful errors when a required model or device is
    unavailable.
-   Log lifecycle events without exposing sensitive data.
-   Prefer configuration over hard-coded provider choices.

## 26. Definition of Success

The architecture is successful if the assistant can eventually support:

``` text
Several microphones / clients
            │
            ▼
   One local assistant service
            │
    ┌───────┼────────┐
    ▼       ▼        ▼
   STT     LLM    Automation
    │                │
    │          ┌─────┴─────┐
    │          ▼           ▼
    │     Home Assistant  MQTT
    │
    ▼
   TTS
    │
    ▼
Cloned local voice
```

while allowing any major provider to be replaced independently.

The guiding principle is:

> **Build the assistant as a local service with replaceable
> capabilities, not as a monolithic voice application.**
