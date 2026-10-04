"""Speech synthesis and recognition."""

from app.speech.device import DeviceUnavailableError, cudaAvailable, resolveDevice
from app.speech.speechToText import (
    STT_PROVIDER_REGISTRY,
    SpeechToTextProvider,
    SttProviderUnavailableError,
    Transcript,
    TranscriptionError,
    UnknownSttProviderError,
    createSpeechToTextProvider,
    registerSttProvider,
    resolveSttProviderClass,
)
from app.speech.textToSpeech import (
    PROVIDER_REGISTRY,
    ProviderUnavailableError,
    SpeechSynthesisError,
    TextToSpeechProvider,
    UnknownProviderError,
    createTextToSpeechProvider,
    registerProvider,
    resolveProviderClass,
)
from app.speech.voices import Voice, VoiceLibrary, VoiceNotFoundError

__all__ = [
    "PROVIDER_REGISTRY",
    "STT_PROVIDER_REGISTRY",
    "DeviceUnavailableError",
    "ProviderUnavailableError",
    "SpeechSynthesisError",
    "SpeechToTextProvider",
    "SttProviderUnavailableError",
    "TextToSpeechProvider",
    "Transcript",
    "TranscriptionError",
    "UnknownProviderError",
    "UnknownSttProviderError",
    "Voice",
    "VoiceLibrary",
    "VoiceNotFoundError",
    "createSpeechToTextProvider",
    "createTextToSpeechProvider",
    "cudaAvailable",
    "registerProvider",
    "registerSttProvider",
    "resolveDevice",
    "resolveProviderClass",
    "resolveSttProviderClass",
]
