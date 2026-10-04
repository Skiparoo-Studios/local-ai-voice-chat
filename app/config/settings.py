"""Settings loading.

Configuration comes from three layers, lowest priority first:

1. Defaults declared on the models in :mod:`app.config.models`.
2. A JSON file, by default ``config/settings.json``. Point ``--config`` at
   another to keep several side by side, rather than editing this default ---
   changing it here makes the file you are editing silently stop being read.
3. Environment variables prefixed ``VOICE_``, using ``__`` to descend into
   sections --- for example ``VOICE_AUTOMATION__ACCESSTOKEN``.

Secrets belong in layer 3. The JSON file is git-ignored, but environment
variables keep credentials out of the working tree entirely.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from app.config.models import (
    ApiSection,
    AssistantSection,
    AudioSection,
    AutomationSection,
    BooksSection,
    ClientSection,
    LlmSection,
    LogLevel,
    PathsSection,
    SentrySection,
    SpeechToTextSection,
    TextToSpeechSection,
    ToddlerSection,
    VadSection,
    WakeWordSection,
)

DEFAULT_CONFIG_PATH = Path("config/settings.json")


class ConfigurationError(Exception):
    """Raised when configuration is missing or invalid."""


class Settings(BaseSettings):
    """Root configuration object."""

    model_config = SettingsConfigDict(
        env_prefix="VOICE_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )

    logLevel: LogLevel = "INFO"
    assistant: AssistantSection = AssistantSection()
    speechToText: SpeechToTextSection = SpeechToTextSection()
    textToSpeech: TextToSpeechSection = TextToSpeechSection()
    books: BooksSection = BooksSection()
    sentry: SentrySection = SentrySection()
    llm: LlmSection = LlmSection()
    toddler: ToddlerSection = ToddlerSection()
    automation: AutomationSection = AutomationSection()
    audio: AudioSection = AudioSection()
    vad: VadSection = VadSection()
    wakeWord: WakeWordSection = WakeWordSection()
    client: ClientSection = ClientSection()
    api: ApiSection = ApiSection()
    paths: PathsSection = PathsSection()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order sources so the environment overrides the JSON file.

        ``loadSettings`` passes file contents as init arguments, which pydantic
        would otherwise rank above environment variables.
        """
        return (env_settings, dotenv_settings, init_settings, file_secret_settings)

    def describeProviders(self) -> dict[str, str]:
        """Summarise the resolved provider selection, without secrets."""
        return {
            "speechToText": f"{self.speechToText.provider} ({self.speechToText.model})",
            "textToSpeech": f"{self.textToSpeech.provider} (voice: {self.textToSpeech.voice})",
            "llm": f"{self.llm.provider} ({self.llm.model or 'unset'})",
            "automation": self.automation.provider,
        }


def readConfigFile(path: Path) -> dict[str, Any]:
    """Read and parse a JSON configuration file."""
    try:
        # utf-8-sig tolerates a byte order mark, which Notepad and PowerShell
        # both write, while reading plain UTF-8 unchanged.
        text = path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise ConfigurationError(f"Could not read configuration file {path}: {error}") from error

    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ConfigurationError(
            f"Configuration file {path} is not valid JSON: {error}"
        ) from error

    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Configuration file {path} must contain a JSON object at the top level"
        )
    return data


def loadSettings(path: Path | None = None) -> Settings:
    """Load settings from ``path``, or from the default location if it exists.

    An explicitly requested path that does not exist is an error. The default
    path is optional, so the assistant runs on defaults out of the box.
    """
    fileData: dict[str, Any] = {}

    if path is not None:
        if not path.is_file():
            raise ConfigurationError(f"Configuration file not found: {path}")
        fileData = readConfigFile(path)
    elif DEFAULT_CONFIG_PATH.is_file():
        fileData = readConfigFile(DEFAULT_CONFIG_PATH)

    try:
        return Settings(**fileData)
    except ValidationError as error:
        raise ConfigurationError(f"Invalid configuration:\n{error}") from error
