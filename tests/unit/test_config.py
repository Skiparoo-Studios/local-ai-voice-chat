"""Configuration loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import ConfigurationError, Settings, loadSettings
from app.config.models import ApiSection


def writeConfig(directory: Path, data: dict) -> Path:
    path = directory / "settings.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestDefaults:
    def testLoadsDefaultsWhenNoFileGiven(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        settings = loadSettings()

        assert settings.assistant.name == "Assistant"
        assert settings.speechToText.provider == "faster-whisper"
        assert settings.textToSpeech.provider == "xtts"
        assert settings.api.host == "127.0.0.1"

    def testExampleConfigIsValid(self):
        """The shipped example must parse, or it misleads whoever copies it."""
        example = Path(__file__).parents[2] / "config" / "settings.example.json"
        settings = loadSettings(example)

        assert settings.textToSpeech.model.endswith("xtts_v2")
        assert settings.llm.baseUrl == "http://localhost:11434"


class TestFileLoading:
    def testFileValuesOverrideDefaults(self, tmp_path: Path):
        path = writeConfig(tmp_path, {"assistant": {"name": "Ada"}, "llm": {"model": "llama3"}})
        settings = loadSettings(path)

        assert settings.assistant.name == "Ada"
        assert settings.llm.model == "llama3"
        # Untouched sections keep their defaults.
        assert settings.speechToText.model == "small"

    def testByteOrderMarkIsTolerated(self, tmp_path: Path):
        """Notepad and PowerShell both write UTF-8 with a BOM on Windows."""
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"assistant": {"name": "Ada"}}), encoding="utf-8-sig")

        assert loadSettings(path).assistant.name == "Ada"

    def testMissingExplicitFileIsAnError(self, tmp_path: Path):
        with pytest.raises(ConfigurationError, match="not found"):
            loadSettings(tmp_path / "absent.json")

    def testMalformedJsonReportsTheFile(self, tmp_path: Path):
        path = tmp_path / "settings.json"
        path.write_text("{ not json", encoding="utf-8")

        with pytest.raises(ConfigurationError, match="not valid JSON"):
            loadSettings(path)

    def testNonObjectJsonIsRejected(self, tmp_path: Path):
        path = tmp_path / "settings.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")

        with pytest.raises(ConfigurationError, match="JSON object"):
            loadSettings(path)

    def testUnknownKeyIsRejected(self, tmp_path: Path):
        """A typo must fail loudly rather than silently use the default."""
        path = writeConfig(tmp_path, {"speechToText": {"modle": "small"}})

        with pytest.raises(ConfigurationError, match="modle"):
            loadSettings(path)

    def testInvalidDeviceIsRejected(self, tmp_path: Path):
        path = writeConfig(tmp_path, {"speechToText": {"device": "tpu"}})

        with pytest.raises(ConfigurationError):
            loadSettings(path)


class TestEnvironmentOverlay:
    def testEnvironmentOverridesFile(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        path = writeConfig(tmp_path, {"llm": {"provider": "ollama", "model": "llama3"}})
        monkeypatch.setenv("VOICE_LLM__MODEL", "mistral")

        settings = loadSettings(path)

        assert settings.llm.model == "mistral"
        assert settings.llm.provider == "ollama"

    def testSecretsComeFromEnvironment(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("VOICE_AUTOMATION__ACCESSTOKEN", "secret-token")

        settings = loadSettings()

        assert settings.automation.accessToken is not None
        assert settings.automation.accessToken.get_secret_value() == "secret-token"

    def testSecretsAreNotExposedByRepr(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VOICE_AUTOMATION__ACCESSTOKEN", "secret-token")
        settings = Settings()

        assert "secret-token" not in repr(settings)


class TestAudioDeviceSelection:
    """PortAudio reads a string as a device name, an integer as an index."""

    def testNumericStringFromTheEnvironmentBecomesAnIndex(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("VOICE_AUDIO__INPUTDEVICE", "2")

        assert Settings().audio.inputDevice == 2

    def testNumericStringInTheFileBecomesAnIndex(self, tmp_path: Path):
        path = writeConfig(tmp_path, {"audio": {"outputDevice": "5"}})

        assert loadSettings(path).audio.outputDevice == 5

    def testAnActualIntegerIsUnchanged(self, tmp_path: Path):
        path = writeConfig(tmp_path, {"audio": {"inputDevice": 3}})

        assert loadSettings(path).audio.inputDevice == 3

    def testADeviceNameStaysAString(self, tmp_path: Path):
        path = writeConfig(tmp_path, {"audio": {"inputDevice": "Realtek"}})

        assert loadSettings(path).audio.inputDevice == "Realtek"

    def testUnsetRemainsNone(self):
        assert Settings().audio.inputDevice is None


class TestApiSecurity:
    """§17: never expose the API publicly by default."""

    def testLoopbackNeedsNoToken(self):
        assert ApiSection(host="127.0.0.1").authToken is None
        assert ApiSection(host="localhost").authToken is None

    def testNonLoopbackWithoutTokenIsRejected(self):
        with pytest.raises(ValueError, match="authToken"):
            ApiSection(host="0.0.0.0")

    def testNonLoopbackWithTokenIsAllowed(self):
        section = ApiSection(host="192.168.1.10", authToken="a-token")

        assert section.authToken.get_secret_value() == "a-token"

    def testUnparseableHostIsRejected(self):
        with pytest.raises(ValueError, match="not a recognised address"):
            ApiSection(host="assistant.local")


class TestProviderSummary:
    def testDescribesResolvedProviders(self):
        settings = Settings()
        summary = settings.describeProviders()

        assert set(summary) == {"speechToText", "textToSpeech", "llm", "automation"}
        assert "faster-whisper" in summary["speechToText"]

    def testSummaryExcludesSecrets(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VOICE_LLM__APIKEY", "secret-key")
        settings = Settings()

        assert "secret-key" not in json.dumps(settings.describeProviders())
