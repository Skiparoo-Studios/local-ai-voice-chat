"""The browser client, and the route that serves it."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.app import createApp
from app.api.auth import UNAUTHENTICATED_PATHS
from app.api.routes import WEB_PAGE
from app.api.service import AssistantService
from app.audio.output import MemoryAudioOutput
from app.config import Settings

TOKEN = "a-secret-token"


def buildSettings(**overrides) -> Settings:
    base = {
        "assistant": {"handler": "rules", "warmUpOnStart": False},
        "speechToText": {"provider": "scripted"},
        "textToSpeech": {"provider": "tone"},
        "audio": {"outputBackend": "file"},
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def client() -> TestClient:
    settings = buildSettings()
    service = AssistantService(settings, audioOutput=MemoryAudioOutput())
    with TestClient(createApp(settings, service=service)) as testClient:
        yield testClient


@pytest.fixture
def securedClient() -> TestClient:
    settings = buildSettings(api={"host": "127.0.0.1", "port": 8000, "authToken": TOKEN})
    service = AssistantService(settings, audioOutput=MemoryAudioOutput())
    with TestClient(createApp(settings, service=service)) as testClient:
        yield testClient


@pytest.fixture(scope="module")
def page() -> str:
    return WEB_PAGE.read_text(encoding="utf-8")


class TestServing:
    def testRootServesHtml(self, client: TestClient):
        response = client.get("/")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "<title>Assistant</title>" in response.text

    def testPageIsServedWithoutAToken(self, securedClient: TestClient):
        """It holds no data; the token is typed into it."""
        assert securedClient.get("/").status_code == 200

    def testTheApiItselfStillNeedsAToken(self, securedClient: TestClient):
        assert securedClient.get("/status").status_code == 401

    def testRootIsExemptFromAuthentication(self):
        assert "/" in UNAUTHENTICATED_PATHS

    def testPageIsNotCached(self, client: TestClient):
        """It ships beside the code, so a stale copy is a support problem."""
        assert client.get("/").headers["cache-control"] == "no-cache"

    def testPageShipsWithThePackage(self):
        assert WEB_PAGE.is_file()
        assert WEB_PAGE.parent.name == "web"


class TestPageContents:
    """Checks the page carries what it needs, without a browser to run it."""

    def testItIsSelfContained(self, page: str):
        """No build step and no network fetches: it is a local tool."""
        external = re.findall(r'(?:src|href)="(https?://[^"]+)"', page)

        assert external == []

    def testItCarriesNoCredentials(self, page: str):
        lowered = page.lower()

        assert "authtoken" not in lowered.replace("access token", "")
        assert TOKEN not in page

    def testItSendsTheTokenInTheQueryString(self, page: str):
        """Browsers cannot set headers on a WebSocket handshake."""
        assert "?token=" in page
        assert "encodeURIComponent" in page

    def testItCapturesAtTheRateWhisperWants(self, page: str):
        assert "CAPTURE_RATE = 16000" in page

    def testItClipsRatherThanWraps(self, page: str):
        assert "Math.max(-1, Math.min(1" in page

    def testItHandlesEveryEventTheServerSends(self, page: str):
        for kind in (
            "assistant.connected",
            "assistant.state",
            "assistant.transcription",
            "assistant.response",
            "assistant.error",
            "audio.reply",
            "audio.replyEnd",
        ):
            assert f'"{kind}"' in page

    def testItUsesTheAudioProtocol(self, page: str):
        assert "audio.start" in page
        assert "audio.end" in page

    def testItDegradesWithoutMicrophoneSupport(self, page: str):
        """Typing must still work on a browser that cannot capture."""
        assert "navigator.mediaDevices" in page
        assert "typing still works" in page

    def testItAdaptsToDarkMode(self, page: str):
        assert "prefers-color-scheme: dark" in page

    def testItIsUsableOnAPhone(self, page: str):
        assert 'name="viewport"' in page
        assert "@media (max-width" in page


class TestDeviceControls:
    def testDevicesAreOfferedWhenTheRouterIsOn(self):
        settings = buildSettings(
            assistant={"handler": "router", "warmUpOnStart": False},
            automation={"provider": "simulated"},
        )
        service = AssistantService(settings, audioOutput=MemoryAudioOutput())

        with TestClient(createApp(settings, service=service)) as client:
            assert client.get("/").status_code == 200
            assert client.get("/automation/devices").status_code == 200

    def testThePageHidesDevicesWhenThereAreNone(self, page: str):
        assert "devicePanel.hidden" in page

    def testDeviceNamingComesFromTheServer(self, page: str):
        """Duplicating the rule client-side produced 'Hallway Hallway light'."""
        assert "device.describedName" in page

    def testTheApiSuppliesTheDescribedName(self):
        settings = buildSettings(
            assistant={"handler": "router", "warmUpOnStart": False},
            automation={"provider": "simulated"},
        )
        service = AssistantService(settings, audioOutput=MemoryAudioOutput())

        with TestClient(createApp(settings, service=service)) as client:
            devices = client.get("/automation/devices").json()

        hallway = next(d for d in devices if d["deviceId"] == "light.hallway")
        assert hallway["describedName"] == "Hallway light"

    def testButtonsNameTheActionNotTheState(self, page: str):
        assert '"Turn off" : "Turn on"' in page


class TestMissingPage:
    def testAMissingFileIsReportedClearly(self, client: TestClient, monkeypatch):
        monkeypatch.setattr(
            "app.api.routes.WEB_PAGE", Path("does") / "not" / "exist.html"
        )

        response = client.get("/")

        assert response.status_code == 404
        assert "web client" in response.json()["detail"]
