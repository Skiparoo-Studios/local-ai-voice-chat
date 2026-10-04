"""The HTTP and WebSocket interfaces.

Built on the stub providers, so the whole API is covered without loading a
model. The service is constructed here and handed to the app, which is why
importing the API does not cost two gigabytes.
"""

from __future__ import annotations

import base64
import json

import pytest
from fastapi.testclient import TestClient

from app.api.app import createApp
from app.api.auth import ApiAuthenticator
from app.api.service import AssistantService
from app.audio.audioBuffer import AudioBuffer
from app.audio.output import MemoryAudioOutput
from app.config import Settings

TOKEN = "a-secret-token"


def buildSettings(**overrides) -> Settings:
    """Settings that select only stub providers."""
    base = {
        "assistant": {"handler": "rules", "warmUpOnStart": False},
        "speechToText": {"provider": "scripted"},
        "textToSpeech": {"provider": "tone"},
        "audio": {"outputBackend": "file"},
    }
    base.update(overrides)
    return Settings(**base)


def buildClient(settings: Settings | None = None) -> TestClient:
    resolved = settings or buildSettings()
    service = AssistantService(resolved, audioOutput=MemoryAudioOutput())
    return TestClient(createApp(resolved, service=service))


@pytest.fixture
def client() -> TestClient:
    with buildClient() as testClient:
        yield testClient


@pytest.fixture
def securedClient() -> TestClient:
    settings = buildSettings(api={"host": "127.0.0.1", "port": 8000, "authToken": TOKEN})
    with buildClient(settings) as testClient:
        yield testClient


class TestHealthAndStatus:
    def testHealthNeedsNoToken(self, securedClient: TestClient):
        """Monitoring should not require a credential."""
        response = securedClient.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def testStatusReportsProvidersAndState(self, client: TestClient):
        body = client.get("/status").json()

        assert body["state"] == "idle"
        assert body["sessionId"]
        assert "tone" in body["providers"]["textToSpeech"]
        assert body["busy"] is False

    def testStatusCountsTurns(self, client: TestClient):
        client.post("/assistant/message", json={"text": "hello there"})

        assert client.get("/status").json()["turnCount"] == 1


class TestMessages:
    def testTextRoundTrips(self, client: TestClient):
        response = client.post("/assistant/message", json={"text": "hello there"})

        assert response.status_code == 200
        body = response.json()
        assert "How can I help" in body["reply"]
        assert body["sessionId"]
        assert body["usedLlm"] is False

    def testTimingsAreReported(self, client: TestClient):
        body = client.post("/assistant/message", json={"text": "hello"}).json()

        assert body["timings"]["synthesisSeconds"] >= 0
        assert "streamed" in body["timings"]

    def testAudioIsReturnedOnlyWhenAsked(self, client: TestClient):
        without = client.post("/assistant/message", json={"text": "hello"}).json()
        with_ = client.post(
            "/assistant/message", json={"text": "hello", "includeAudio": True}
        ).json()

        assert without["audio"] is None
        assert with_["audio"]
        restored = AudioBuffer.fromWavBytes(base64.b64decode(with_["audio"]))
        assert restored.durationSeconds > 0

    def testConversationIsRemembered(self, client: TestClient):
        client.post("/assistant/message", json={"text": "hello"})
        client.post("/assistant/message", json={"text": "hello again"})

        assert client.get("/status").json()["turnCount"] == 2

    def testEmptyTextIsRejected(self, client: TestClient):
        assert client.post("/assistant/message", json={"text": ""}).status_code == 422

    def testUnknownFieldIsRejected(self, client: TestClient):
        """A misspelled field is a client bug worth failing on."""
        response = client.post(
            "/assistant/message", json={"text": "hello", "speek": True}
        )

        assert response.status_code == 422

    def testOverlongTextIsRejected(self, client: TestClient):
        response = client.post("/assistant/message", json={"text": "x" * 5000})

        assert response.status_code == 422


class TestSpeech:
    def testSynthesiseReturnsWav(self, client: TestClient):
        response = client.post("/speech/synthesise", json={"text": "hello there"})

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert AudioBuffer.fromWavBytes(response.content).durationSeconds > 0

    def testTranscribeAcceptsAWavUpload(self, client: TestClient):
        audio = AudioBuffer.tone(440.0, 1.0, 16000).toWavBytes()

        response = client.post(
            "/speech/transcribe", files={"audio": ("utterance.wav", audio, "audio/wav")}
        )

        assert response.status_code == 200
        assert response.json()["text"] == "turn the kitchen light off"

    def testTranscribeRejectsNonWav(self, client: TestClient):
        response = client.post(
            "/speech/transcribe",
            files={"audio": ("notes.txt", b"this is not audio", "text/plain")},
        )

        assert response.status_code == 400

    def testTranscribeRejectsEmptyUpload(self, client: TestClient):
        response = client.post(
            "/speech/transcribe", files={"audio": ("empty.wav", b"", "audio/wav")}
        )

        assert response.status_code == 400


class TestAutomation:
    def testDevicesAreListedThroughTheRouter(self):
        settings = buildSettings(
            assistant={"handler": "router", "warmUpOnStart": False},
            automation={"provider": "simulated"},
        )
        with buildClient(settings) as client:
            body = client.get("/automation/devices").json()

        assert any(device["deviceId"] == "light.hallway" for device in body)

    def testActionIsExecuted(self):
        settings = buildSettings(
            assistant={"handler": "router", "warmUpOnStart": False},
            automation={"provider": "simulated"},
        )
        with buildClient(settings) as client:
            response = client.post(
                "/automation/execute",
                json={"action": "light.turnOff", "target": "bedroom light"},
            )

        assert response.status_code == 200
        assert response.json()["succeeded"] is True

    def testForbiddenActionIsRefused(self):
        """An API client is not more trusted than the language model."""
        settings = buildSettings(
            assistant={"handler": "router", "warmUpOnStart": False},
            automation={"provider": "simulated", "allowedActions": ["light.turnOff"]},
        )
        with buildClient(settings) as client:
            response = client.post(
                "/automation/execute",
                json={"action": "lock.unlock", "target": "front door"},
            )

        assert response.status_code == 403

    def testAutomationIsAbsentWithoutTheRouter(self, client: TestClient):
        response = client.get("/automation/devices")

        assert response.status_code == 404
        assert "router" in response.json()["detail"]


class TestAuthentication:
    """§17: authentication before remote clients are allowed."""

    def testRequestWithoutTokenIsRejected(self, securedClient: TestClient):
        response = securedClient.get("/status")

        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"

    def testRequestWithTokenIsAccepted(self, securedClient: TestClient):
        response = securedClient.get(
            "/status", headers={"Authorization": f"Bearer {TOKEN}"}
        )

        assert response.status_code == 200

    def testWrongTokenIsRejected(self, securedClient: TestClient):
        response = securedClient.get(
            "/status", headers={"Authorization": "Bearer not-the-token"}
        )

        assert response.status_code == 401

    def testNoTokenConfiguredMeansNoAuthentication(self, client: TestClient):
        assert client.get("/status").status_code == 200

    def testWebSocketWithoutTokenIsClosed(self, securedClient: TestClient):
        from fastapi import WebSocketDisconnect

        with (
            securedClient.websocket_connect("/ws/assistant") as socket,
            pytest.raises(WebSocketDisconnect),
        ):
            socket.receive_json()

    def testWebSocketAcceptsATokenInTheQueryString(self, securedClient: TestClient):
        """Browsers cannot set headers on a WebSocket handshake."""
        with securedClient.websocket_connect(
            f"/ws/assistant?token={TOKEN}"
        ) as socket:
            assert socket.receive_json()["type"] == "assistant.connected"


class TestAuthenticator:
    def testMissingHeaderFailsWhenTokenRequired(self):
        assert not ApiAuthenticator("secret").isAuthorised(None)

    def testBareTokenIsAccepted(self):
        assert ApiAuthenticator("secret").isAuthorised("secret")

    def testBearerPrefixIsAccepted(self):
        assert ApiAuthenticator("secret").isAuthorised("Bearer secret")

    def testPrefixIsCaseInsensitive(self):
        assert ApiAuthenticator("secret").isAuthorised("bearer secret")

    def testEverythingPassesWhenNoTokenConfigured(self):
        authenticator = ApiAuthenticator(None)

        assert authenticator.isAuthorised(None)
        assert not authenticator.required


class TestWebSocket:
    def testConnectionAnnouncesTheSession(self, client: TestClient):
        with client.websocket_connect("/ws/assistant") as socket:
            hello = socket.receive_json()

        assert hello["type"] == "assistant.connected"
        assert hello["sessionId"]

    def testEventsReachTheClient(self, client: TestClient):
        with client.websocket_connect("/ws/assistant") as socket:
            socket.receive_json()
            socket.send_text(json.dumps({"type": "assistant.message", "text": "hello"}))

            kinds = [socket.receive_json()["type"] for _ in range(4)]

        assert "assistant.state" in kinds
        assert "assistant.response" in kinds

    def testTwoClientsBothObserveTheSameTurn(self, client: TestClient):
        """Exit criterion: several clients, one assistant."""
        with (
            client.websocket_connect("/ws/assistant") as first,
            client.websocket_connect("/ws/assistant") as second,
        ):
            first.receive_json()
            second.receive_json()

            first.send_text(json.dumps({"type": "assistant.message", "text": "hello"}))

            firstKinds = [first.receive_json()["type"] for _ in range(4)]
            secondKinds = [second.receive_json()["type"] for _ in range(4)]

        assert "assistant.response" in firstKinds
        assert "assistant.response" in secondKinds

    def testResponseIsAnnouncedBeforeItIsSpoken(self, client: TestClient):
        """Otherwise a client shows the reply only once the audio has finished."""
        with client.websocket_connect("/ws/assistant") as socket:
            socket.receive_json()
            socket.send_text(json.dumps({"type": "assistant.message", "text": "hello"}))

            kinds = [socket.receive_json()["type"] for _ in range(4)]

        assert kinds.index("assistant.response") < kinds.index(
            "assistant.speechGenerationStarted"
        )

    def testStatusCanBeRequested(self, client: TestClient):
        with client.websocket_connect("/ws/assistant") as socket:
            socket.receive_json()
            socket.send_text(json.dumps({"type": "assistant.status"}))

            reply = socket.receive_json()

        assert reply["type"] == "assistant.status"
        assert reply["state"] == "idle"

    def testPingIsAnswered(self, client: TestClient):
        with client.websocket_connect("/ws/assistant") as socket:
            socket.receive_json()
            socket.send_text(json.dumps({"type": "assistant.ping"}))

            assert socket.receive_json()["type"] == "assistant.pong"

    def testMalformedJsonIsReported(self, client: TestClient):
        with client.websocket_connect("/ws/assistant") as socket:
            socket.receive_json()
            socket.send_text("not json at all")

            reply = socket.receive_json()

        assert reply["type"] == "assistant.error"

    def testUnknownCommandIsReported(self, client: TestClient):
        with client.websocket_connect("/ws/assistant") as socket:
            socket.receive_json()
            socket.send_text(json.dumps({"type": "assistant.nonsense"}))

            reply = socket.receive_json()

        assert reply["type"] == "assistant.error"

    def testMessageWithoutTextIsReported(self, client: TestClient):
        with client.websocket_connect("/ws/assistant") as socket:
            socket.receive_json()
            socket.send_text(json.dumps({"type": "assistant.message", "text": "  "}))

            reply = socket.receive_json()

        assert reply["type"] == "assistant.error"


class TestModelLifetime:
    def testProvidersAreNotReloadedBetweenRequests(self, client: TestClient):
        """§18: a service that reloaded models per request would be unusable."""
        first = client.get("/status").json()["providers"]

        for _ in range(3):
            client.post("/assistant/message", json={"text": "hello"})

        assert client.get("/status").json()["providers"] == first

    def testTurnsAreSerialised(self, client: TestClient):
        """One assistant, one speaker: replies must not overlap."""
        for _ in range(3):
            assert (
                client.post("/assistant/message", json={"text": "hello"}).status_code
                == 200
            )

        assert client.get("/status").json()["busy"] is False
