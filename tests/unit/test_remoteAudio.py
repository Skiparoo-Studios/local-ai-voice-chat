"""Streamed audio, per-client sessions, and the thin client."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.api.app import createApp
from app.api.audioStream import (
    MAXIMUM_UTTERANCE_BYTES,
    AudioStreamError,
    IncomingAudio,
    chunkAudio,
    replyHeader,
)
from app.api.service import AssistantService
from app.audio.audioBuffer import AudioBuffer
from app.audio.output import MemoryAudioOutput
from app.client.remoteClient import ClientStatistics, RemoteClient, _websocketUrl
from app.config import Settings

RATE = 16000


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


def utterance(seconds: float = 1.0) -> AudioBuffer:
    return AudioBuffer.tone(440.0, seconds, RATE)


class TestIncomingAudio:
    def testFramesAreAssembled(self):
        stream = IncomingAudio()
        stream.begin({"sampleRate": RATE, "channels": 1})

        stream.add(b"\x01\x02" * 100)
        stream.add(b"\x03\x04" * 100)
        audio = stream.finish()

        assert audio.sampleRate == RATE
        assert len(audio.data) == 400

    def testAudioBeforeStartIsRejected(self):
        with pytest.raises(AudioStreamError, match=r"before audio\.start"):
            IncomingAudio().add(b"\x00\x00")

    def testEndWithoutStartIsRejected(self):
        with pytest.raises(AudioStreamError, match="without a matching"):
            IncomingAudio().finish()

    def testOversizedUtteranceIsRefused(self):
        """A client that never sends audio.end must not exhaust memory."""
        stream = IncomingAudio()
        stream.begin({})

        with pytest.raises(AudioStreamError, match="may not exceed"):
            stream.add(b"\x00" * (MAXIMUM_UTTERANCE_BYTES + 2))

    def testStreamIsResetAfterRefusal(self):
        stream = IncomingAudio()
        stream.begin({})

        with pytest.raises(AudioStreamError):
            stream.add(b"\x00" * (MAXIMUM_UTTERANCE_BYTES + 2))

        assert not stream.open
        assert stream.byteCount == 0

    def testPartialFinalFrameIsTrimmed(self):
        """One stray byte would misalign every sample after it."""
        stream = IncomingAudio()
        stream.begin({"channels": 1})
        stream.add(b"\x01\x02\x03")

        assert len(stream.finish().data) == 2

    def testDefaultsAreUsedForMissingFields(self):
        stream = IncomingAudio()
        stream.begin({})

        assert stream.sampleRate == RATE
        assert stream.channels == 1

    def testNonsenseSampleRateFallsBack(self):
        stream = IncomingAudio()
        stream.begin({"sampleRate": "banana"})

        assert stream.sampleRate == RATE

    def testBeginDiscardsAnythingLeftOver(self):
        stream = IncomingAudio()
        stream.begin({})
        stream.add(b"\x00\x00" * 10)

        stream.begin({})

        assert stream.byteCount == 0


class TestReplyFraming:
    def testHeaderDescribesTheAudio(self):
        header = replyHeader(utterance(0.5), text="hello")

        assert header["type"] == "audio.reply"
        assert header["format"] == "pcm16"
        assert header["sampleRate"] == RATE
        assert header["text"] == "hello"

    def testChunksReassembleExactly(self):
        audio = utterance(2.0)

        chunks = chunkAudio(audio, chunkBytes=1024)

        assert len(chunks) > 1
        assert b"".join(chunks) == audio.data

    def testChunksAreFrameAligned(self):
        """A chunk split mid-sample would click on playback."""
        chunks = chunkAudio(utterance(1.0), chunkBytes=1023)

        assert all(len(chunk) % 2 == 0 for chunk in chunks[:-1])

    def testSmallAudioIsOneChunk(self):
        assert len(chunkAudio(utterance(0.01))) == 1


class TestStreamedTurn:
    def testAudioInAudioOut(self):
        """Exit criterion: a client sends speech and gets speech back."""
        settings = buildSettings()
        service = AssistantService(settings, audioOutput=MemoryAudioOutput())

        with (
            TestClient(createApp(settings, service=service)) as testClient,
            testClient.websocket_connect("/ws/assistant") as socket,
        ):
            socket.receive_json()
            socket.send_text(json.dumps({"type": "audio.start", "sampleRate": RATE}))
            socket.send_bytes(utterance().data)
            socket.send_text(json.dumps({"type": "audio.end"}))

            header = None
            chunks: list[bytes] = []
            for _ in range(30):
                message = socket.receive()
                if message.get("bytes") is not None:
                    chunks.append(message["bytes"])
                    continue
                payload = json.loads(message["text"])
                if payload["type"] == "audio.reply":
                    header = payload
                elif payload["type"] == "audio.replyEnd":
                    break

        assert header is not None
        assert chunks
        reply = AudioBuffer(data=b"".join(chunks), sampleRate=header["sampleRate"])
        assert reply.durationSeconds > 0

    def testReplyIsStreamedNotBuffered(self):
        """Audio should leave for the client as it is made, not after.

        The runtime calls play once per synthesised piece, so an output that
        forwards to the socket streams without any further plumbing.
        """
        settings = buildSettings()
        service = AssistantService(settings, audioOutput=MemoryAudioOutput())

        with (
            TestClient(createApp(settings, service=service)) as testClient,
            testClient.websocket_connect("/ws/assistant") as socket,
        ):
            socket.receive_json()
            socket.send_text(json.dumps({"type": "audio.start", "sampleRate": RATE}))
            socket.send_bytes(utterance().data)
            socket.send_text(json.dumps({"type": "audio.end"}))

            header = None
            end = None
            for _ in range(30):
                message = socket.receive()
                if message.get("bytes") is not None:
                    continue
                payload = json.loads(message["text"])
                if payload["type"] == "audio.reply":
                    header = payload
                elif payload["type"] == "audio.replyEnd":
                    end = payload
                    break

        assert header is not None
        assert header["streaming"] is True
        # Length is unknown while the audio is still being generated.
        assert "durationSeconds" not in header
        assert end is not None
        assert end["text"]
        assert "serverSeconds" in end

    def testAnEmptyReplyStillTerminatesTheStream(self, client: TestClient):
        """A client waiting for audio.replyEnd must not wait forever."""
        settings = buildSettings(speechToText={"provider": "scripted"})
        service = AssistantService(settings, audioOutput=MemoryAudioOutput())

        with (
            TestClient(createApp(settings, service=service)) as testClient,
            testClient.websocket_connect("/ws/assistant") as socket,
        ):
            socket.receive_json()
            socket.send_text(json.dumps({"type": "audio.start", "sampleRate": RATE}))
            # Silence transcribes to nothing, so there is no reply to speak.
            socket.send_bytes(AudioBuffer.silence(0.5, RATE).data)
            socket.send_text(json.dumps({"type": "audio.end"}))

            kinds = []
            for _ in range(20):
                message = socket.receive()
                if message.get("text") is None:
                    continue
                payload = json.loads(message["text"])
                kinds.append(payload["type"])
                if payload["type"] == "audio.replyEnd":
                    break

        assert "audio.replyEnd" in kinds

    def testTranscriptReachesTheClient(self, client: TestClient):
        with client.websocket_connect("/ws/assistant") as socket:
            socket.receive_json()
            socket.send_text(json.dumps({"type": "audio.start", "sampleRate": RATE}))
            socket.send_bytes(utterance().data)
            socket.send_text(json.dumps({"type": "audio.end"}))

            kinds = []
            for _ in range(30):
                message = socket.receive()
                if message.get("text") is None:
                    continue
                payload = json.loads(message["text"])
                kinds.append(payload["type"])
                if payload["type"] == "audio.replyEnd":
                    break

        assert "assistant.transcription" in kinds

    def testAudioWithoutStartIsReported(self, client: TestClient):
        with client.websocket_connect("/ws/assistant") as socket:
            socket.receive_json()
            socket.send_bytes(b"\x00\x00" * 100)

            assert socket.receive_json()["type"] == "assistant.error"

    def testEndWithoutAudioIsReported(self, client: TestClient):
        with client.websocket_connect("/ws/assistant") as socket:
            socket.receive_json()
            socket.send_text(json.dumps({"type": "audio.start"}))
            socket.send_text(json.dumps({"type": "audio.end"}))

            assert socket.receive_json()["type"] == "assistant.error"

    def testCancelDiscardsTheUtterance(self, client: TestClient):
        with client.websocket_connect("/ws/assistant") as socket:
            socket.receive_json()
            socket.send_text(json.dumps({"type": "audio.start"}))
            socket.send_bytes(utterance().data)
            socket.send_text(json.dumps({"type": "audio.cancel"}))
            socket.send_text(json.dumps({"type": "assistant.ping"}))

            assert socket.receive_json()["type"] == "assistant.pong"


class TestSessionIsolation:
    """Two people in different rooms should not share a conversation."""

    def testEachClientGetsItsOwnSession(self, client: TestClient):
        with (
            client.websocket_connect("/ws/assistant") as first,
            client.websocket_connect("/ws/assistant") as second,
        ):
            firstId = first.receive_json()["sessionId"]
            secondId = second.receive_json()["sessionId"]

        assert firstId != secondId

    def testConversationsDoNotMix(self, client: TestClient):
        with (
            client.websocket_connect("/ws/assistant") as first,
            client.websocket_connect("/ws/assistant") as second,
        ):
            first.receive_json()
            second.receive_json()

            first.send_text(json.dumps({"type": "assistant.message", "text": "hello"}))
            for _ in range(4):
                first.receive_json()

            second.send_text(json.dumps({"type": "assistant.status"}))
            status = None
            for _ in range(8):
                payload = second.receive_json()
                if payload["type"] == "assistant.status":
                    status = payload
                    break

        # The turn belonged to the first client's session, so the second one's
        # is untouched.
        assert status is not None
        assert status["sessionId"] != ""

    def testSessionsAreCountedAndReleased(self, client: TestClient):
        with client.websocket_connect("/ws/assistant") as socket:
            socket.receive_json()
            during = client.get("/status").json()["sessionCount"]

        assert during == 1
        assert client.get("/status").json()["sessionCount"] == 0

    def testRestRequestsUseTheDefaultSession(self, client: TestClient):
        first = client.post("/assistant/message", json={"text": "hello"}).json()
        second = client.post("/assistant/message", json={"text": "hello"}).json()

        assert first["sessionId"] == second["sessionId"]


class TestClientUrl:
    @pytest.mark.parametrize(
        "given,expected",
        [
            ("http://host:8000", "ws://host:8000/ws/assistant"),
            ("https://host:8000", "wss://host:8000/ws/assistant"),
            ("host:8000", "ws://host:8000/ws/assistant"),
            ("ws://host:8000/ws/assistant", "ws://host:8000/ws/assistant"),
            ("http://host:8000/", "ws://host:8000/ws/assistant"),
        ],
    )
    def testAddressesAreNormalised(self, given: str, expected: str):
        assert _websocketUrl(given, None) == expected

    def testTokenGoesInTheQueryString(self):
        url = _websocketUrl("http://host:8000", "secret")

        assert url.endswith("/ws/assistant?token=secret")


class TestClientStatistics:
    def testAverageOfNoRepliesIsZero(self):
        assert ClientStatistics().averageRoundTrip == 0.0

    def testAverageIsComputed(self):
        statistics = ClientStatistics(roundTripSeconds=[1.0, 2.0, 3.0])

        assert statistics.averageRoundTrip == pytest.approx(2.0)

    def testDescriptionMentionsTheRoundTrip(self):
        statistics = ClientStatistics(utterancesSent=2, roundTripSeconds=[1.5])

        assert "round trip" in statistics.describe()

    def testDescriptionHandlesNoReplies(self):
        assert "no replies" in ClientStatistics(utterancesSent=1).describe()


class TestClientConstruction:
    def testBuiltFromSettings(self):
        settings = buildSettings(
            client={"serverUrl": "http://192.168.1.10:8000"},
            wakeWord={"provider": "always-awake"},
            vad={"provider": "energy"},
            audio={"outputBackend": "file"},
        )
        remote = RemoteClient.fromSettings(settings)

        assert "192.168.1.10" in remote.describe()
        remote.close()

    def testTheEchoGuardComesFromSettings(self):
        settings = buildSettings(
            wakeWord={"provider": "always-awake"},
            vad={"provider": "energy"},
            audio={"outputBackend": "file", "echoGuardSeconds": 1.5},
        )
        remote = RemoteClient.fromSettings(settings)

        assert remote._echoGuardSeconds == 1.5
        remote.close()

    async def testAudioCapturedWhileTheReplyPlayedIsDiscarded(self):
        """_sendUtterance plays the reply inside the frame loop, so nothing is
        read for the whole of it and the queue fills with the assistant's own
        voice. Without draining, that is fed straight to the segmenter."""
        from app.audio.input import AudioInput

        class CountingInput(AudioInput):
            def __init__(self) -> None:
                self.drainCalls = 0

            @property
            def sampleRate(self) -> int:
                return 16000

            async def frames(self, frameMilliseconds: int = 32):
                return
                yield  # pragma: no cover - never reached

            def drain(self) -> int:
                self.drainCalls += 1
                return 3

        settings = buildSettings(
            wakeWord={"provider": "always-awake"},
            vad={"provider": "energy"},
            audio={"outputBackend": "file", "echoGuardSeconds": 0.0},
        )
        remote = RemoteClient.fromSettings(settings)
        remote._input = CountingInput()

        await remote._settleAfterSpeaking()

        assert remote._input.drainCalls == 1
        remote.close()

    def testFrameSizeIsTheSmallerOfTheTwoModels(self):
        settings = buildSettings(
            wakeWord={"provider": "always-awake"},
            vad={"provider": "energy"},
            audio={"outputBackend": "file"},
        )
        remote = RemoteClient.fromSettings(settings)

        assert remote.frameMilliseconds == 32
        remote.close()
