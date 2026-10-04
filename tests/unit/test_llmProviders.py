"""Language model providers, tested against a mock HTTP transport.

httpx's MockTransport exercises the real request building, response parsing and
error handling without a running service, so the protocol handling is covered
even on a machine with no Ollama installed.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import Settings
from app.intelligence.llmProvider import (
    LLM_PROVIDER_REGISTRY,
    LlmError,
    LlmProvider,
    LlmResponse,
    LlmUnavailableError,
    UnknownLlmProviderError,
    createLlmProvider,
    registerLlmProvider,
    resolveLlmProviderClass,
)
from app.intelligence.providers.ollamaProvider import OllamaProvider
from app.intelligence.providers.openAiCompatibleProvider import OpenAiCompatibleProvider
from app.intelligence.providers.scriptedLlmProvider import ScriptedLlmProvider

MESSAGES = [{"role": "user", "content": "turn the kitchen light off"}]


def ndjson(*objects: dict) -> bytes:
    return b"".join(json.dumps(item).encode() + b"\n" for item in objects)


def sse(*chunks: str) -> bytes:
    return b"".join(f"data: {chunk}\n\n".encode() for chunk in chunks)


def ollamaFragment(content: str, done: bool = False) -> dict:
    return {"model": "test", "message": {"role": "assistant", "content": content}, "done": done}


def openAiFragment(content: str) -> str:
    return json.dumps({"choices": [{"delta": {"content": content}, "index": 0}]})


def mockOllama(handler) -> OllamaProvider:
    return OllamaProvider(model="test", transport=httpx.MockTransport(handler))


def mockOpenAi(handler) -> OpenAiCompatibleProvider:
    return OpenAiCompatibleProvider(model="test", transport=httpx.MockTransport(handler))


class TestRegistry:
    def testResolvesRegisteredProviders(self):
        assert resolveLlmProviderClass("ollama") is OllamaProvider
        assert resolveLlmProviderClass("openai-compatible") is OpenAiCompatibleProvider

    def testUnknownProviderListsAvailableOnes(self):
        with pytest.raises(UnknownLlmProviderError, match="Available:"):
            resolveLlmProviderClass("does-not-exist")

    def testProvidersCanBeRegistered(self):
        registerLlmProvider(
            "custom", "app.intelligence.providers.scriptedLlmProvider:ScriptedLlmProvider"
        )
        try:
            assert resolveLlmProviderClass("custom") is ScriptedLlmProvider
        finally:
            LLM_PROVIDER_REGISTRY.pop("custom", None)

    def testConfigurationSelectsTheProvider(self):
        """Exit criterion: swapping providers is configuration, not code."""
        ollama = createLlmProvider(Settings(llm={"provider": "ollama", "model": "llama3.2"}))
        openAi = createLlmProvider(
            Settings(
                llm={
                    "provider": "openai-compatible",
                    "model": "local",
                    "baseUrl": "http://localhost:8080/v1",
                }
            )
        )

        assert isinstance(ollama, OllamaProvider)
        assert isinstance(openAi, OpenAiCompatibleProvider)
        assert isinstance(ollama, LlmProvider)


class TestOllamaGenerate:
    async def testParsesACompleteReply(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "llama3.2",
                    "message": {"role": "assistant", "content": "The light is off."},
                    "prompt_eval_count": 12,
                    "eval_count": 5,
                },
            )

        result = await mockOllama(handler).generate(MESSAGES)

        assert result.text == "The light is off."
        assert result.promptTokens == 12
        assert result.completionTokens == 5

    async def testSendsModelAndMessages(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": "ok"}})

        provider = OllamaProvider(
            model="qwen2.5", temperature=0.3, maxTokens=64, transport=httpx.MockTransport(handler)
        )
        await provider.generate(MESSAGES)

        assert captured["model"] == "qwen2.5"
        assert captured["messages"] == MESSAGES
        assert captured["stream"] is False
        assert captured["options"]["temperature"] == 0.3
        assert captured["options"]["num_predict"] == 64

    async def testKeepAliveIsSentOnlyWhenConfigured(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": "ok"}})

        await mockOllama(handler).generate(MESSAGES)
        assert "keep_alive" not in captured

        provider = OllamaProvider(
            model="test", keepAlive="10m", transport=httpx.MockTransport(handler)
        )
        await provider.generate(MESSAGES)
        assert captured["keep_alive"] == "10m"

    async def testMissingModelIsReportedWithTheFix(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "model not found"})

        with pytest.raises(LlmUnavailableError, match="ollama pull"):
            await mockOllama(handler).generate(MESSAGES)

    async def testConnectionFailureNamesTheAddress(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        with pytest.raises(LlmUnavailableError, match="Could not reach Ollama"):
            await mockOllama(handler).generate(MESSAGES)

    def testEmptyModelIsRejectedAtConstruction(self):
        with pytest.raises(LlmError, match="No language model configured"):
            OllamaProvider(model="")

    def testTheMessageSaysHowToFixIt(self):
        """A configuration error is only useful if it names the setting."""
        with pytest.raises(LlmError) as raised:
            OllamaProvider(model="")

        message = str(raised.value)
        assert "llm.model" in message
        assert "VOICE_LLM__MODEL" in message
        assert "ollama list" in message
        assert "--handler rules" in message


class TestOllamaStreaming:
    async def testYieldsFragmentsInOrder(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=ndjson(
                    ollamaFragment("The light "),
                    ollamaFragment("is now "),
                    ollamaFragment("off."),
                    ollamaFragment("", done=True),
                ),
            )

        fragments = [f async for f in mockOllama(handler).generateStream(MESSAGES)]

        assert fragments == ["The light ", "is now ", "off."]

    async def testRequestsStreaming(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, content=ndjson(ollamaFragment("x")))

        [f async for f in mockOllama(handler).generateStream(MESSAGES)]

        assert captured["stream"] is True

    async def testBlankAndUnparseableLinesAreSkipped(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=b"\n" + ndjson(ollamaFragment("ok")) + b"not json\n\n",
            )

        fragments = [f async for f in mockOllama(handler).generateStream(MESSAGES)]

        assert fragments == ["ok"]

    async def testErrorInStreamIsRaised(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=ndjson({"error": "out of memory"}))

        with pytest.raises(LlmError, match="out of memory"):
            [f async for f in mockOllama(handler).generateStream(MESSAGES)]


class TestOllamaIntrospection:
    async def testAvailabilityIsAQuestionNotAFailure(self):
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        assert await mockOllama(refuse).isAvailable() is False

    async def testAvailabilityWhenReachable(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"models": []})

        assert await mockOllama(handler).isAvailable() is True

    async def testListsInstalledModels(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"models": [{"name": "qwen2.5:3b"}, {"name": "llama3.2"}]}
            )

        assert await mockOllama(handler).listModels() == ["llama3.2", "qwen2.5:3b"]


class TestOpenAiCompatible:
    async def testParsesACompleteReply(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "local",
                    "choices": [{"message": {"content": "The light is off."}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                },
            )

        result = await mockOpenAi(handler).generate(MESSAGES)

        assert result.text == "The light is off."
        assert result.promptTokens == 10

    async def testStreamYieldsDeltas(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse(
                    openAiFragment("The light "),
                    openAiFragment("is off."),
                    "[DONE]",
                ),
            )

        fragments = [f async for f in mockOpenAi(handler).generateStream(MESSAGES)]

        assert fragments == ["The light ", "is off."]

    async def testStreamStopsAtDoneMarker(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=sse(openAiFragment("first"), "[DONE]", openAiFragment("ignored"))
            )

        fragments = [f async for f in mockOpenAi(handler).generateStream(MESSAGES)]

        assert fragments == ["first"]

    async def testApiKeyBecomesABearerHeader(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["authorization"] = request.headers.get("authorization")
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        provider = OpenAiCompatibleProvider(
            model="local", apiKey="secret-key", transport=httpx.MockTransport(handler)
        )
        await provider.generate(MESSAGES)

        assert captured["authorization"] == "Bearer secret-key"

    async def testRejectedCredentialsAreExplained(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "unauthorized"})

        with pytest.raises(LlmUnavailableError, match="VOICE_LLM__APIKEY"):
            await mockOpenAi(handler).generate(MESSAGES)

    async def testNotFoundMentionsTheBaseUrlConvention(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "no such model"})

        with pytest.raises(LlmUnavailableError, match="/v1"):
            await mockOpenAi(handler).generate(MESSAGES)

    async def testApiKeyIsReadFromSettings(self):
        settings = Settings(
            llm={
                "provider": "openai-compatible",
                "model": "local",
                "apiKey": "from-config",
            }
        )
        provider = createLlmProvider(settings)

        assert provider._apiKey == "from-config"


class TestScriptedProvider:
    async def testReturnsRepliesInOrder(self):
        provider = ScriptedLlmProvider(replies=("first", "second"))

        assert (await provider.generate(MESSAGES)).text == "first"
        assert (await provider.generate(MESSAGES)).text == "second"

    async def testStreamsWordByWord(self):
        provider = ScriptedLlmProvider(replies=("one two three",))

        fragments = [f async for f in provider.generateStream(MESSAGES)]

        assert fragments == ["one", " two", " three"]
        assert "".join(fragments) == "one two three"

    async def testRepliesRepeat(self):
        provider = ScriptedLlmProvider(replies=("only",))

        await provider.generate(MESSAGES)

        assert (await provider.generate(MESSAGES)).text == "only"

    async def testCompleteGenerationCostsTheSameAsStreaming(self):
        """Otherwise a streaming comparison measures against a false baseline."""
        reply = "one two three four five"
        delay = 0.01
        provider = ScriptedLlmProvider(replies=(reply,), fragmentDelaySeconds=delay)

        result = await provider.generate(MESSAGES)

        assert result.durationSeconds >= delay * len(reply.split()) * 0.8


class TestLlmResponse:
    def testEmptyIsDetected(self):
        assert LlmResponse(text="  ").isEmpty
        assert not LlmResponse(text="hello").isEmpty

    def testTokenRateIsComputed(self):
        response = LlmResponse(text="x", completionTokens=50, durationSeconds=2.0)

        assert response.tokensPerSecond == pytest.approx(25.0)

    def testTokenRateIsZeroWithoutData(self):
        assert LlmResponse(text="x").tokensPerSecond == 0.0

    def testDescriptionMentionsTokensAndTime(self):
        description = LlmResponse(
            text="x", completionTokens=50, durationSeconds=2.0
        ).describe()

        assert "50 tokens" in description
        assert "tok/s" in description
