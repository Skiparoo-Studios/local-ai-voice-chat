"""Event bus behaviour."""

from __future__ import annotations

import asyncio

import pytest

from app.events import (
    AssistantError,
    AssistantState,
    AssistantStateChanged,
    Event,
    EventBus,
    TranscriptionCompleted,
)


class TestSubscription:
    async def testHandlerReceivesMatchingEvent(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(TranscriptionCompleted, received.append)

        await bus.publish(TranscriptionCompleted(text="lights off"))

        assert len(received) == 1
        assert received[0].text == "lights off"

    async def testHandlerIgnoresOtherEventTypes(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(TranscriptionCompleted, received.append)

        await bus.publish(AssistantStateChanged(state=AssistantState.listening))

        assert received == []

    async def testSubscribeAllReceivesEverything(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribeAll(received.append)

        await bus.publish(TranscriptionCompleted(text="hello"))
        await bus.publish(AssistantStateChanged(state=AssistantState.thinking))

        assert len(received) == 2

    async def testSubscribingToBaseClassReceivesSubclasses(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(Event, received.append)

        await bus.publish(TranscriptionCompleted(text="hello"))

        assert len(received) == 1

    async def testAsyncHandlersAreAwaited(self):
        bus = EventBus()
        received: list[str] = []

        async def handler(event: TranscriptionCompleted) -> None:
            await asyncio.sleep(0)
            received.append(event.text)

        bus.subscribe(TranscriptionCompleted, handler)
        await bus.publish(TranscriptionCompleted(text="done"))

        assert received == ["done"]

    async def testSyncAndAsyncHandlersCoexist(self):
        bus = EventBus()
        received: list[str] = []

        async def asyncHandler(event: TranscriptionCompleted) -> None:
            received.append("async")

        bus.subscribe(TranscriptionCompleted, asyncHandler)
        bus.subscribe(TranscriptionCompleted, lambda event: received.append("sync"))

        await bus.publish(TranscriptionCompleted(text="x"))

        assert sorted(received) == ["async", "sync"]

    async def testPublishWithNoHandlersIsHarmless(self):
        await EventBus().publish(AssistantStateChanged(state=AssistantState.idle))


class TestUnsubscribe:
    async def testUnsubscribeStopsDelivery(self):
        bus = EventBus()
        received: list[Event] = []
        subscription = bus.subscribe(TranscriptionCompleted, received.append)

        await bus.publish(TranscriptionCompleted(text="first"))
        subscription.unsubscribe()
        await bus.publish(TranscriptionCompleted(text="second"))

        assert [event.text for event in received] == ["first"]
        assert not subscription.active

    async def testUnsubscribeIsIdempotent(self):
        bus = EventBus()
        subscription = bus.subscribe(TranscriptionCompleted, lambda event: None)

        subscription.unsubscribe()
        subscription.unsubscribe()

    async def testUnsubscribeGlobalHandler(self):
        bus = EventBus()
        received: list[Event] = []
        subscription = bus.subscribeAll(received.append)
        subscription.unsubscribe()

        await bus.publish(TranscriptionCompleted(text="x"))

        assert received == []

    async def testSubscriptionWorksAsContextManager(self):
        bus = EventBus()
        received: list[Event] = []

        with bus.subscribe(TranscriptionCompleted, received.append):
            await bus.publish(TranscriptionCompleted(text="inside"))
        await bus.publish(TranscriptionCompleted(text="outside"))

        assert [event.text for event in received] == ["inside"]


class TestErrorIsolation:
    async def testFailingHandlerDoesNotStopOthers(self, caplog: pytest.LogCaptureFixture):
        bus = EventBus()
        received: list[Event] = []

        def brokenHandler(event: Event) -> None:
            raise RuntimeError("handler exploded")

        bus.subscribe(TranscriptionCompleted, brokenHandler)
        bus.subscribe(TranscriptionCompleted, received.append)

        await bus.publish(TranscriptionCompleted(text="survives"))

        assert len(received) == 1
        assert "handler exploded" in caplog.text

    async def testFailingHandlerDoesNotPropagateToPublisher(self):
        bus = EventBus()

        async def brokenHandler(event: Event) -> None:
            raise ValueError("nope")

        bus.subscribe(TranscriptionCompleted, brokenHandler)

        await bus.publish(TranscriptionCompleted(text="x"))


class TestBackgroundPublication:
    async def testPublishNoWaitDeliversAfterDrain(self):
        bus = EventBus()
        received: list[Event] = []

        async def slowHandler(event: Event) -> None:
            await asyncio.sleep(0.01)
            received.append(event)

        bus.subscribe(TranscriptionCompleted, slowHandler)

        bus.publishNoWait(TranscriptionCompleted(text="deferred"))
        assert received == []

        await bus.drain()
        assert len(received) == 1

    async def testDrainWithNothingOutstandingReturns(self):
        await EventBus().drain()


class TestEventSerialisation:
    def testStateEventMatchesWireFormat(self):
        payload = AssistantStateChanged(state=AssistantState.listening).toDict()

        assert payload["type"] == "assistant.state"
        assert payload["state"] == "listening"

    def testTranscriptionEventMatchesWireFormat(self):
        payload = TranscriptionCompleted(text="turn the lounge room light off").toDict()

        assert payload["type"] == "assistant.transcription"
        assert payload["text"] == "turn the lounge room light off"

    def testTimestampSerialisesAsIsoString(self):
        payload = AssistantError(message="failed").toDict()

        assert isinstance(payload["timestamp"], str)
        assert "T" in payload["timestamp"]

    def testEventsAreImmutable(self):
        event = TranscriptionCompleted(text="original")

        with pytest.raises((AttributeError, TypeError)):
            event.text = "changed"
