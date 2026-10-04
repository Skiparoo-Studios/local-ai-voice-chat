"""In-process publish/subscribe bus.

Deliberately small. It exists now because every later stage --- transcription,
tool execution, speech generation --- needs to report progress without knowing
who is listening, and because clients described in §9 of the brief consume
exactly this stream.

Handlers may be synchronous or coroutine functions. A handler that raises is
logged and skipped; one broken subscriber must not stop the others or the
publisher.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from app.events.events import Event

logger = logging.getLogger(__name__)

EventT = TypeVar("EventT", bound=Event)
Handler = Callable[[Any], Awaitable[None] | None]


class Subscription:
    """Handle returned by :meth:`EventBus.subscribe`."""

    __slots__ = ("_active", "_bus", "_eventType", "_handler")

    def __init__(self, bus: EventBus, eventType: type[Event] | None, handler: Handler) -> None:
        self._bus = bus
        self._eventType = eventType
        self._handler = handler
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def unsubscribe(self) -> None:
        """Remove the handler. Safe to call more than once."""
        if self._active:
            self._bus._remove(self._eventType, self._handler)
            self._active = False

    def __enter__(self) -> Subscription:
        return self

    def __exit__(self, *exceptionDetails: object) -> None:
        self.unsubscribe()


class EventBus:
    """Routes events to subscribers."""

    def __init__(self) -> None:
        self._handlers: dict[type[Event], list[Handler]] = {}
        self._globalHandlers: list[Handler] = []
        self._backgroundTasks: set[asyncio.Task[None]] = set()

    # --- Subscription ----------------------------------------------------

    def subscribe(self, eventType: type[EventT], handler: Handler) -> Subscription:
        """Subscribe to ``eventType`` and its subclasses."""
        self._handlers.setdefault(eventType, []).append(handler)
        return Subscription(self, eventType, handler)

    def subscribeAll(self, handler: Handler) -> Subscription:
        """Subscribe to every event."""
        self._globalHandlers.append(handler)
        return Subscription(self, None, handler)

    def _remove(self, eventType: type[Event] | None, handler: Handler) -> None:
        handlers = self._globalHandlers if eventType is None else self._handlers.get(eventType)
        if handlers is None:
            return
        with contextlib.suppress(ValueError):
            handlers.remove(handler)

    def handlersFor(self, eventType: type[Event]) -> list[Handler]:
        """Resolve handlers for ``eventType``, walking its base classes.

        Subscribing to a base class therefore receives subclass events, which
        is what makes :class:`~app.events.events.Event` itself a usable
        subscription target.
        """
        resolved: list[Handler] = []
        for candidate in eventType.__mro__:
            if not (isinstance(candidate, type) and issubclass(candidate, Event)):
                continue
            resolved.extend(self._handlers.get(candidate, ()))
        resolved.extend(self._globalHandlers)
        return resolved

    # --- Publication -----------------------------------------------------

    async def publish(self, event: Event) -> None:
        """Deliver ``event`` to all handlers and wait for them to finish."""
        handlers = self.handlersFor(type(event))
        if not handlers:
            logger.debug("No handlers for %s", event.eventType)
            return

        results = await asyncio.gather(
            *(self._invoke(handler, event) for handler in handlers),
            return_exceptions=True,
        )
        for handler, result in zip(handlers, results, strict=True):
            if isinstance(result, BaseException):
                logger.exception(
                    "Handler %s failed while processing %s",
                    getattr(handler, "__qualname__", repr(handler)),
                    event.eventType,
                    exc_info=result,
                )

    def publishNoWait(self, event: Event) -> asyncio.Task[None]:
        """Publish without waiting, for callers that must not block.

        The returned task is retained by the bus until it completes so that it
        is not garbage collected mid-flight; :meth:`drain` awaits any that are
        still outstanding.
        """
        task = asyncio.create_task(self.publish(event))
        self._backgroundTasks.add(task)
        task.add_done_callback(self._backgroundTasks.discard)
        return task

    async def drain(self) -> None:
        """Wait for events published via :meth:`publishNoWait` to be delivered."""
        while self._backgroundTasks:
            await asyncio.gather(*tuple(self._backgroundTasks), return_exceptions=True)

    @staticmethod
    async def _invoke(handler: Handler, event: Event) -> None:
        result = handler(event)
        if inspect.isawaitable(result):
            await result
