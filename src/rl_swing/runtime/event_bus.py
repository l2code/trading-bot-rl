"""In-memory ``EventBus`` implementation.

Listeners are called synchronously in registration order. If a listener
raises, the exception is logged but does not stop other listeners.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from rl_swing.domain import AuditEvent
from rl_swing.ports import EventListener

_log = logging.getLogger(__name__)


class InMemoryEventBus:
    def __init__(self) -> None:
        self._listeners: list[EventListener] = []

    def subscribe(self, listener: EventListener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def unsubscribe(self, listener: EventListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def publish(self, event: AuditEvent) -> None:
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:  # pragma: no cover - defensive
                _log.exception("event listener raised on %s", event.event_type)


def make_audit_logger(min_level: int = logging.INFO) -> Callable[[AuditEvent], None]:
    """A simple listener that prints events through stdlib logging."""
    logger = logging.getLogger("rl_swing.audit")

    def _listener(ev: AuditEvent) -> None:
        logger.log(
            min_level,
            "%s correlation=%s payload=%s",
            ev.event_type.value,
            ev.correlation_id,
            ev.payload,
        )

    return _listener
