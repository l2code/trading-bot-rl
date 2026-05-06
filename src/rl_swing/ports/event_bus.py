"""EventBus port.

A tiny in-process pub-sub used by services to emit ``AuditEvent``s.
The default in-memory implementation is in
``rl_swing.runtime.event_bus``.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from rl_swing.domain import AuditEvent

EventListener = Callable[[AuditEvent], None]


@runtime_checkable
class EventBus(Protocol):
    def publish(self, event: AuditEvent) -> None: ...
    def subscribe(self, listener: EventListener) -> None: ...
    def unsubscribe(self, listener: EventListener) -> None: ...
