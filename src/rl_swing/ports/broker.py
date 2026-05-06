"""BrokerAdapter port.

Adapters: SimulatedBrokerAdapter, NoOpShadowBrokerAdapter,
AlpacaPaperBrokerAdapter, AlpacaLiveBrokerAdapter. No broker SDK
imports outside ``rl_swing.adapters.broker.*``.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from rl_swing.domain import (
    AccountSnapshot,
    BrokerOrder,
    Environment,
    OrderIntent,
    PositionSnapshot,
)


@runtime_checkable
class BrokerAdapter(Protocol):
    broker_id: str
    environment: Environment

    def submit_order(self, intent: OrderIntent) -> BrokerOrder: ...

    def cancel_order(self, broker_order_id: str) -> None: ...

    def list_open_orders(self) -> list[BrokerOrder]: ...

    def list_positions(self) -> list[PositionSnapshot]: ...

    def get_account_snapshot(self) -> AccountSnapshot: ...
