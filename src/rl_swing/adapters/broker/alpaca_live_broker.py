"""AlpacaLiveBrokerAdapter — STUB.

Live trading is gated by the ``RL_SWING_LIVE_APPROVAL_TOKEN`` env var,
the runtime ``allow_live_trading`` flag, and the runtime ``place_orders``
flag. All three must be true before this adapter will be allowed to
actually place a real order. See ADR 0005 and the safety check in
``runtime.dependency_container._safety_check``.
"""
from __future__ import annotations

from rl_swing.domain import (
    AccountSnapshot,
    BrokerOrder,
    OrderIntent,
    PositionSnapshot,
)


class AlpacaLiveBrokerAdapter:
    broker_id: str = "alpaca_live"
    environment = "live"

    def __init__(self, **_: object) -> None:
        pass

    def submit_order(self, intent: OrderIntent) -> BrokerOrder:
        raise NotImplementedError(
            "AlpacaLiveBrokerAdapter is intentionally unimplemented in this build."
        )

    def cancel_order(self, broker_order_id: str) -> None:
        raise NotImplementedError

    def list_open_orders(self) -> list[BrokerOrder]:
        raise NotImplementedError

    def list_positions(self) -> list[PositionSnapshot]:
        raise NotImplementedError

    def get_account_snapshot(self) -> AccountSnapshot:
        raise NotImplementedError
