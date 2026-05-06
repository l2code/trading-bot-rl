"""AlpacaPaperBrokerAdapter — STUB.

This adapter is intentionally not implemented yet. The current build
focuses on Phases 0A–5 (research-grade core). Wiring up real Alpaca
calls is Phase 8.

When you implement it:
    * Use ``alpaca-py`` (the official SDK).
    * Read keys from ``ALPACA_PAPER_API_KEY`` / ``ALPACA_PAPER_SECRET_KEY``.
    * Use ``OrderIntent.client_order_id`` for idempotency on every submit.
    * Translate Alpaca order/position/account objects to the domain types
      defined in ``rl_swing.domain``. No alpaca-py types may leak past
      this adapter.
"""
from __future__ import annotations

from rl_swing.domain import (
    AccountSnapshot,
    BrokerOrder,
    OrderIntent,
    PositionSnapshot,
)


class AlpacaPaperBrokerAdapter:
    broker_id: str = "alpaca_paper"
    environment = "paper"

    def __init__(self, **_: object) -> None:
        pass

    def submit_order(self, intent: OrderIntent) -> BrokerOrder:
        raise NotImplementedError(
            "AlpacaPaperBrokerAdapter not implemented in this build. "
            "Phase 8 of the spec; see docs/adr/0003-alpaca-as-broker-adapter.md."
        )

    def cancel_order(self, broker_order_id: str) -> None:
        raise NotImplementedError("AlpacaPaperBrokerAdapter not implemented.")

    def list_open_orders(self) -> list[BrokerOrder]:
        raise NotImplementedError("AlpacaPaperBrokerAdapter not implemented.")

    def list_positions(self) -> list[PositionSnapshot]:
        raise NotImplementedError("AlpacaPaperBrokerAdapter not implemented.")

    def get_account_snapshot(self) -> AccountSnapshot:
        raise NotImplementedError("AlpacaPaperBrokerAdapter not implemented.")
