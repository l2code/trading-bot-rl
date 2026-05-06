"""NoOpShadowBrokerAdapter — accepts intents, never sends them anywhere.

Used in shadow mode. Returns ``BrokerOrder`` records with status
``ACCEPTED`` so the rest of the pipeline can run, but nothing leaves
the process. Positions/account always come back empty.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from rl_swing.domain import (
    AccountSnapshot,
    BrokerOrder,
    OrderIntent,
    PositionSnapshot,
)


class NoOpShadowBrokerAdapter:
    broker_id: str = "shadow_noop"
    environment = "shadow"

    def submit_order(self, intent: OrderIntent) -> BrokerOrder:
        now = datetime.utcnow()
        return BrokerOrder(
            internal_order_id=intent.intent_id,
            broker_order_id=f"shadow-{uuid.uuid4().hex[:8]}",
            client_order_id=intent.client_order_id,
            environment="shadow",
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            time_in_force=intent.time_in_force,
            requested_qty=intent.quantity,
            limit_price=intent.limit_price,
            status="ACCEPTED",
            submitted_at=now,
            updated_at=now,
            raw_request={"shadow": True},
            raw_response={"shadow": True, "no_op": True},
        )

    def cancel_order(self, broker_order_id: str) -> None:
        return None

    def list_open_orders(self) -> list[BrokerOrder]:
        return []

    def list_positions(self) -> list[PositionSnapshot]:
        return []

    def get_account_snapshot(self) -> AccountSnapshot:
        now = datetime.utcnow()
        return AccountSnapshot(
            as_of=now, source="internal",
            cash=0.0, buying_power=0.0, equity=0.0, portfolio_value=0.0,
            raw={"shadow": True},
        )
