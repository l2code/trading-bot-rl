"""Order-flow domain types.

These are intentionally broker-agnostic. ``BrokerOrder`` records the
broker's view; ``OrderIntent`` records ours. Both carry the same
``client_order_id`` so reconciliation has an idempotent join key.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit", "stop", "stop_limit"]
TimeInForce = Literal["day", "gtc", "opg", "cls", "ioc", "fok"]
Environment = Literal["backtest", "shadow", "paper", "live"]
OrderStatus = Literal[
    "PROPOSED",
    "RISK_APPROVED",
    "SUBMITTED",
    "ACCEPTED",
    "PARTIALLY_FILLED",
    "FILLED",
    "CANCELED",
    "REJECTED",
    "EXPIRED",
    "RECONCILED",
]


@dataclass(frozen=True, slots=True)
class OrderIntent:
    intent_id: str
    as_of: datetime
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType
    time_in_force: TimeInForce
    limit_price: float | None
    source_decision_id: str          # links back to RiskDecision.decision_id
    environment: Environment
    client_order_id: str             # idempotency key sent to the broker


@dataclass(frozen=True, slots=True)
class BrokerOrder:
    """Broker-side view of an order. Updated by polling/streaming."""

    internal_order_id: str
    broker_order_id: str | None
    client_order_id: str
    environment: Environment
    symbol: str
    side: Side
    order_type: OrderType
    time_in_force: TimeInForce
    requested_qty: float
    limit_price: float | None
    status: OrderStatus
    submitted_at: datetime
    updated_at: datetime
    raw_request: dict = field(default_factory=dict)
    raw_response: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FillEvent:
    fill_id: str
    internal_order_id: str
    broker_order_id: str | None
    symbol: str
    side: Side
    filled_qty: float
    filled_avg_price: float
    filled_at: datetime
    raw_fill: dict = field(default_factory=dict)
