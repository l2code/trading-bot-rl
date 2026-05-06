"""SimulatedBrokerAdapter — fills orders against a price oracle.

Used in research/backtest mode. The price oracle is a callable
``(symbol, timestamp) -> float`` that the surrounding ExecutionSimulator
sets via ``set_price_oracle``. Without an oracle the broker cannot fill
orders and returns ``REJECTED``.

This adapter intentionally does NOT model spread/slippage — those live
in ``rl_swing.rl.env.cost_model.EquityExecutionModel`` so the env can
swap cost models without re-wiring the broker.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from datetime import datetime

from rl_swing.domain import (
    AccountSnapshot,
    BrokerOrder,
    FillEvent,
    OrderIntent,
    PositionSnapshot,
)

_log = logging.getLogger(__name__)

PriceOracle = Callable[[str, datetime], float | None]


class SimulatedBrokerAdapter:
    broker_id: str = "simulated"
    environment = "backtest"

    def __init__(self, starting_cash: float = 100_000.0) -> None:
        self._starting_cash = float(starting_cash)
        self._cash: float = float(starting_cash)
        self._positions: dict[str, PositionSnapshot] = {}
        self._orders: dict[str, BrokerOrder] = {}
        self._fills: list[FillEvent] = []
        self._oracle: PriceOracle | None = None

    # -- wiring --------------------------------------------------------
    def set_price_oracle(self, oracle: PriceOracle) -> None:
        self._oracle = oracle

    @property
    def fills(self) -> list[FillEvent]:
        return list(self._fills)

    # -- BrokerAdapter contract ---------------------------------------
    def submit_order(self, intent: OrderIntent) -> BrokerOrder:
        now = intent.as_of
        broker_id = f"sim-{uuid.uuid4().hex[:8]}"
        order = BrokerOrder(
            internal_order_id=intent.intent_id,
            broker_order_id=broker_id,
            client_order_id=intent.client_order_id,
            environment="backtest",
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            time_in_force=intent.time_in_force,
            requested_qty=intent.quantity,
            limit_price=intent.limit_price,
            status="SUBMITTED",
            submitted_at=now,
            updated_at=now,
            raw_request={"intent_id": intent.intent_id},
            raw_response={},
        )

        if self._oracle is None:
            order = self._update_status(order, "REJECTED", reason="no_price_oracle")
            self._orders[broker_id] = order
            return order

        price = self._oracle(intent.symbol, intent.as_of)
        if price is None or price <= 0:
            order = self._update_status(order, "REJECTED", reason="no_price")
            self._orders[broker_id] = order
            return order

        # Cash check (long-only enforcement)
        notional = price * intent.quantity
        if intent.side == "buy" and self._cash < notional:
            order = self._update_status(order, "REJECTED", reason="insufficient_cash")
            self._orders[broker_id] = order
            return order

        # Fill at the oracle price.
        fill = FillEvent(
            fill_id=f"fill-{uuid.uuid4().hex[:8]}",
            internal_order_id=order.internal_order_id,
            broker_order_id=broker_id,
            symbol=intent.symbol,
            side=intent.side,
            filled_qty=intent.quantity,
            filled_avg_price=price,
            filled_at=now,
            raw_fill={"oracle": True},
        )
        self._fills.append(fill)
        self._apply_fill(fill)

        order = self._update_status(order, "FILLED")
        self._orders[broker_id] = order
        return order

    def cancel_order(self, broker_order_id: str) -> None:
        order = self._orders.get(broker_order_id)
        if order and order.status in {"SUBMITTED", "PARTIALLY_FILLED", "ACCEPTED"}:
            self._orders[broker_order_id] = self._update_status(order, "CANCELED")

    def list_open_orders(self) -> list[BrokerOrder]:
        return [
            o for o in self._orders.values()
            if o.status in {"SUBMITTED", "PARTIALLY_FILLED", "ACCEPTED"}
        ]

    def list_positions(self) -> list[PositionSnapshot]:
        return list(self._positions.values())

    def get_account_snapshot(self) -> AccountSnapshot:
        market_value = sum(p.market_value for p in self._positions.values())
        equity = self._cash + market_value
        return AccountSnapshot(
            as_of=datetime.utcnow(),
            source="simulated",
            cash=self._cash,
            buying_power=self._cash,
            equity=equity,
            portfolio_value=equity,
            raw={"starting_cash": self._starting_cash},
        )

    # -- helpers -------------------------------------------------------
    def _update_status(
        self, order: BrokerOrder, status: str, reason: str | None = None
    ) -> BrokerOrder:
        from dataclasses import replace
        return replace(
            order,
            status=status,  # type: ignore[arg-type]
            updated_at=datetime.utcnow(),
            raw_response={**order.raw_response, "reason": reason} if reason else order.raw_response,
        )

    def _apply_fill(self, fill: FillEvent) -> None:
        notional = fill.filled_avg_price * fill.filled_qty
        if fill.side == "buy":
            self._cash -= notional
            existing = self._positions.get(fill.symbol)
            if existing is None:
                self._positions[fill.symbol] = PositionSnapshot(
                    as_of=fill.filled_at, source="simulated", symbol=fill.symbol,
                    quantity=fill.filled_qty,
                    market_value=notional,
                    avg_entry_price=fill.filled_avg_price,
                    unrealized_pnl=0.0,
                )
            else:
                new_qty = existing.quantity + fill.filled_qty
                new_avg = (
                    (existing.avg_entry_price or 0) * existing.quantity
                    + fill.filled_avg_price * fill.filled_qty
                ) / max(new_qty, 1e-9)
                self._positions[fill.symbol] = PositionSnapshot(
                    as_of=fill.filled_at, source="simulated", symbol=fill.symbol,
                    quantity=new_qty,
                    market_value=new_avg * new_qty,
                    avg_entry_price=new_avg,
                    unrealized_pnl=0.0,
                )
        else:  # sell
            self._cash += notional
            existing = self._positions.get(fill.symbol)
            if existing is None:
                # Long-only MVP: reject implicit shorts at the broker layer.
                _log.warning("simulated broker received sell for non-position %s", fill.symbol)
                return
            new_qty = existing.quantity - fill.filled_qty
            if new_qty <= 1e-9:
                self._positions.pop(fill.symbol, None)
            else:
                self._positions[fill.symbol] = PositionSnapshot(
                    as_of=fill.filled_at, source="simulated", symbol=fill.symbol,
                    quantity=new_qty,
                    market_value=(existing.avg_entry_price or 0) * new_qty,
                    avg_entry_price=existing.avg_entry_price,
                    unrealized_pnl=0.0,
                )
