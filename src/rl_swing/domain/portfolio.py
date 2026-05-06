"""Portfolio-state domain types.

``PortfolioState`` is the read-model the policy and risk layers see.
``PositionSnapshot`` and ``AccountSnapshot`` are the persisted views
that get reconciled against broker state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

PositionSource = Literal["simulated", "alpaca", "internal"]


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    as_of: datetime
    source: PositionSource
    symbol: str
    quantity: float
    market_value: float
    avg_entry_price: float | None = None
    unrealized_pnl: float | None = None
    days_held: int = 0


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    as_of: datetime
    source: PositionSource
    cash: float
    buying_power: float
    equity: float
    portfolio_value: float
    margin_used: float = 0.0
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """Snapshot of internal portfolio state passed to scorers/risk rules.

    Kept narrow on purpose — anything more is a feature, not state.
    """

    as_of: datetime
    cash: float
    equity: float
    positions: tuple[PositionSnapshot, ...] = ()
    realized_pnl_20d: float = 0.0
    current_drawdown_pct: float = 0.0
    daily_loss_pct: float = 0.0
    open_positions_count: int = 0

    @property
    def gross_exposure_pct(self) -> float:
        if self.equity <= 0:
            return 0.0
        gross = sum(abs(p.market_value) for p in self.positions)
        return gross / self.equity

    def position_for(self, symbol: str) -> PositionSnapshot | None:
        for p in self.positions:
            if p.symbol == symbol:
                return p
        return None
