"""ExecutionSimulator.

Given a single approved trade (symbol, entry timestamp, size, holding
plan) and a history of bars, simulates the fill, the holding-period
path, and the exit. Returns a ``TradeOutcome`` with realized return,
holding days, and intra-trade peak drawdown.

The trade is filled at the next bar's open. Stops/targets are checked
against bar high/low (approximate path simulation per the spec). If
neither stop nor target hit by ``max_holding_days``, the trade exits
at the close of the final day at horizon.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from rl_swing.domain import MarketBar


@dataclass(frozen=True)
class TradeOutcome:
    symbol: str
    entry_timestamp: datetime
    exit_timestamp: datetime
    entry_price: float
    exit_price: float
    qty: float
    notional: float
    return_pct: float           # net of cost (passed in by env)
    raw_return_pct: float       # before cost
    holding_days: int
    peak_drawdown_pct: float
    exit_reason: str            # "stop" | "target" | "time" | "no_data"
    cost_bps: float


class ExecutionSimulator:
    def __init__(
        self,
        atr_stop_mult: float = 2.0,
        atr_target_mult: float = 4.0,
    ) -> None:
        self.atr_stop_mult = float(atr_stop_mult)
        self.atr_target_mult = float(atr_target_mult)

    def simulate(
        self,
        bars: Sequence[MarketBar],
        entry_index: int,
        size_pct: float,
        max_holding_days: int,
        cost_bps: float,
        atr_pct: float,
        starting_equity: float = 100_000.0,
    ) -> TradeOutcome | None:
        if size_pct <= 0:
            return None
        # Need at least one future bar to enter.
        if entry_index + 1 >= len(bars):
            return None

        entry_bar = bars[entry_index + 1]
        entry_price = entry_bar.open
        if entry_price <= 0:
            return None
        notional = starting_equity * size_pct
        qty = notional / entry_price

        stop_pct = self.atr_stop_mult * max(atr_pct, 1e-4)
        target_pct = self.atr_target_mult * max(atr_pct, 1e-4)
        stop_price = entry_price * (1.0 - stop_pct)
        target_price = entry_price * (1.0 + target_pct)

        peak_dd = 0.0
        last_close = entry_price
        exit_index = min(entry_index + 1 + max_holding_days, len(bars) - 1)
        exit_reason = "time"
        exit_price = entry_price
        exit_ts = entry_bar.timestamp

        for i in range(entry_index + 1, exit_index + 1):
            bar = bars[i]
            # Track running drawdown using lows.
            this_dd = max(0.0, (entry_price - bar.low) / entry_price)
            peak_dd = max(peak_dd, this_dd)
            # Stop hit?
            if bar.low <= stop_price:
                exit_price = stop_price
                exit_reason = "stop"
                exit_ts = bar.timestamp
                last_close = exit_price
                break
            # Target hit?
            if bar.high >= target_price:
                exit_price = target_price
                exit_reason = "target"
                exit_ts = bar.timestamp
                last_close = exit_price
                break
            last_close = bar.close
            exit_ts = bar.timestamp
        else:
            # Reached the end without stop/target — exit at last close.
            exit_price = last_close

        if exit_reason == "time":
            exit_price = last_close

        raw_return = (exit_price - entry_price) / entry_price
        net_return = raw_return - cost_bps / 10_000.0
        holding_days = max(1, (exit_ts - entry_bar.timestamp).days or 1)

        return TradeOutcome(
            symbol=entry_bar.symbol,
            entry_timestamp=entry_bar.timestamp,
            exit_timestamp=exit_ts,
            entry_price=entry_price,
            exit_price=exit_price,
            qty=qty,
            notional=notional,
            return_pct=net_return,
            raw_return_pct=raw_return,
            holding_days=holding_days,
            peak_drawdown_pct=peak_dd,
            exit_reason=exit_reason,
            cost_bps=cost_bps,
        )
