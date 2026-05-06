"""Buy-and-hold baselines.

These are NOT ``PolicyScorer``s — they're a different unit of analysis
(per-symbol passive returns, not per-candidate decisions). The
walk-forward harness shows them alongside the policy comparisons so a
reader can sanity-check whether the strategies/RL were even worth it
versus just holding SPY.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import date

from rl_swing.domain import MarketBar


def buy_and_hold_return(
    bars: Iterable[MarketBar],
    symbol: str,
    start: date,
    end: date,
) -> float:
    """Total return from holding ``symbol`` between ``start`` and ``end``."""
    sym_bars = [b for b in bars if b.symbol == symbol and start <= b.timestamp.date() <= end]
    sym_bars.sort(key=lambda b: b.timestamp)
    if len(sym_bars) < 2:
        return 0.0
    p0 = sym_bars[0].adjusted_close or sym_bars[0].close
    p1 = sym_bars[-1].adjusted_close or sym_bars[-1].close
    if p0 <= 0:
        return 0.0
    return float(p1 / p0 - 1.0)
