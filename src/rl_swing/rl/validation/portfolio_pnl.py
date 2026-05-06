"""Date-ordered daily-portfolio-P&L helpers (FIX-#36).

The legacy ``validation_composite_score`` in ``metrics.py`` computes
Sharpe / max-drawdown on the per-trade return sequence — i.e., it
treats each trade as if it were a separate trading day. That ignores:

- **Concurrent positions:** five trades opening the same day each
  see ``starting_equity`` as their notional base. The synthesized
  per-trade equity curve compounds them sequentially, but in
  reality their P&L is realized on the same set of days.
- **Date order:** two trades earning ±10% on the same day net to
  zero in real life; the legacy curve treats them as two days of
  opposite returns.
- **Compounding semantics:** the legacy DD is on a curve indexed by
  trade number, not by calendar date.

This module provides the date-ordered alternative. ``TradeRecord``
captures the minimum a trade must report to spread its contribution
across its holding period. ``daily_portfolio_pnl`` spreads each
trade's return_pct uniformly across its holding days. The resulting
daily P&L series is what real Sharpe / max-DD should be computed on.

This module deliberately does NOT enforce capital constraints
(gross-exposure caps, available-cash checks). Those are tracked as
a separate follow-up — adding them changes the simulator's
acceptance semantics and warrants its own design discussion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """One executed trade summarized for portfolio-P&L purposes.

    ``return_pct`` is the **portfolio contribution** — already
    scaled by ``size_pct`` and net of costs (per FIX-22 + FIX-23).
    A trade with size_pct=0.10 and a +10% asset move yields
    return_pct ≈ +0.01 (1% portfolio contribution).
    """
    entry_date: date
    exit_date: date
    return_pct: float
    size_pct: float


def daily_portfolio_pnl(
    trades: list[TradeRecord],
    *,
    window_start: date | None = None,
    window_end: date | None = None,
    trading_days: list[date] | None = None,
) -> dict[date, float]:
    """Spread each trade's portfolio return_pct across its holding
    period. Returns a dict keyed by date; values are *additive*
    daily contributions to portfolio equity.

    Two trades on the same day add their contributions on that day.

    Spreading semantics (FIX-#57):
    - If ``trading_days`` is provided, returns are spread across the
      sequence of trading days from entry_date to exit_date,
      INCLUSIVE — Saturdays, Sundays, and exchange holidays are
      skipped. Annualization with ``sqrt(252)`` is then consistent.
    - If ``trading_days`` is omitted, falls back to calendar-day
      spreading (legacy behavior — applies P&L to weekend dates
      and over-divides by calendar count). Use only when a trading
      calendar isn't available.

    Idle-day fill (FIX-#52 + #56):
    - If ``trading_days`` provided: every trading day in the list
      gets a zero entry, then trade contributions are added.
    - Else if ``window_start``+``window_end`` provided: every
      weekday (Mon-Fri) in [window_start, window_end] gets a zero
      entry. Approximate but better than nothing.
    - Else: only active days appear in the result.

    The window-or-calendar fill is critical for correct Sharpe /
    max-DD on policies that are flat for parts of the window.
    """
    daily: dict[date, float] = {}

    # Build the lookup set used for both idle-day fill and trade
    # spreading.
    if trading_days is not None:
        td_sorted = sorted(set(trading_days))
        for d in td_sorted:
            daily[d] = 0.0
    elif window_start is not None and window_end is not None:
        d = window_start
        while d <= window_end:
            if d.weekday() < 5:    # Mon-Fri only — approximate
                daily[d] = 0.0
            d = d + timedelta(days=1)
        td_sorted = sorted(daily.keys())
    else:
        td_sorted = None

    for t in trades:
        if td_sorted is not None:
            # Spread on TRADING days from entry to exit, inclusive.
            from bisect import bisect_left, bisect_right
            lo = bisect_left(td_sorted, t.entry_date)
            hi = bisect_right(td_sorted, t.exit_date)
            window_days = td_sorted[lo:hi]
            if not window_days:
                # Trade fell outside known calendar — skip silently.
                # (Could happen if entry/exit predate window_start.)
                continue
            per_day = t.return_pct / len(window_days)
            for d in window_days:
                daily[d] = daily.get(d, 0.0) + per_day
        else:
            # Legacy calendar-day fallback.
            n_days = max(1, (t.exit_date - t.entry_date).days)
            per_day = t.return_pct / n_days
            d = t.entry_date
            for _ in range(n_days):
                daily[d] = daily.get(d, 0.0) + per_day
                d = d + timedelta(days=1)
    return daily


def daily_gross_exposure(
    trades: list[TradeRecord],
) -> dict[date, float]:
    """Sum ``size_pct`` across all trades open on each day. Useful
    for the capital-constraint follow-up; the daily-P&L computation
    itself doesn't need this."""
    expo: dict[date, float] = {}
    for t in trades:
        d = t.entry_date
        while d <= t.exit_date:
            expo[d] = expo.get(d, 0.0) + t.size_pct
            d = d + timedelta(days=1)
    return expo


# ---------------------------------------------------------------------
def total_return_from_daily_pnl(daily_pnl: dict[date, float]) -> float:
    """Compounded total return over the daily P&L series. Treats
    each daily entry as an additive P&L fraction (so daily equity
    changes as ``equity *= (1 + daily_pnl)``)."""
    if not daily_pnl:
        return 0.0
    eq = 1.0
    for d in sorted(daily_pnl.keys()):
        eq *= (1.0 + daily_pnl[d])
    return eq - 1.0


def annualized_sharpe_from_daily_pnl(daily_pnl: dict[date, float]) -> float:
    """Sharpe computed on daily P&L (not per-trade). Annualization
    uses sqrt(252) for trading days."""
    if len(daily_pnl) < 2:
        return 0.0
    values = [daily_pnl[d] for d in sorted(daily_pnl.keys())]
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / max(1, n - 1)
    std = math.sqrt(var)
    if std <= 0:
        return 0.0
    return (mean / std) * math.sqrt(252.0)


def max_drawdown_from_daily_pnl(daily_pnl: dict[date, float]) -> float:
    """Peak-to-trough drawdown on the daily equity curve."""
    if not daily_pnl:
        return 0.0
    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    for d in sorted(daily_pnl.keys()):
        eq *= (1.0 + daily_pnl[d])
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


def profit_factor_from_daily_pnl(daily_pnl: dict[date, float]) -> float:
    """Sum of positive daily P&L over absolute sum of negative
    daily P&L. Comparable to the per-trade profit factor but at the
    daily-portfolio level."""
    if not daily_pnl:
        return 1.0
    pos = sum(v for v in daily_pnl.values() if v > 0)
    neg = -sum(v for v in daily_pnl.values() if v < 0)
    if neg <= 0:
        return float(pos) if pos > 0 else 1.0
    return float(pos / neg)
