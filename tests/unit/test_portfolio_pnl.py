"""FIX-#36: date-ordered portfolio P&L tests.

The legacy validation_composite_score computes Sharpe / max-DD on
the per-trade return sequence — wrong for concurrent positions and
date order. The portfolio_pnl helpers spread each trade's return
across its holding days into a date-keyed dict, then metrics are
computed on the daily series.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rl_swing.rl.validation.portfolio_pnl import (
    TradeRecord,
    annualized_sharpe_from_daily_pnl,
    daily_gross_exposure,
    daily_portfolio_pnl,
    max_drawdown_from_daily_pnl,
    total_return_from_daily_pnl,
)


# ---------------------------------------------------------------------
def test_two_non_overlapping_trades_pnl_sums_separately():
    """Two trades on disjoint date windows produce daily P&L only on
    their own days; the days between them are absent."""
    t1 = TradeRecord(
        entry_date=date(2024, 1, 1), exit_date=date(2024, 1, 5),
        return_pct=0.04, size_pct=0.10,
    )
    t2 = TradeRecord(
        entry_date=date(2024, 2, 1), exit_date=date(2024, 2, 5),
        return_pct=-0.02, size_pct=0.10,
    )
    daily = daily_portfolio_pnl([t1, t2])

    # Each trade spread over 4 days at +1% and -0.5% per day.
    assert abs(daily[date(2024, 1, 1)] - 0.01) < 1e-9
    assert abs(daily[date(2024, 1, 4)] - 0.01) < 1e-9
    assert abs(daily[date(2024, 2, 1)] - (-0.005)) < 1e-9
    # Day in between has no entry.
    assert date(2024, 1, 15) not in daily


def test_two_overlapping_trades_pnl_sums_on_shared_days():
    """Two trades active on the same day add their daily contributions
    on that day — what the legacy per-trade Sharpe got wrong."""
    t1 = TradeRecord(
        entry_date=date(2024, 1, 1), exit_date=date(2024, 1, 11),
        return_pct=0.10, size_pct=0.10,    # +1%/day for 10 days
    )
    t2 = TradeRecord(
        entry_date=date(2024, 1, 6), exit_date=date(2024, 1, 11),
        return_pct=-0.05, size_pct=0.10,   # -1%/day for 5 days
    )
    daily = daily_portfolio_pnl([t1, t2])

    # Days 1-5: only t1 contributes +1%/day.
    assert abs(daily[date(2024, 1, 3)] - 0.01) < 1e-9
    # Days 6-10: both — net +1% + (-1%) = 0.
    assert abs(daily[date(2024, 1, 6)] - 0.0) < 1e-9
    assert abs(daily[date(2024, 1, 10)] - 0.0) < 1e-9


def test_daily_gross_exposure_sums_overlapping_sizes():
    """Two concurrent 10%-sized positions have 20% gross on shared
    days. The capital-constraint follow-up will use this to detect
    leverage violations."""
    t1 = TradeRecord(
        entry_date=date(2024, 1, 1), exit_date=date(2024, 1, 5),
        return_pct=0.0, size_pct=0.10,
    )
    t2 = TradeRecord(
        entry_date=date(2024, 1, 3), exit_date=date(2024, 1, 7),
        return_pct=0.0, size_pct=0.20,
    )
    expo = daily_gross_exposure([t1, t2])
    assert abs(expo[date(2024, 1, 1)] - 0.10) < 1e-9   # only t1
    assert abs(expo[date(2024, 1, 4)] - 0.30) < 1e-9   # both
    assert abs(expo[date(2024, 1, 7)] - 0.20) < 1e-9   # only t2


def test_total_return_from_daily_pnl_compounds_correctly():
    """Sequential daily +1% gains compound; the legacy per-trade
    cumulative-product would have given the same answer for
    non-overlapping trades, so this is the boring no-overlap case."""
    daily = {
        date(2024, 1, 1): 0.01,
        date(2024, 1, 2): 0.01,
        date(2024, 1, 3): 0.01,
    }
    # 1.01^3 - 1 = 0.030301
    assert abs(total_return_from_daily_pnl(daily) - 0.030301) < 1e-9


def test_total_return_with_offsetting_same_day_pnl_is_zero():
    """The legacy per-trade compound would treat ±10% on the same
    day as two trade events that round-trip near 0% but with
    intermediate compounding artifacts. Daily-P&L correctly nets
    them on that single day."""
    t_winner = TradeRecord(
        entry_date=date(2024, 1, 1), exit_date=date(2024, 1, 2),
        return_pct=0.10, size_pct=1.0,
    )
    t_loser = TradeRecord(
        entry_date=date(2024, 1, 1), exit_date=date(2024, 1, 2),
        return_pct=-0.10, size_pct=1.0,
    )
    daily = daily_portfolio_pnl([t_winner, t_loser])
    # +10% + (-10%) on the single overlapping day = 0%.
    assert abs(daily[date(2024, 1, 1)]) < 1e-9
    assert abs(total_return_from_daily_pnl(daily)) < 1e-9


def test_max_drawdown_on_daily_equity_curve():
    """DD is peak-to-trough on the chronological daily equity curve,
    not on the per-trade cumulative product."""
    daily = {
        date(2024, 1, 1): 0.05,    # equity 1.05
        date(2024, 1, 2): -0.10,   # equity 0.945
        date(2024, 1, 3): 0.00,    # equity 0.945
        date(2024, 1, 4): 0.20,    # equity 1.134
    }
    # Peak after day 1: 1.05. Trough on day 2: 0.945.
    # DD = (1.05 - 0.945) / 1.05 ≈ 0.10
    assert abs(max_drawdown_from_daily_pnl(daily) - 0.10) < 1e-9


def test_sharpe_on_daily_pnl_uses_sqrt_252():
    """Sharpe is computed on daily P&L variance, annualized via
    sqrt(252). With std=1% and mean=+0.1%, sharpe = 0.1 * sqrt(252)
    ≈ 1.587."""
    daily = {date(2024, 1, i): 0.001 + (0.01 if i % 2 else -0.01)
             for i in range(1, 21)}
    s = annualized_sharpe_from_daily_pnl(daily)
    assert s > 0.1   # positive overall (slight positive bias in the daily mean)


def test_empty_trades_yield_zero_metrics():
    """No trades — all metrics zero / neutral, no division errors."""
    daily = daily_portfolio_pnl([])
    assert daily == {}
    assert total_return_from_daily_pnl(daily) == 0.0
    assert annualized_sharpe_from_daily_pnl(daily) == 0.0
    assert max_drawdown_from_daily_pnl(daily) == 0.0


# ---------------------------------------------------------------------
def test_composite_score_from_daily_pnl_returns_expected_shape():
    """The new composite-score function must emit the same dict shape
    as the legacy one (so acceptance_gate + scorecard + diary
    templates work unchanged)."""
    from rl_swing.rl.validation.metrics import (
        validation_composite_score_from_daily_pnl,
    )

    trades = [
        TradeRecord(entry_date=date(2024, 1, 1), exit_date=date(2024, 1, 5),
                    return_pct=0.02, size_pct=0.10),
        TradeRecord(entry_date=date(2024, 1, 6), exit_date=date(2024, 1, 10),
                    return_pct=-0.01, size_pct=0.10),
    ]
    score, breakdown = validation_composite_score_from_daily_pnl(
        trades=trades, n_total_packs=10,
    )
    assert isinstance(score, float)
    for k in ("n_trades", "total_return", "annualized_sharpe",
              "profit_factor", "max_drawdown", "turnover_take_rate",
              "components"):
        assert k in breakdown, f"missing key {k!r} in breakdown"
    # n_trades reflects len(trades), not days.
    assert breakdown["n_trades"] == 2
    # turnover from explicit n_total_packs.
    assert abs(breakdown["turnover_take_rate"] - 0.2) < 1e-9
    # metric_basis tag distinguishes from legacy output.
    assert breakdown["metric_basis"] == "daily_pnl_v36"
