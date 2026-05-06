"""Validation metrics.

Spec §12.5.6 specifies the validation composite score. We expose the
raw breakdown so the validation report can show every component and
also rank models by individual metrics where helpful.

The composite formula (defaults from the spec):

    score =
        0.35 * normalized_total_return
      + 0.25 * normalized_sharpe_or_sortino
      + 0.20 * normalized_profit_factor
      - 0.15 * normalized_max_drawdown
      - 0.05 * normalized_turnover
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class WeightConfig:
    return_weight: float = 0.35
    sharpe_weight: float = 0.25
    profit_factor_weight: float = 0.20
    drawdown_weight: float = 0.15
    turnover_weight: float = 0.05


def _normalize_clipped(x: float, lo: float, hi: float) -> float:
    """Normalize x into [0, 1] given calibration anchors lo/hi."""
    if hi <= lo:
        return 0.5
    z = (x - lo) / (hi - lo)
    return max(0.0, min(1.0, z))


def total_return(net_returns: list[float]) -> float:
    if not net_returns:
        return 0.0
    return float(np.prod(1.0 + np.array(net_returns)) - 1.0)


def annualized_sharpe(net_returns: list[float], holding_days: list[int]) -> float:
    if not net_returns:
        return 0.0
    arr = np.asarray(net_returns)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    if std <= 0:
        return 0.0
    avg_holding = float(np.mean(holding_days)) if holding_days else 1.0
    trades_per_year = max(1.0, 252.0 / max(avg_holding, 1.0))
    return mean / std * math.sqrt(trades_per_year)


def profit_factor(net_returns: list[float]) -> float:
    if not net_returns:
        return 1.0
    arr = np.asarray(net_returns)
    gains = arr[arr > 0].sum()
    losses = -arr[arr < 0].sum()
    if losses <= 0:
        return float(gains) if gains > 0 else 1.0
    return float(gains / losses)


def max_drawdown_from_returns(net_returns: list[float]) -> float:
    if not net_returns:
        return 0.0
    eq = np.cumprod(1.0 + np.asarray(net_returns))
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(-dd.min()) if len(dd) else 0.0


def turnover_metric(actions: list[str]) -> float:
    """Crude turnover proxy — fraction of trades taken."""
    if not actions:
        return 0.0
    takes = sum(1 for a in actions if a == "take")
    return float(takes / len(actions))


def validation_composite_score(
    *,
    net_returns: list[float],
    cost_bps: list[float] | None = None,
    holding_days: list[int] | None = None,
    rewards: list[float] | None = None,
    actions: list[str] | None = None,
    weights: WeightConfig | None = None,
) -> tuple[float, dict]:
    """Per-trade composite score (legacy, pre-FIX-#36).

    Computes Sharpe / max-DD on the per-trade return sequence — i.e.,
    treats each trade as if it were a separate day. This silently
    handles concurrent positions and date order incorrectly. New
    callers should prefer ``validation_composite_score_from_daily_pnl``;
    this function is kept for reproducing pre-FIX-#36 results.
    """
    weights = weights or WeightConfig()

    cum_ret = total_return(net_returns)
    sharpe = annualized_sharpe(net_returns, holding_days or [])
    pf = profit_factor(net_returns)
    mdd = max_drawdown_from_returns(net_returns)
    turn = turnover_metric(actions or [])

    # Normalize via reasonable swing-trading anchors.
    n_total = _normalize_clipped(cum_ret, lo=-0.5, hi=0.5)        # -50%..+50%
    n_sharpe = _normalize_clipped(sharpe, lo=-1.0, hi=2.0)         # 0..2 sharpe
    n_pf = _normalize_clipped(pf, lo=0.5, hi=2.0)                  # 0.5..2
    n_mdd = _normalize_clipped(mdd, lo=0.0, hi=0.5)                # 0..50% dd
    n_turn = _normalize_clipped(turn, lo=0.0, hi=1.0)

    score = (
        weights.return_weight * n_total
        + weights.sharpe_weight * n_sharpe
        + weights.profit_factor_weight * n_pf
        - weights.drawdown_weight * n_mdd
        - weights.turnover_weight * n_turn
    )

    return float(score), {
        "n_trades": len(net_returns),
        "total_return": cum_ret,
        "annualized_sharpe": sharpe,
        "profit_factor": pf,
        "max_drawdown": mdd,
        "turnover_take_rate": turn,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "metric_basis": "per_trade_legacy",
        "components": {
            "n_total_return": n_total,
            "n_sharpe": n_sharpe,
            "n_profit_factor": n_pf,
            "n_max_drawdown": n_mdd,
            "n_turnover": n_turn,
        },
    }


def validation_composite_score_from_daily_pnl(
    *,
    trades,                            # list[TradeRecord]
    n_total_packs: int = 0,
    rewards: list[float] | None = None,
    actions: list[str] | None = None,
    weights: WeightConfig | None = None,
    window_start=None,                 # date | None — for FIX-#52 idle-day fill
    window_end=None,                   # date | None
    trading_days=None,                 # list[date] | None — FIX-#57
) -> tuple[float, dict]:
    """FIX-#36 — date-ordered portfolio metrics.

    Computes Sharpe / max-DD on a daily P&L series spread from the
    given trades, NOT on the per-trade return sequence. Properly
    handles concurrent positions (their daily contributions add)
    and date order (two ±10% trades on the same day net to zero).

    Returns the same composite_score / breakdown shape as the legacy
    ``validation_composite_score`` so the acceptance gate, scorecard,
    and diary template work unchanged.

    ``trades`` must be a list of ``TradeRecord`` (from
    ``portfolio_pnl``). ``n_total_packs`` is used for the turnover
    metric (``len(trades) / n_total_packs``); pass 0 to fall back to
    the legacy ``actions``-derived turnover.
    """
    from rl_swing.rl.validation.portfolio_pnl import (
        annualized_sharpe_from_daily_pnl,
        daily_portfolio_pnl,
        max_drawdown_from_daily_pnl,
        profit_factor_from_daily_pnl,
        total_return_from_daily_pnl,
    )

    weights = weights or WeightConfig()
    daily_pnl = daily_portfolio_pnl(
        trades,
        window_start=window_start,
        window_end=window_end,
        trading_days=trading_days,
    )

    cum_ret = total_return_from_daily_pnl(daily_pnl)
    sharpe = annualized_sharpe_from_daily_pnl(daily_pnl)
    pf = profit_factor_from_daily_pnl(daily_pnl)
    mdd = max_drawdown_from_daily_pnl(daily_pnl)
    if n_total_packs > 0:
        turn = float(len(trades) / max(1, n_total_packs))
    else:
        turn = turnover_metric(actions or [])

    n_total = _normalize_clipped(cum_ret, lo=-0.5, hi=0.5)
    n_sharpe = _normalize_clipped(sharpe, lo=-1.0, hi=2.0)
    n_pf = _normalize_clipped(pf, lo=0.5, hi=2.0)
    n_mdd = _normalize_clipped(mdd, lo=0.0, hi=0.5)
    n_turn = _normalize_clipped(turn, lo=0.0, hi=1.0)

    score = (
        weights.return_weight * n_total
        + weights.sharpe_weight * n_sharpe
        + weights.profit_factor_weight * n_pf
        - weights.drawdown_weight * n_mdd
        - weights.turnover_weight * n_turn
    )

    return float(score), {
        "n_trades": len(trades),
        "n_trading_days": len(daily_pnl),
        "total_return": cum_ret,
        "annualized_sharpe": sharpe,
        "profit_factor": pf,
        "max_drawdown": mdd,
        "turnover_take_rate": turn,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "metric_basis": "daily_pnl_v36",
        "components": {
            "n_total_return": n_total,
            "n_sharpe": n_sharpe,
            "n_profit_factor": n_pf,
            "n_max_drawdown": n_mdd,
            "n_turnover": n_turn,
        },
    }
