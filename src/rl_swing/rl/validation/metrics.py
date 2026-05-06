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
        "components": {
            "n_total_return": n_total,
            "n_sharpe": n_sharpe,
            "n_profit_factor": n_pf,
            "n_max_drawdown": n_mdd,
            "n_turnover": n_turn,
        },
    }
