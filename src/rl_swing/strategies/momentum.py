"""MomentumStrategy.

Long-only swing momentum candidate generator. Fires when:
    - 20d return > 0,
    - close above SMA50 and SMA200,
    - relative strength vs SPY >= ``min_relative_strength``.

The exact thresholds are config-driven so the same strategy class can
power both 20/60 and 50/200 variants.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from rl_swing.domain import CandidateTrade, FeatureFrame, PortfolioState


@dataclass
class MomentumStrategy:
    strategy_id: str = "momentum_20_60"
    feature_dependencies: tuple[str, ...] = (
        "return_20d", "close_vs_sma_50", "close_vs_sma_200", "spy_return_20d"
    )
    lookback_short: int = 20
    lookback_long: int = 60
    min_relative_strength: float = 0.0
    max_holding_days: int = 10
    base_size_pct: float = 0.10
    # Lower bound on 20-day return. Defaults to >0 (only positive
    # momentum). Set to a small negative number to admit weak/marginal
    # candidates that the RL filter then has to discriminate.
    min_r20: float = 0.0
    # Whether to require close > SMA200. Default True keeps legacy
    # behavior (only longer-term uptrending names). Disable when you
    # want to widen the candidate funnel for the RL filter.
    require_sma200_above: bool = True

    def generate(
        self,
        features: Iterable[FeatureFrame],
        portfolio_state: PortfolioState,
    ) -> Iterable[CandidateTrade]:
        for frame in features:
            v = frame.values
            r20 = v.get("return_20d", 0.0)
            close_vs_sma50 = v.get("close_vs_sma_50", 0.0)
            close_vs_sma200 = v.get("close_vs_sma_200", 0.0)
            spy_r20 = v.get("spy_return_20d", 0.0)

            rs = r20 - spy_r20
            if r20 <= self.min_r20:
                continue
            if close_vs_sma50 <= 0:
                continue
            if self.require_sma200_above and close_vs_sma200 <= 0:
                continue
            if rs < self.min_relative_strength:
                continue

            score = _logistic(rs * 5.0 + r20 * 3.0)

            yield CandidateTrade(
                candidate_id=_cid(frame, self.strategy_id),
                as_of=frame.as_of,
                symbol=frame.symbol,
                strategy_id=self.strategy_id,
                direction="long",
                entry_timing="next_open",
                base_size_pct=float(self.base_size_pct),
                max_holding_days=int(self.max_holding_days),
                stop_rule_id="atr_2x",
                exit_rule_id="time_or_target",
                signal_strength=float(score),
                metadata={
                    "return_20d": r20,
                    "rel_strength_vs_spy": rs,
                    "feature_version": frame.feature_version,
                    "avg_dollar_volume": v.get("dollar_volume", -1.0),
                },
            )


def _logistic(x: float) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-x))


def _cid(frame: FeatureFrame, strategy_id: str) -> str:
    s = f"{frame.as_of.date().isoformat()}_{frame.symbol}_{strategy_id}_{frame.feature_version}"
    return hashlib.sha1(s.encode()).hexdigest()[:16] + f"_{frame.symbol}"
