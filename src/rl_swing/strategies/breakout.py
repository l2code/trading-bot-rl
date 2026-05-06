"""BreakoutStrategy.

Long-only N-day high breakout. Fires when:
    - Today's close is at or above the 20-day high (distance == 0),
    - Volume relatively higher than the 20-day average (>=1.0),
    - SPY is in a non-bear regime (SPY above SMA50).
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from rl_swing.domain import CandidateTrade, FeatureFrame, PortfolioState


@dataclass
class BreakoutStrategy:
    strategy_id: str = "breakout_20d"
    feature_dependencies: tuple[str, ...] = (
        "distance_from_20d_high", "relative_volume_20", "spy_above_sma_50"
    )
    breakout_lookback: int = 20
    min_relative_volume: float = 1.0
    max_holding_days: int = 10
    base_size_pct: float = 0.07

    def generate(
        self,
        features: Iterable[FeatureFrame],
        portfolio_state: PortfolioState,
    ) -> Iterable[CandidateTrade]:
        for frame in features:
            v = frame.values
            dist_high = v.get("distance_from_20d_high", -1.0)
            rel_vol = v.get("relative_volume_20", 0.0)
            spy_ok = v.get("spy_above_sma_50", 0.0) >= 0.5

            # Within 0.2% of the 20-day high — counts as a breakout.
            if dist_high < -0.002:
                continue
            if rel_vol < self.min_relative_volume:
                continue
            if not spy_ok:
                continue

            score = _logistic(rel_vol * 0.5 + max(0.0, dist_high) * 50.0)

            yield CandidateTrade(
                candidate_id=_cid(frame, self.strategy_id),
                as_of=frame.as_of,
                symbol=frame.symbol,
                strategy_id=self.strategy_id,
                direction="long",
                entry_timing="next_open",
                base_size_pct=float(self.base_size_pct),
                max_holding_days=int(self.max_holding_days),
                stop_rule_id="below_breakout",
                exit_rule_id="time_or_target",
                signal_strength=float(score),
                metadata={
                    "distance_from_20d_high": dist_high,
                    "relative_volume_20": rel_vol,
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
