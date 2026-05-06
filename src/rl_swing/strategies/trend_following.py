"""TrendFollowingStrategy.

Long-only longer-term trend pullback entry. Fires when:
    - Close is above SMA50 and SMA200,
    - SMA50 above SMA200 (proxied by close_vs_sma_200 - close_vs_sma_50 > 0),
    - Recent pullback resolved upward (return_5d > 0),
    - Volatility not extreme (atr_pct_14 < 0.06).
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from rl_swing.domain import CandidateTrade, FeatureFrame, PortfolioState


@dataclass
class TrendFollowingStrategy:
    strategy_id: str = "trend_following"
    feature_dependencies: tuple[str, ...] = (
        "close_vs_sma_50", "close_vs_sma_200", "return_5d", "atr_pct_14"
    )
    sma_short: int = 50
    sma_long: int = 200
    max_holding_days: int = 20
    base_size_pct: float = 0.07
    max_atr_pct: float = 0.06

    def generate(
        self,
        features: Iterable[FeatureFrame],
        portfolio_state: PortfolioState,
    ) -> Iterable[CandidateTrade]:
        for frame in features:
            v = frame.values
            close_vs_50 = v.get("close_vs_sma_50", 0.0)
            close_vs_200 = v.get("close_vs_sma_200", 0.0)
            r5 = v.get("return_5d", 0.0)
            atr_pct = v.get("atr_pct_14", 1.0)

            if close_vs_50 <= 0 or close_vs_200 <= 0:
                continue
            if r5 <= 0:
                continue
            if atr_pct >= self.max_atr_pct:
                continue

            score = _logistic(r5 * 8.0 + close_vs_200 * 3.0)

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
                exit_rule_id="trend_break_or_time",
                signal_strength=float(score),
                metadata={
                    "return_5d": r5,
                    "close_vs_sma_50": close_vs_50,
                    "close_vs_sma_200": close_vs_200,
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
