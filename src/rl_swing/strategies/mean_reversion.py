"""RsiMeanReversionStrategy.

Long-only short-term mean reversion. Fires when:
    - RSI(rsi_window) below ``rsi_threshold``,
    - longer-term trend still intact (close above SMA50),
    - market regime not hostile (SPY above SMA200).
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from rl_swing.domain import CandidateTrade, FeatureFrame, PortfolioState


@dataclass
class RsiMeanReversionStrategy:
    strategy_id: str = "mean_reversion_rsi"
    feature_dependencies: tuple[str, ...] = (
        "rsi_5", "rsi_2", "close_vs_sma_50", "spy_above_sma_200"
    )
    rsi_window: int = 5
    rsi_threshold: float = 25.0
    require_uptrend: bool = True
    max_holding_days: int = 5
    base_size_pct: float = 0.07

    def generate(
        self,
        features: Iterable[FeatureFrame],
        portfolio_state: PortfolioState,
    ) -> Iterable[CandidateTrade]:
        rsi_field = f"rsi_{self.rsi_window}" if self.rsi_window in (2, 5, 14) else "rsi_5"
        for frame in features:
            v = frame.values
            rsi = v.get(rsi_field, 50.0)
            close_vs_sma50 = v.get("close_vs_sma_50", 0.0)
            spy_ok = v.get("spy_above_sma_200", 0.0) >= 0.5

            if rsi >= self.rsi_threshold:
                continue
            if self.require_uptrend and close_vs_sma50 <= -0.02:
                continue
            if not spy_ok:
                continue

            # Score = how oversold * how strong the structural trend is.
            score = _logistic((self.rsi_threshold - rsi) / 10.0 + close_vs_sma50 * 5.0)

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
                exit_rule_id="rsi_normalize_or_time",
                signal_strength=float(score),
                metadata={
                    rsi_field: rsi,
                    "close_vs_sma_50": close_vs_sma50,
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
