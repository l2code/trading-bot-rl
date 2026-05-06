"""StrategyAggregator — runs many strategies and dedupes overlapping
candidates (same symbol, same date) by keeping the higher-strength one.

Used in places where the env should not see two candidates for the
same symbol-day; the validation harness can also use the unaggregated
list to attribute outcomes back to the originating strategy.
"""
from __future__ import annotations

from collections.abc import Iterable

from rl_swing.domain import CandidateTrade, FeatureFrame, PortfolioState
from rl_swing.ports import CandidateStrategy


class StrategyAggregator:
    strategy_id: str = "aggregator"
    feature_dependencies: tuple[str, ...] = ()

    def __init__(self, strategies: list[CandidateStrategy]) -> None:
        self.strategies = strategies

    def generate(
        self,
        features: Iterable[FeatureFrame],
        portfolio_state: PortfolioState,
    ) -> Iterable[CandidateTrade]:
        # Materialize so each strategy can iterate independently.
        frames = list(features)
        all_candidates: list[CandidateTrade] = []
        for s in self.strategies:
            all_candidates.extend(s.generate(frames, portfolio_state))
        # Group by (symbol, date) and keep the highest signal_strength.
        keyed: dict[tuple[str, str], CandidateTrade] = {}
        for c in all_candidates:
            key = (c.symbol, c.as_of.date().isoformat())
            existing = keyed.get(key)
            if existing is None or c.signal_strength > existing.signal_strength:
                keyed[key] = c
        return sorted(keyed.values(), key=lambda c: (c.as_of, c.symbol))
