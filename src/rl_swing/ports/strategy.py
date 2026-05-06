"""CandidateStrategy port."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from rl_swing.domain import CandidateTrade, FeatureFrame, PortfolioState


@runtime_checkable
class CandidateStrategy(Protocol):
    strategy_id: str
    feature_dependencies: tuple[str, ...]

    def generate(
        self,
        features: Iterable[FeatureFrame],
        portfolio_state: PortfolioState,
    ) -> Iterable[CandidateTrade]: ...
