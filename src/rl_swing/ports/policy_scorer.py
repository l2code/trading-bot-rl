"""PolicyScorer port.

Adapters: random, always-take, never-take, rule, PPO, DQN, ensemble,
manual override. Baselines are policies, not separate backtest paths.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from rl_swing.domain import CandidateTrade, FeatureFrame, PolicyDecision, PortfolioState


@runtime_checkable
class PolicyScorer(Protocol):
    model_id: str

    def score(
        self,
        candidate: CandidateTrade,
        features: FeatureFrame,
        portfolio_state: PortfolioState,
    ) -> PolicyDecision: ...
