"""ObservationBuilder.

Turns ``(CandidateTrade, FeatureFrame, PortfolioState)`` into the fixed
numeric observation vector the policy network sees. Order matches
``feature_pipeline.feature_names`` plus a small fixed-order suffix of
candidate / portfolio features.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from rl_swing.domain import CandidateTrade, FeatureFrame, PortfolioState


# Strategy id -> integer id used as a categorical-ish feature so the
# policy can condition on the source strategy. Keep this list stable;
# a new strategy goes at the end.
STRATEGY_INDEX = {
    "momentum_20_60": 0,
    "mean_reversion_rsi": 1,
    "breakout_20d": 2,
    "trend_following": 3,
    "volatility_contraction": 4,
    "unknown": 5,
}

# Candidate-side features tacked onto the end of the feature-frame
# vector. Order is contractual.
CANDIDATE_FEATURE_NAMES: tuple[str, ...] = (
    "cand_signal_strength",
    "cand_base_size_pct",
    "cand_max_holding_days_norm",
    "cand_strategy_index_norm",
    "portfolio_gross_exposure_pct",
    "portfolio_open_positions_count_norm",
    "portfolio_daily_loss_pct",
    "portfolio_drawdown_pct",
)


@dataclass
class ObservationBuilder:
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        self._all_names = tuple(list(self.feature_names) + list(CANDIDATE_FEATURE_NAMES))

    @property
    def all_feature_names(self) -> tuple[str, ...]:
        return self._all_names

    @property
    def dim(self) -> int:
        return len(self._all_names)

    def build(
        self,
        candidate: CandidateTrade,
        frame: FeatureFrame,
        portfolio_state: PortfolioState,
    ) -> np.ndarray:
        # 1. Pull the feature-frame vector in declared order.
        feat_vec = np.array([frame.values[name] for name in self.feature_names], dtype=np.float32)

        # 2. Candidate features.
        strat_id = STRATEGY_INDEX.get(candidate.strategy_id, STRATEGY_INDEX["unknown"])
        n_strategies = len(STRATEGY_INDEX)
        cand_feats = np.array([
            candidate.signal_strength,
            candidate.base_size_pct,
            candidate.max_holding_days / 30.0,
            strat_id / max(1, n_strategies - 1),
            portfolio_state.gross_exposure_pct,
            portfolio_state.open_positions_count / 10.0,
            portfolio_state.daily_loss_pct,
            portfolio_state.current_drawdown_pct,
        ], dtype=np.float32)

        return np.concatenate([feat_vec, cand_feats])

    def hash(self, obs: np.ndarray) -> str:
        return hashlib.sha1(obs.tobytes()).hexdigest()[:12]
