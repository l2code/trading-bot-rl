"""MultiStrategyObservationBuilder — observation packing for v2.

Where v1's observation is ``[features, candidate_features, portfolio]``
for a single candidate, v2 packs ALL strategy proposals for the
current ``(symbol, date)`` side by side: per-strategy slots of
``[fired, signal_strength, base_size_pct, max_holding_days_norm]``
plus the shared market features and portfolio. The agent sees the
full slate at once and can condition its choice on cross-strategy
agreement / disagreement.

Layout:
    [feature_frame_values..., per-strategy slots..., portfolio_state...]

Each per-strategy slot is fixed shape, zero-padded when that strategy
didn't fire on this (symbol, date), so the observation dimension is
constant across packs.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from rl_swing.domain import FeatureFrame, PortfolioState
from rl_swing.strategies.multi_strategy_packer import StrategyPack


# Per-strategy slot fields (in order). Keep stable.
PER_STRATEGY_SLOT_FIELDS: tuple[str, ...] = (
    "fired",                # 0/1 — did this strategy emit a candidate?
    "signal_strength",      # [0, 1]
    "base_size_pct",        # [0, 0.2] typically
    "max_holding_days_norm",  # holding_days / 30.0
)
SLOT_DIM = len(PER_STRATEGY_SLOT_FIELDS)

# Portfolio fields tacked onto the end. Mirror v1's layout.
PORTFOLIO_FIELDS: tuple[str, ...] = (
    "portfolio_gross_exposure_pct",
    "portfolio_open_positions_count_norm",
    "portfolio_daily_loss_pct",
    "portfolio_drawdown_pct",
)


@dataclass
class MultiStrategyObservationBuilder:
    feature_names: tuple[str, ...]
    n_strategies: int

    def __post_init__(self) -> None:
        slot_names: list[str] = []
        for i in range(self.n_strategies):
            for fld in PER_STRATEGY_SLOT_FIELDS:
                slot_names.append(f"strat_{i}_{fld}")
        self._all_names = tuple(
            list(self.feature_names) + slot_names + list(PORTFOLIO_FIELDS)
        )

    @property
    def all_feature_names(self) -> tuple[str, ...]:
        return self._all_names

    @property
    def dim(self) -> int:
        return len(self._all_names)

    def build(
        self,
        pack: StrategyPack,
        frame: FeatureFrame,
        portfolio_state: PortfolioState,
    ) -> np.ndarray:
        feat_vec = np.array(
            [frame.values.get(name, 0.0) for name in self.feature_names],
            dtype=np.float32,
        )
        # Per-strategy slots. Zero-padded if strategy didn't fire.
        slot_vec = np.zeros(self.n_strategies * SLOT_DIM, dtype=np.float32)
        for i, c in enumerate(pack.candidates):
            base = i * SLOT_DIM
            if c is None:
                continue
            slot_vec[base + 0] = 1.0
            slot_vec[base + 1] = float(c.signal_strength)
            slot_vec[base + 2] = float(c.base_size_pct)
            slot_vec[base + 3] = float(c.max_holding_days) / 30.0
        # Portfolio.
        port_vec = np.array([
            portfolio_state.gross_exposure_pct,
            portfolio_state.open_positions_count / 10.0,
            portfolio_state.daily_loss_pct,
            portfolio_state.current_drawdown_pct,
        ], dtype=np.float32)
        return np.concatenate([feat_vec, slot_vec, port_vec])

    def hash(self, obs: np.ndarray) -> str:
        return hashlib.sha1(obs.tobytes()).hexdigest()[:12]

    def fired_mask(self, pack: StrategyPack) -> np.ndarray:
        """Binary mask of which strategies fired. Useful for action
        masking in MaskablePPO once we add it."""
        return np.array(
            [1 if c is not None else 0 for c in pack.candidates],
            dtype=np.int8,
        )
