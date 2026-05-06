"""Selector-style scorers for v2.

Where v1's PolicyScorer port takes a single CandidateTrade and emits
take/skip, v2 scorers operate on a StrategyPack — they pick *which*
strategy (or skip). This file defines a small port plus baseline +
PPO-backed implementations used by both training-time eval and walk-
forward.
"""
from __future__ import annotations

import logging
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from rl_swing.domain import FeatureFrame, PortfolioState
from rl_swing.features.pipelines import ALL_FEATURE_NAMES
from rl_swing.rl.env.multi_strategy_observation import (
    MultiStrategyObservationBuilder,
)
from rl_swing.strategies.multi_strategy_packer import StrategyPack

_log = logging.getLogger(__name__)


@runtime_checkable
class SelectorScorer(Protocol):
    """Inference port for v2."""

    model_id: str

    def select(
        self,
        pack: StrategyPack,
        feature: FeatureFrame,
        portfolio_state: PortfolioState,
    ) -> int:
        """Returns 0 (skip) or 1..N (take strategy k-1).

        Implementations must respect ``pack.candidates[k-1] is None``
        and either fall back to skip (0) or to a fired strategy.
        """


# ---------------------------------------------------------------------
@dataclass
class RandomSelectorScorer:
    model_id: str = "selector_baseline_random"
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def select(self, pack, feature, portfolio_state) -> int:
        # Skip 30% of the time, otherwise pick uniformly from fired
        # strategies. Mirrors v1's RandomPolicyScorer behavior.
        fired_idxs = [i for i, c in enumerate(pack.candidates) if c is not None]
        if not fired_idxs:
            return 0
        if self._rng.random() < 0.3:
            return 0
        return 1 + self._rng.choice(fired_idxs)


@dataclass
class AlwaysSkipSelectorScorer:
    model_id: str = "selector_baseline_always_skip"

    def select(self, pack, feature, portfolio_state) -> int:
        return 0


@dataclass
class AlwaysFirstFiredSelectorScorer:
    """Always pick the lowest-index strategy that fired. Mimics 'just
    take the first available candidate' — the dumbest non-skip rule.
    """
    model_id: str = "selector_baseline_first_fired"

    def select(self, pack, feature, portfolio_state) -> int:
        for i, c in enumerate(pack.candidates):
            if c is not None:
                return 1 + i
        return 0


@dataclass
class HighestSignalSelectorScorer:
    """Pick the strategy with the highest signal_strength among those
    that fired. This is essentially what v1's StrategyAggregator
    dedupe used to do before flattening to a single candidate."""
    model_id: str = "selector_baseline_highest_signal"

    def select(self, pack, feature, portfolio_state) -> int:
        best_idx = -1
        best_strength = -1.0
        for i, c in enumerate(pack.candidates):
            if c is None:
                continue
            if c.signal_strength > best_strength:
                best_idx = i
                best_strength = c.signal_strength
        if best_idx < 0:
            return 0
        return 1 + best_idx


# ---------------------------------------------------------------------
@dataclass
class PpoSelectorScorer:
    """sb3 PPO loaded from a saved artifact, wrapped in the
    SelectorScorer interface.

    Loading is lazy so the scorer can be instantiated in environments
    without sb3/torch (e.g. for the type to be importable in unit
    tests that don't run training)."""
    model_id: str
    artifact_path: str
    n_strategies: int
    feature_version: str = "features_v001_core_daily"
    feature_names: tuple[str, ...] = ALL_FEATURE_NAMES

    def __post_init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._obs_builder = MultiStrategyObservationBuilder(
            feature_names=self.feature_names,
            n_strategies=self.n_strategies,
        )

    def _load(self):
        if self._model is not None:
            return self._model
        path = Path(self.artifact_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Selector model artifact not found: {self.artifact_path}."
            )
        from stable_baselines3 import PPO
        self._model = PPO.load(str(path))
        return self._model

    def select(self, pack, feature, portfolio_state) -> int:
        if feature.feature_version != self.feature_version:
            raise RuntimeError(
                f"Feature version mismatch: model trained on "
                f"{self.feature_version!r}, frame is "
                f"{feature.feature_version!r}."
            )
        obs = self._obs_builder.build(pack, feature, portfolio_state)
        with self._lock:
            model = self._load()
            action, _state = model.predict(obs, deterministic=True)
        try:
            raw = int(np.asarray(action).reshape(-1)[0])
        except Exception:
            raw = int(action)  # type: ignore[arg-type]
        # Defensive: if the policy picks a non-fired strategy at
        # inference time, fall back to skip rather than an illegal
        # action. The training reward already discourages this; this
        # is just safety.
        if raw < 0 or raw > self.n_strategies:
            return 0
        if raw > 0 and pack.candidates[raw - 1] is None:
            return 0
        return raw
