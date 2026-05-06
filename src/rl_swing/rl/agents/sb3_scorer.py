"""Common ``PolicyScorer`` wrapping a stable-baselines3 model.

Both ``PpoPolicyScorer`` and ``DqnPolicyScorer`` reuse this class to
avoid two copies of the loading + observation-construction logic.

Loading is lazy — the artifact is opened on first ``score()`` so the
adapter can be instantiated in environments without sb3/torch.
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rl_swing.domain import (
    CandidateTrade,
    FeatureFrame,
    PolicyDecision,
    PortfolioState,
)
from rl_swing.features.pipelines import ALL_FEATURE_NAMES
from rl_swing.rl.env.action_mapper import to_literal
from rl_swing.rl.env.observation_builder import ObservationBuilder

_log = logging.getLogger(__name__)


@dataclass
class _Sb3Scorer:
    model_id: str
    artifact_path: str
    feature_version: str = "features_v001_core_daily"
    feature_names: tuple[str, ...] = ALL_FEATURE_NAMES
    algorithm: str = "PPO"

    def __post_init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._obs_builder = ObservationBuilder(feature_names=self.feature_names)

    def _load(self):
        if self._model is not None:
            return self._model
        path = Path(self.artifact_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {self.artifact_path}. "
                f"Train first via ``rl-swing train --experiment ...``."
            )
        if self.algorithm.upper() == "PPO":
            from stable_baselines3 import PPO
            self._model = PPO.load(str(path))
        elif self.algorithm.upper() == "DQN":
            from stable_baselines3 import DQN
            self._model = DQN.load(str(path))
        else:
            raise ValueError(f"unknown algorithm: {self.algorithm}")
        return self._model

    def score(
        self,
        candidate: CandidateTrade,
        features: FeatureFrame,
        portfolio_state: PortfolioState,
    ) -> PolicyDecision:
        # Build observation. If feature version doesn't match what the
        # model was trained on, refuse rather than silently scoring on
        # the wrong vector.
        if features.feature_version != self.feature_version:
            raise RuntimeError(
                f"Feature version mismatch: model trained on {self.feature_version!r}, "
                f"frame is {features.feature_version!r}. "
                "Reconciliation/promotion gate caught a mismatch."
            )
        obs = self._obs_builder.build(candidate, features, portfolio_state)
        with self._lock:
            model = self._load()
            action, _state = model.predict(obs, deterministic=True)
        try:
            raw = int(np.asarray(action).reshape(-1)[0])
        except Exception:
            raw = int(action)  # type: ignore[arg-type]
        action_literal = to_literal(raw)
        from rl_swing.domain import ACTION_TO_SIZE
        target = candidate.base_size_pct * ACTION_TO_SIZE[action_literal]
        return PolicyDecision(
            decision_id=f"dec-{uuid.uuid4().hex[:12]}",
            candidate_id=candidate.candidate_id,
            as_of=candidate.as_of,
            model_id=self.model_id,
            action=action_literal,
            confidence=None,    # PPO doesn't expose probs cheaply; DQN gives Q-values
            target_size_pct=target,
            raw_action=raw,
            observation_hash=self._obs_builder.hash(obs),
            explanation={
                "algorithm": self.algorithm,
                "feature_version": self.feature_version,
            },
        )
