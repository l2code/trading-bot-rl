"""DqnPolicyScorer — sb3 DQN wrapped in the PolicyScorer port."""
from __future__ import annotations

from dataclasses import dataclass

from rl_swing.features.pipelines import ALL_FEATURE_NAMES
from rl_swing.rl.agents.sb3_scorer import _Sb3Scorer


@dataclass
class DqnPolicyScorer(_Sb3Scorer):
    algorithm: str = "DQN"
    feature_names: tuple[str, ...] = ALL_FEATURE_NAMES
