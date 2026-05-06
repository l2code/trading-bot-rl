"""PpoPolicyScorer — sb3 PPO wrapped in the PolicyScorer port."""
from __future__ import annotations

from dataclasses import dataclass

from rl_swing.features.pipelines import ALL_FEATURE_NAMES
from rl_swing.rl.agents.sb3_scorer import _Sb3Scorer


@dataclass
class PpoPolicyScorer(_Sb3Scorer):
    algorithm: str = "PPO"
    feature_names: tuple[str, ...] = ALL_FEATURE_NAMES
