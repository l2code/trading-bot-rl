"""RL training variants — pluggable per-architecture bundles.

Each variant is a self-contained "training architecture": its own env
class, observation shape, action space, candidate aggregation, and
walk-forward evaluator. The trainer and walk-forward harness dispatch
via the component registry so adding v3, v4, ... is a one-file change
plus a registry entry.

Variant lifecycle:
    1. Experiment YAML names a variant via ``experiment.rl_variant``.
    2. Trainer resolves the variant via the registry and asks it to
       build a training env from (bars, frames, cost, reward, ...).
    3. Walk-forward resolves the same variant and asks it to evaluate
       the trained model + a set of baselines on the test window.
    4. Each variant is responsible for the inference path that
       matches its env shape — selector envs need pack-based
       baselines, filter envs need candidate-based baselines.
"""
from rl_swing.rl.variants.base import (
    EnvBuildContext,
    EvaluationContext,
    PolicyResult,
    TrainingVariant,
)

__all__ = [
    "EnvBuildContext",
    "EvaluationContext",
    "PolicyResult",
    "TrainingVariant",
]
