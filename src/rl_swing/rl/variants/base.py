"""TrainingVariant — the pluggable interface every RL architecture
implements.

A variant owns:
    1. ``build_env(EnvBuildContext)`` — constructs a Gymnasium env
       suitable for the variant's action space and observation shape.
    2. ``evaluate(EvaluationContext)`` — runs the variant's full
       walk-forward inference: every baseline + the trained model
       (if available), returning a list of PolicyResult.

That's it. The trainer doesn't know what shape the env is. The walk-
forward harness doesn't know what kind of policy a variant uses. The
trainer/harness just dispatch to the variant.

Adding a new variant is:
    1. Create ``rl_swing/rl/variants/<name>.py`` with a TrainingVariant
       impl.
    2. Register it in ``configs/components/components.yaml`` under
       category ``rl_variants``.
    3. Set ``experiment.rl_variant: <name>`` in the experiment YAML.

No changes to trainer.py or walk_forward.py.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import gymnasium as gym

from rl_swing.domain import FeatureFrame, MarketBar, PortfolioState
from rl_swing.rl.env.cost_model import EquityExecutionModel
from rl_swing.rl.env.reward_model import RewardModel


# ---------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EnvBuildContext:
    """Everything a variant needs to build a training/validation env.

    Bars + frames are pre-loaded by the trainer (via the experiment's
    data provider) so each variant works against the same input
    universe and time slice. The variant decides how to aggregate
    candidates, build observations, and shape rewards beyond the
    common cost/reward models.
    """
    bars: Sequence[MarketBar]
    frames: Sequence[FeatureFrame]
    portfolio: PortfolioState
    sampler_kind: str               # "random" | "chronological"
    seed: int
    cost_model: EquityExecutionModel
    reward_model: RewardModel
    # Pass-through of the experiment YAML's full ``experiment`` block,
    # so variants can read variant-specific knobs (e.g. action space
    # size, strategy config) without breaking the common interface.
    experiment_config: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    """Everything a variant needs to run walk-forward inference."""
    bars: Sequence[MarketBar]
    frames: Sequence[FeatureFrame]
    test_start: date
    test_end: date
    cost_model: EquityExecutionModel
    reward_model: RewardModel
    artifact_path: Path | None        # None = no trained model present
    model_id: str
    include_baselines: tuple[str, ...]
    include_cost_stress: bool
    experiment_config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PolicyResult:
    """One row in a walk-forward report — same shape across variants
    so the report stays consumable by downstream tools regardless of
    which architecture produced it."""
    model_id: str
    n_trades: int
    total_return: float
    annualized_sharpe: float
    profit_factor: float
    max_drawdown: float
    turnover_take_rate: float
    mean_reward: float
    validation_composite_score: float
    components: dict[str, float]
    cost_stress_multiplier: float = 1.0
    # Variant-specific extras (e.g. selector: action distribution per
    # strategy; filter: action distribution per size bucket).
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "model_id": self.model_id,
            "n_trades": self.n_trades,
            "total_return": self.total_return,
            "annualized_sharpe": self.annualized_sharpe,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "turnover_take_rate": self.turnover_take_rate,
            "mean_reward": self.mean_reward,
            "validation_composite_score": self.validation_composite_score,
            "components": dict(self.components),
            "cost_stress_multiplier": self.cost_stress_multiplier,
        }
        if self.extras:
            d["extras"] = dict(self.extras)
        return d


# ---------------------------------------------------------------------
@runtime_checkable
class TrainingVariant(Protocol):
    """The interface every variant implements."""

    name: str           # human-readable, matches the registry key

    def build_env(self, ctx: EnvBuildContext) -> gym.Env:
        """Build a Gymnasium env for training or validation."""

    def evaluate(self, ctx: EvaluationContext) -> list[PolicyResult]:
        """Run walk-forward evaluation and return one PolicyResult
        per (baseline | trained_model) (× cost-stress variants if
        ``ctx.include_cost_stress``).

        If ``ctx.artifact_path`` is None or doesn't exist, the variant
        should still return baseline results — never throw.
        """


# ---------------------------------------------------------------------
# Convenience: load a variant by name from the registry.
# ---------------------------------------------------------------------
def load_variant(
    name: str,
    *,
    components_path: str | Path = "configs/components/components.yaml",
) -> TrainingVariant:
    """Resolve a TrainingVariant by registry name.

    Trainer and walk-forward both call this so the dispatch logic
    lives in one place.
    """
    from rl_swing.runtime.registry import ComponentRegistry

    reg = ComponentRegistry.from_yaml(components_path)
    variant = reg.build("rl_variants", name)
    if not isinstance(variant, TrainingVariant):
        raise TypeError(
            f"Component rl_variants.{name!r} resolves to {type(variant)} "
            f"which doesn't implement TrainingVariant (needs build_env "
            f"and evaluate methods)."
        )
    return variant
