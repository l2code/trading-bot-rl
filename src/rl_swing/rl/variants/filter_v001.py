"""FilterV001Variant — the original v1 trade-filter architecture.

Per-candidate decisions: agent sees one CandidateTrade at a time and
chooses skip / take_25 / take_50 / take_100. Strategies are deduped
by ``(symbol, date)`` keeping the highest-signal-strength one before
the agent sees them.

This file extracts what used to live inline in trainer.py and
walk_forward.py so the same logic is reachable through the
TrainingVariant interface.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

import gymnasium as gym

from rl_swing.domain import PortfolioState
from rl_swing.features.pipelines import ALL_FEATURE_NAMES
from rl_swing.ports import PolicyScorer
from rl_swing.rl.agents.baseline_scorers import (
    AlwaysTakePolicyScorer,
    NeverTakePolicyScorer,
    RandomPolicyScorer,
)
from rl_swing.rl.env.swing_env import SwingTradingEnv
from rl_swing.rl.validation.metrics import validation_composite_score
from rl_swing.rl.variants.base import (
    EnvBuildContext,
    EvaluationContext,
    PolicyResult,
    TrainingVariant,
)
from rl_swing.strategies.aggregator import StrategyAggregator
from rl_swing.strategies.breakout import BreakoutStrategy
from rl_swing.strategies.mean_reversion import RsiMeanReversionStrategy
from rl_swing.strategies.momentum import MomentumStrategy

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
def _build_default_strategies() -> list[Any]:
    """The looser strategy config that the v1 filter trains on. Kept
    here so changing it is a one-line variant change rather than a
    cross-file edit.
    """
    return [
        MomentumStrategy(
            min_relative_strength=-0.05,
            min_r20=-0.02,
            require_sma200_above=False,
        ),
        RsiMeanReversionStrategy(rsi_threshold=35.0),
        BreakoutStrategy(
            min_relative_volume=0.7,
            max_distance_below_high=-0.02,
        ),
    ]


def _aggregate_candidates(frames, portfolio):
    return list(
        StrategyAggregator(_build_default_strategies()).generate(frames, portfolio)
    )


# ---------------------------------------------------------------------
class FilterV001Variant:
    name: str = "filter_v001"

    # ---- env -----------------------------------------------------
    def build_env(self, ctx: EnvBuildContext) -> gym.Env:
        candidates = _aggregate_candidates(ctx.frames, ctx.portfolio)
        return SwingTradingEnv(
            bars=ctx.bars,
            candidates=candidates,
            feature_frames=ctx.frames,
            feature_names=ALL_FEATURE_NAMES,
            sampler_kind=ctx.sampler_kind,
            sampler_seed=ctx.seed,
            sampler_window_days=120,
            cost_model=ctx.cost_model,
            reward_model=ctx.reward_model,
        )

    # ---- evaluation ---------------------------------------------
    def evaluate(self, ctx: EvaluationContext) -> list[PolicyResult]:
        # Build candidates the same way the env did during training.
        portfolio = PortfolioState(
            as_of=datetime(ctx.test_end.year, ctx.test_end.month, ctx.test_end.day),
            cash=100_000.0, equity=100_000.0,
        )
        candidates = _aggregate_candidates(ctx.frames, portfolio)

        scorers: list[PolicyScorer] = []
        if "random" in ctx.include_baselines:
            scorers.append(RandomPolicyScorer(model_id="baseline_random", seed=42))
        if "always_take_100" in ctx.include_baselines:
            scorers.append(AlwaysTakePolicyScorer(
                model_id="baseline_always_take_100", action="take_100"))
        if "always_take_50" in ctx.include_baselines:
            scorers.append(AlwaysTakePolicyScorer(
                model_id="baseline_always_take_50", action="take_50"))
        if "never_take" in ctx.include_baselines:
            scorers.append(NeverTakePolicyScorer(model_id="baseline_never_take"))

        rl_added = False
        if ctx.artifact_path is not None and ctx.artifact_path.exists():
            from rl_swing.rl.agents.dqn_scorer import DqnPolicyScorer
            from rl_swing.rl.agents.ppo_scorer import PpoPolicyScorer
            algorithm = (ctx.experiment_config.get("algorithm") or "PPO").upper()
            AlgoCls = PpoPolicyScorer if algorithm == "PPO" else DqnPolicyScorer
            scorers.append(AlgoCls(
                model_id=ctx.model_id,
                artifact_path=str(ctx.artifact_path),
                feature_version="features_v001_core_daily",
            ))
            rl_added = True

        results: list[PolicyResult] = []
        for s in scorers:
            res = self._evaluate_scorer(
                s, candidates, ctx, cost_stress_multiplier=1.0,
            )
            results.append(res)
            if ctx.include_cost_stress:
                res2 = self._evaluate_scorer(
                    s, candidates, ctx, cost_stress_multiplier=2.0,
                )
                results.append(PolicyResult(
                    **{**res2.to_dict(), "model_id": res2.model_id + "_cost2x"}
                ))

        # Stash whether we found a trained model so the report can
        # surface it.
        for r in results:
            r.extras.setdefault("rl_model_present", rl_added)
        return results

    # ---- internals ----------------------------------------------
    def _evaluate_scorer(
        self, scorer, candidates, ctx: EvaluationContext, *,
        cost_stress_multiplier: float,
    ) -> PolicyResult:
        # Reuse the existing per-candidate evaluator.
        from rl_swing.rl.validation.walk_forward import evaluate_policy as _eval

        d = _eval(
            scorer, ctx.bars, candidates, ctx.frames,
            cost_model=ctx.cost_model,
            reward_model=ctx.reward_model,
            cost_stress_multiplier=cost_stress_multiplier,
        )
        # ``evaluate_policy`` returns a dict with our metric keys plus
        # 'decisions' (which we don't need to repeat in the result).
        comp = d.get("components", {})
        return PolicyResult(
            model_id=d.get("model_id", scorer.model_id),
            n_trades=int(d.get("n_trades", 0)),
            total_return=float(d.get("total_return", 0.0)),
            annualized_sharpe=float(d.get("annualized_sharpe", 0.0)),
            profit_factor=float(d.get("profit_factor", 0.0)),
            max_drawdown=float(d.get("max_drawdown", 0.0)),
            turnover_take_rate=float(d.get("turnover_take_rate", 0.0)),
            mean_reward=float(d.get("mean_reward", 0.0)),
            validation_composite_score=float(d.get("validation_composite_score", 0.0)),
            components=dict(comp),
            cost_stress_multiplier=float(cost_stress_multiplier),
        )
