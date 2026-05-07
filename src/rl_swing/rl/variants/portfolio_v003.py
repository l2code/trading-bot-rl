"""PortfolioV003Variant — v3 chronological selector (FEAT-32 M1).

Steps once per trading day. Action = ``Discrete(1 + max_top_k)``
(no-op + take-top-k by signal_strength). State sees the daily slate
aggregate + portfolio. Trains via standard PPO/MaskablePPO over the
chronological MDP.

M1 scope: env build + day-ordered evaluate path that produces one
PolicyResult per portfolio baseline (no_op, top1, top2, random_action)
plus the trained PPO model when its artifact is on disk. Computes
the validation_composite_score from the per-day P&L series the
``PortfolioStateTracker`` accumulates.

Out of scope for M1 (filed as M3+ work):
  - PpoPortfolioPolicy with masking. M1 uses base PPO/MlpPolicy
    over the small action lattice (Discrete(3) for K=2).
  - Cost-stress alternate runs. M1 reports cost_stress_multiplier=1.0
    rows only.
"""
from __future__ import annotations

import logging
from datetime import datetime

import gymnasium as gym

from rl_swing.domain import PortfolioState
from rl_swing.rl.agents.portfolio_baselines import (
    BCPortfolioPolicy,
    NoOpPortfolioPolicy,
    RandomActionPortfolioPolicy,
    TopKPortfolioPolicy,
)
from rl_swing.rl.env.chronological_swing_env import ChronologicalSwingEnv
from rl_swing.rl.variants.base import (
    EnvBuildContext,
    EvaluationContext,
    PolicyResult,
)
from rl_swing.strategies.breakout import BreakoutStrategy
from rl_swing.strategies.mean_reversion import RsiMeanReversionStrategy
from rl_swing.strategies.momentum import MomentumStrategy
from rl_swing.strategies.multi_strategy_packer import MultiStrategyPacker

_log = logging.getLogger(__name__)


def _build_default_strategies() -> list:
    """Same loose defaults as v002 — preserves apples-to-apples
    candidate sets so the v002↔v003 comparison isolates decision
    shape, not candidate generation."""
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


def _pack_for_window(frames, portfolio):
    packer = MultiStrategyPacker(_build_default_strategies())
    return packer.pack(frames, portfolio), packer.n_slots


class PortfolioV003Variant:
    name: str = "portfolio_v003"

    # ---- env ---------------------------------------------------------
    def build_env(self, ctx: EnvBuildContext) -> gym.Env:
        packs, _n_slots = _pack_for_window(ctx.frames, ctx.portfolio)
        exp = ctx.experiment_config or {}
        max_top_k = int(exp.get("v003_max_top_k", 2))
        sampler_window = int(exp.get("v003_sampler_window_days", 60))
        return ChronologicalSwingEnv(
            bars=ctx.bars,
            packs=packs,
            feature_frames=ctx.frames,
            cost_model=ctx.cost_model,
            reward_model=ctx.reward_model,
            starting_equity=100_000.0,
            max_top_k=max_top_k,
            sampler_kind=ctx.sampler_kind,
            sampler_seed=ctx.seed,
            sampler_window_days=sampler_window,
        )

    # ---- evaluate ----------------------------------------------------
    def evaluate(self, ctx: EvaluationContext) -> list[PolicyResult]:
        """Run each baseline (and trained PPO if present) over the
        FULL test window chronologically. Returns one PolicyResult
        per policy. M1 reports cost_stress_multiplier=1.0 only.
        """
        portfolio = PortfolioState(
            as_of=datetime(ctx.test_end.year, ctx.test_end.month, ctx.test_end.day),
            cash=100_000.0, equity=100_000.0,
        )
        packs, _n_slots = _pack_for_window(ctx.frames, portfolio)
        exp = ctx.experiment_config or {}
        max_top_k = int(exp.get("v003_max_top_k", 2))
        n_actions = 1 + max_top_k

        # Policies (per the M1 plan):
        #   no_op, top1, top2 (capped at max_top_k), random_action,
        #   plus trained PPO if model.zip exists.
        policies: list = [
            NoOpPortfolioPolicy(),
            TopKPortfolioPolicy(k=1),
        ]
        if max_top_k >= 2:
            policies.append(TopKPortfolioPolicy(k=2))
        policies.append(RandomActionPortfolioPolicy(n_actions=n_actions, seed=42))

        # FEAT-32 M2: behavioral-cloning baseline. Auto-included when
        # the artifact exists at the standard path. Trained offline via
        # ``scripts/train_bc_v003.py``. The diagnostic question is:
        # can a supervised classifier imitate a non-trivial state-
        # dependent target on this env? If not, the env is unlearnable
        # and PPO won't escape that. Mirrors FEAT-30 supervised-ranker
        # auto-inclusion in selector_v002.
        from pathlib import Path as _Path
        bc_path = _Path("data/models/portfolio_baseline_bc/model.joblib")
        if bc_path.exists():
            policies.append(BCPortfolioPolicy(
                artifact_path=str(bc_path), n_actions=n_actions,
            ))

        rl_added = False
        algorithm = str(exp.get("algorithm", "PPO")).strip()
        is_maskable = algorithm.upper() in ("MASKABLEPPO", "MASKABLE_PPO")
        if ctx.artifact_path is not None and ctx.artifact_path.exists():
            try:
                if is_maskable:
                    # FEAT-32 M3: sb3-contrib MaskablePPO inference.
                    # Mirrors v002 FEAT-29 MaskablePpoSelectorScorer:
                    # the wrapper rebuilds the same mask the trainer
                    # used and routes it through model.predict().
                    from sb3_contrib import MaskablePPO  # type: ignore[import-not-found]
                    trained_model = MaskablePPO.load(str(ctx.artifact_path))
                    trained_id = ctx.model_id
                    policies.append(_TrainedMaskablePpoWrapper(
                        trained_model, trained_id,
                    ))
                else:
                    from stable_baselines3 import PPO
                    trained_model = PPO.load(str(ctx.artifact_path))
                    trained_id = ctx.model_id
                    policies.append(_TrainedPpoWrapper(trained_model, trained_id))
                rl_added = True
            except Exception as e:  # pragma: no cover
                _log.warning("portfolio_v003 trained model load failed: %s", e)

        results: list[PolicyResult] = []
        for pol in policies:
            res = self._eval_policy(
                pol, packs=packs, frames=ctx.frames, bars=ctx.bars,
                cost_model=ctx.cost_model, reward_model=ctx.reward_model,
                test_start=ctx.test_start, test_end=ctx.test_end,
                max_top_k=max_top_k,
            )
            results.append(res)
        for r in results:
            r.extras.setdefault("rl_model_present", rl_added)
            r.extras.setdefault("variant", "portfolio_v003")
        return results

    def _eval_policy(
        self, policy, *, packs, frames, bars, cost_model, reward_model,
        test_start, test_end, max_top_k: int,
    ) -> PolicyResult:
        """Run one full chronological pass over the test window."""
        from rl_swing.rl.validation.metrics import (
            validation_composite_score_from_daily_pnl,
        )
        from rl_swing.rl.validation.portfolio_pnl import TradeRecord

        env = ChronologicalSwingEnv(
            bars=bars, packs=packs, feature_frames=frames,
            cost_model=cost_model, reward_model=reward_model,
            starting_equity=100_000.0,
            max_top_k=max_top_k,
            sampler_kind="chronological",
            sampler_seed=0,
        )
        obs, _info = env.reset()
        # FEAT-32 M3: bind env on the maskable wrapper so its decide()
        # can route env.action_masks() through model.predict(). Vanilla
        # wrappers ignore set_env (or don't define it).
        if hasattr(policy, "set_env"):
            policy.set_env(env)
        rewards: list[float] = []
        actions_taken: list[str] = []
        per_action_counts = [0] * (1 + max_top_k)
        done = False
        n_packs_seen_total = 0
        while not done:
            action = policy.decide(obs)
            per_action_counts[int(action)] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            n_packs_seen_total += int(info.get("n_packs_today", 0))
            n_op = int(info.get("n_opened_today", 0))
            for _ in range(n_op):
                actions_taken.append("take")
            # No-op days produce a synthetic 'skip' so the take_rate
            # diagnostic is consistent with v002's (n_takes /
            # (n_takes + n_skips)).
            if n_op == 0:
                actions_taken.append("skip")
            done = bool(terminated) or bool(truncated)

        # Build per-day TradeRecord-equivalent series. The tracker
        # already records realized trades; convert to TradeRecord so
        # validation_composite_score_from_daily_pnl can score it on
        # the same canonical metric stack as v001/v002.
        trade_records: list[TradeRecord] = [
            TradeRecord(
                entry_date=t.entry_date,
                exit_date=t.exit_date,
                return_pct=t.net_return_pct,
                size_pct=t.size_pct,
            )
            for t in env.tracker.closed_trades
        ]
        trading_days = sorted({
            b.timestamp.date() for b in bars
            if test_start <= b.timestamp.date() <= test_end
        })
        score, breakdown = validation_composite_score_from_daily_pnl(
            trades=trade_records,
            n_total_packs=len(actions_taken),
            rewards=rewards, actions=actions_taken,
            window_start=test_start, window_end=test_end,
            trading_days=trading_days or None,
        )
        extras = {
            "metric_basis": breakdown.get("metric_basis"),
            "n_trading_days": breakdown.get("n_trading_days", 0),
            "per_action_counts": list(per_action_counts),
            "n_packs_seen_total": int(n_packs_seen_total),
            "n_trades_opened": int(env.tracker.n_trades_opened),
            "n_trades_closed": int(env.tracker.n_trades_closed),
        }
        return PolicyResult(
            model_id=policy.model_id,
            n_trades=int(breakdown.get("n_trades", 0)),
            total_return=float(breakdown.get("total_return", 0.0)),
            annualized_sharpe=float(breakdown.get("annualized_sharpe", 0.0)),
            profit_factor=float(breakdown.get("profit_factor", 0.0)),
            max_drawdown=float(breakdown.get("max_drawdown", 0.0)),
            turnover_take_rate=float(breakdown.get("turnover_take_rate", 0.0)),
            mean_reward=float(breakdown.get("mean_reward", 0.0)),
            validation_composite_score=float(score),
            components=dict(breakdown.get("components", {})),
            cost_stress_multiplier=1.0,
            extras=extras,
        )


class _TrainedPpoWrapper:
    """Adapts an sb3 PPO model into the policy.decide() interface
    that ChronologicalSwingEnv's evaluate path expects."""
    def __init__(self, model, model_id: str) -> None:
        self._model = model
        self.model_id = model_id

    def decide(self, obs) -> int:
        action, _ = self._model.predict(obs, deterministic=True)
        return int(action)


class _TrainedMaskablePpoWrapper:
    """FEAT-32 M3: adapts an sb3-contrib MaskablePPO model into the
    policy.decide() interface. Rebuilds the same action mask the
    trainer used (via ``env.action_masks()``) and routes it through
    ``model.predict(obs, action_masks=mask)`` so eval-time inference
    matches the masking the policy was trained under.

    Mirrors v002's ``MaskablePpoSelectorScorer``. The eval loop calls
    ``set_env(env)`` once after ``env.reset()``; subsequent ``decide``
    calls fetch the mask from the bound env (which advances per-step
    inside ``env.step``)."""
    def __init__(self, model, model_id: str) -> None:
        self._model = model
        self.model_id = model_id
        self._env = None

    def set_env(self, env) -> None:
        self._env = env

    def decide(self, obs) -> int:
        import numpy as np
        mask = None
        if self._env is not None and hasattr(self._env, "action_masks"):
            mask = self._env.action_masks()
        if mask is None:
            action, _ = self._model.predict(obs, deterministic=True)
        else:
            action, _ = self._model.predict(
                obs, deterministic=True, action_masks=mask,
            )
        # MaskablePPO.predict returns either a scalar or a 1-element
        # array depending on the obs shape; coerce to scalar.
        if isinstance(action, np.ndarray):
            action = action.item() if action.size == 1 else int(action[0])
        return int(action)
