"""Trainer.

Loads an experiment YAML, builds a training env and a validation env
on the appropriate date ranges, and runs PPO/DQN training with an
evaluation callback that early-stops when validation stops improving.

The validation composite score is computed by
``rl_swing.rl.validation.metrics.validation_composite_score``.

This trainer is callable from the CLI (``rl-swing train``) and from
the Colab notebook (``rl_swing.rl.training.colab_entrypoint.train``).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

from rl_swing.adapters.data.synthetic_provider import SyntheticProvider
from rl_swing.features.pipelines import CoreDailyPipeline
from rl_swing.rl.env.cost_model import EquityExecutionModel
from rl_swing.rl.env.reward_model import RewardModel

# Calendar days of bars to load before the requested window so that
# long-lookback features (sma_200, return_60d, atr_pct_14, etc.) are
# fully populated on the FIRST in-window day. 1.5 trading years
# covers the longest lookback (sma_200 = 200 trading days ≈ 280
# calendar days) with margin for non-trading days. See FIX-24.
_FEATURE_WARMUP_DAYS = int(252 * 1.5)


def _load_bars_with_warmup(provider, symbols, start: date, end: date,
                           warmup_days: int = _FEATURE_WARMUP_DAYS):
    """Load bars from (start - warmup) to end so feature pipelines
    can populate long-lookback fields before the in-window region.
    Returns (bars, warmup_start)."""
    warmup_start = start - timedelta(days=warmup_days)
    bars = list(provider.get_bars(symbols, warmup_start, end, "1d", True))
    return bars, warmup_start


def _filter_frames_to_window(frames, start: date, end: date):
    """Keep only frames whose ``as_of`` falls in [start, end]. Used
    after building features over the extended window so candidates
    only fire in the actual eval window."""
    return [
        f for f in frames
        if start <= f.as_of.date() <= end
    ]

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
@dataclass
class _ExperimentCfg:
    name: str
    algorithm: str
    feature_pipeline: str
    universe: str
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date
    total_timesteps_initial: int
    total_timesteps_max: int
    eval_interval_timesteps: int
    early_stopping_patience: int
    min_validation_delta: float
    seeds: list[int]
    hyperparams: dict
    cost_model: dict
    reward: dict
    artifact_root: str
    data_provider: str = "synthetic_momentum"
    # Which RL variant to use (registry name in components.yaml under
    # category 'rl_variants'). Defaults to filter_v001 for backward
    # compatibility with experiments that don't set it.
    rl_variant: str = "filter_v001"
    # Pass-through of the raw experiment YAML block — variants can
    # read variant-specific knobs without us having to thread every
    # knob through this dataclass.
    raw_experiment: dict = None  # type: ignore[assignment]

    @classmethod
    def from_yaml(cls, path: str | Path) -> _ExperimentCfg:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        e = data["experiment"]
        return cls(
            name=e["name"],
            algorithm=e["algorithm"],
            feature_pipeline=e["feature_pipeline"],
            universe=e["universe"],
            train_start=date.fromisoformat(e["train_start"]),
            train_end=date.fromisoformat(e["train_end"]),
            validation_start=date.fromisoformat(e["validation_start"]),
            validation_end=date.fromisoformat(e["validation_end"]),
            test_start=date.fromisoformat(e["test_start"]),
            test_end=date.fromisoformat(e["test_end"]),
            total_timesteps_initial=int(e["total_timesteps_initial"]),
            total_timesteps_max=int(e["total_timesteps_max"]),
            eval_interval_timesteps=int(e["eval_interval_timesteps"]),
            early_stopping_patience=int(e.get("early_stopping_patience_evaluations", 10)),
            min_validation_delta=float(e.get("min_validation_delta", 0.0)),
            seeds=list(e.get("seeds") or [11]),
            hyperparams=dict(e.get("hyperparams") or {}),
            cost_model=dict(e.get("cost_model") or {}),
            reward=dict(e.get("reward") or {}),
            artifact_root=str(e.get("artifact_root", "data/models/")),
            data_provider=e.get("data_provider", "synthetic_momentum"),
            rl_variant=str(e.get("rl_variant", "filter_v001")),
            raw_experiment=dict(e),
        )


# ---------------------------------------------------------------------
def _load_universe_symbols(name: str) -> list[str]:
    candidate_paths = [
        Path("configs/universes") / f"{name}.yaml",
        Path(name),
    ]
    for p in candidate_paths:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return list((yaml.safe_load(f) or {}).get("universe", {}).get("symbols", []))
    raise FileNotFoundError(f"universe {name!r} not found in configs/universes/")


def _build_provider(provider_name: str):
    """Pick a provider by short name. Smoke-test mode uses synthetic
    data; full runs (yfinance/wrds) go through the registry."""
    if provider_name.startswith("synthetic_"):
        regime = provider_name.split("_", 1)[1]
        seed_for_regime = {"momentum": 11, "mean_reversion": 22, "random_walk": 33}
        return SyntheticProvider(regime=regime, seed=seed_for_regime.get(regime, 11))
    if provider_name == "yfinance_daily":
        from rl_swing.adapters.data.yfinance_provider import YFinanceProvider
        return YFinanceProvider()
    if provider_name == "wrds_parquet":
        from rl_swing.adapters.data.wrds_parquet_provider import WrdsParquetProvider
        return WrdsParquetProvider()
    raise ValueError(f"unknown data provider for training: {provider_name!r}")


def _build_env(
    cfg: _ExperimentCfg,
    *,
    start: date,
    end: date,
    sampler_kind: str,
    seed: int,
    provider_name: str,
):
    """Build a training/validation env for the experiment's RL
    variant. Dispatches to ``rl_swing.rl.variants`` via the component
    registry — adding a new variant is a registry entry + new file.
    """
    from rl_swing.domain import PortfolioState
    from rl_swing.rl.variants import EnvBuildContext
    from rl_swing.rl.variants.base import load_variant

    provider = _build_provider(provider_name)
    symbols = _load_universe_symbols(cfg.universe)
    # FIX-24: load with warmup so long-lookback features (sma_200,
    # return_60d, etc.) are populated on the first in-window day.
    # Without this, the first ~200 days of every test window have
    # degraded or missing features.
    bars, _warmup_start = _load_bars_with_warmup(provider, symbols, start, end)

    pipeline = CoreDailyPipeline()
    all_frames = list(pipeline.build(bars))
    # Filter frames to the in-window region so candidates only fire
    # in the actual eval period. Bars are kept full so the simulator
    # can find entry indices and apply ATR-based exits with proper
    # historical context.
    frames = _filter_frames_to_window(all_frames, start, end)

    portfolio = PortfolioState(
        as_of=datetime(end.year, end.month, end.day),
        cash=100_000.0, equity=100_000.0,
    )

    cost = EquityExecutionModel(**cfg.cost_model) if cfg.cost_model else EquityExecutionModel()
    # FIX-#62: missing-key fallbacks now match the FIX-#58 dataclass
    # defaults so a partial reward YAML doesn't silently revert to
    # the pre-FIX-49 calibration. Source-of-truth for these defaults
    # is RewardModel; keep this list in sync if defaults change.
    reward = RewardModel(
        target_risk_pct=0.02,
        drawdown_penalty_weight=cfg.reward.get("drawdown_penalty_weight", 0.10),
        turnover_penalty_weight=cfg.reward.get("turnover_penalty_weight", 0.05),
        holding_period_penalty_weight=cfg.reward.get("holding_period_penalty_weight", 0.02),
        skip_counterfactual_scale=cfg.reward.get("skip_counterfactual_scale", 1.0),
    )

    variant_name = cfg.rl_variant or "filter_v001"
    variant = load_variant(variant_name)
    env_ctx = EnvBuildContext(
        bars=bars, frames=frames, portfolio=portfolio,
        sampler_kind=sampler_kind, seed=int(seed),
        cost_model=cost, reward_model=reward,
        experiment_config=cfg.raw_experiment,
    )
    return variant.build_env(env_ctx)


# ---------------------------------------------------------------------
def train_from_experiment(
    experiment_path: str | Path,
    *,
    total_timesteps_override: int | None = None,
    seed_override: int | None = None,
    data_provider_override: str | None = None,
    artifact_root_override: str | None = None,
    n_envs: int = 1,
    hyperparam_overrides: dict | None = None,
) -> dict:
    """Train from an experiment YAML.

    ``hyperparam_overrides`` is a shallow dict that merges over
    ``cfg.hyperparams`` (override wins). Useful for CLI / sweep
    drivers that want to A/B different ``ent_coef`` or
    ``learning_rate`` values without editing the YAML.
    """
    cfg = _ExperimentCfg.from_yaml(experiment_path)
    if hyperparam_overrides:
        merged = dict(cfg.hyperparams)
        merged.update(hyperparam_overrides)
        cfg.hyperparams = merged
    total_timesteps = int(total_timesteps_override or cfg.total_timesteps_initial)
    seeds = [seed_override] if seed_override is not None else cfg.seeds
    artifact_root = Path(artifact_root_override or cfg.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    provider_name = data_provider_override or cfg.data_provider
    n_envs = max(1, int(n_envs))

    summary = {
        "experiment": cfg.name,
        "algorithm": cfg.algorithm,
        "total_timesteps": total_timesteps,
        "seeds": seeds,
        "data_provider": provider_name,
        "n_envs": n_envs,
        "hyperparam_overrides": dict(hyperparam_overrides) if hyperparam_overrides else None,
        "runs": [],
    }

    for seed in seeds:
        run_summary = _run_single_seed(
            cfg=cfg,
            seed=int(seed),
            total_timesteps=total_timesteps,
            artifact_root=artifact_root,
            provider_name=provider_name,
            n_envs=n_envs,
        )
        summary["runs"].append(run_summary)

    # FIX-#53: write the top-level model.zip alias pointing at the
    # best checkpoint **across all seeds** (was per-seed, last-wins).
    best_run = max(
        (r for r in summary["runs"] if r.get("best_validation_score") is not None),
        key=lambda r: r["best_validation_score"],
        default=None,
    )
    if best_run is not None and best_run.get("best_path"):
        alias = artifact_root / cfg.name / "model.zip"
        try:
            if alias.exists():
                alias.unlink()
            alias.write_bytes(Path(best_run["best_path"]).read_bytes())
            _log.info(
                "Wrote model.zip alias from seed=%s (best_val=%.4f)",
                best_run["seed"], best_run["best_validation_score"],
            )
        except Exception as e:  # pragma: no cover
            _log.warning("failed to write model.zip alias: %s", e)

    summary_path = artifact_root / cfg.name / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    _log.info("Wrote training summary to %s", summary_path)
    return summary


def _run_single_seed(
    cfg: _ExperimentCfg,
    seed: int,
    total_timesteps: int,
    artifact_root: Path,
    provider_name: str,
    n_envs: int = 1,
) -> dict:
    from gymnasium.wrappers import TimeLimit
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    out_dir = artifact_root / cfg.name / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Each parallel env gets a slightly different seed so the rollouts
    # diverge — otherwise SubprocVecEnv with identical seeds is just
    # wasted compute.
    def make_factory(env_idx: int):
        def _factory():
            return Monitor(
                TimeLimit(
                    _build_env(
                        cfg, start=cfg.train_start, end=cfg.train_end,
                        sampler_kind="random",
                        seed=int(seed) * 100 + env_idx,
                        provider_name=provider_name,
                    ),
                    max_episode_steps=512,
                )
            )
        return _factory

    factories = [make_factory(i) for i in range(max(1, n_envs))]
    if n_envs > 1:
        _log.info("seed=%s using SubprocVecEnv with %d parallel envs", seed, n_envs)
        # ``fork`` is critical on Linux runners (Kaggle, Colab, CI):
        # ``spawn`` re-executes the parent script in each child, which
        # would re-run the kaggle_train.py top-level logic (git clone,
        # sys.path mutation) once per worker and crash. ``fork`` shares
        # the parent's already-prepared state and lets closure-based
        # env factories pickle correctly.
        train_env = SubprocVecEnv(factories, start_method="fork")
    else:
        train_env = DummyVecEnv(factories)

    val_env = _build_env(
        cfg, start=cfg.validation_start, end=cfg.validation_end,
        sampler_kind="chronological", seed=seed,
        provider_name=provider_name,
    )

    algo = cfg.algorithm.upper()
    if algo == "PPO":
        model = PPO(
            "MlpPolicy", train_env, seed=seed, verbose=0,
            **cfg.hyperparams,
        )
    elif algo == "DQN":
        model = DQN(
            "MlpPolicy", train_env, seed=seed, verbose=0,
            **cfg.hyperparams,
        )
    elif algo in ("MASKABLEPPO", "MASKABLE_PPO"):
        # FEAT-29: sb3-contrib MaskablePPO uses env.action_masks() to
        # mask out illegal actions BEFORE the policy samples. The
        # selector_v002_masked variant exposes a mask of
        # [True, fired_slot_0, ..., fired_slot_N-1] so the policy
        # cannot select non-fired strategy slots at all.
        from sb3_contrib import MaskablePPO  # type: ignore[import-not-found]
        model = MaskablePPO(
            "MlpPolicy", train_env, seed=seed, verbose=0,
            **cfg.hyperparams,
        )
    else:
        raise ValueError(f"unsupported algorithm: {cfg.algorithm}")

    eval_history: list[dict] = []
    best_score: float = -float("inf")
    best_path = out_dir / "best.zip"
    no_improve = 0

    eval_interval = max(1, cfg.eval_interval_timesteps)
    patience = max(1, cfg.early_stopping_patience)
    min_delta = float(cfg.min_validation_delta)

    class EvalCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.last_eval_step = 0

        def _on_step(self) -> bool:
            nonlocal best_score, no_improve
            if self.num_timesteps - self.last_eval_step < eval_interval:
                return True
            self.last_eval_step = self.num_timesteps
            score, breakdown = _evaluate(
                model, val_env,
                window_start=cfg.validation_start,    # FIX-#56
                window_end=cfg.validation_end,        # FIX-#56
            )
            eval_history.append({
                "timesteps": int(self.num_timesteps),
                "validation_composite_score": score,
                **breakdown,
            })
            if score > best_score + min_delta:
                best_score = score
                no_improve = 0
                model.save(str(best_path))
                _log.info("seed=%s step=%s NEW BEST val=%.4f", seed, self.num_timesteps, score)
            else:
                no_improve += 1
                _log.info(
                    "seed=%s step=%s val=%.4f (best=%.4f, patience=%d/%d)",
                    seed, self.num_timesteps, score, best_score, no_improve, patience,
                )
                if no_improve >= patience:
                    _log.info("Early stopping at step %s", self.num_timesteps)
                    return False
            return True

    callback = EvalCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)

    # Always also save the last model — distinct from best.
    last_path = out_dir / "last.zip"
    model.save(str(last_path))

    if not best_path.exists():
        # No eval ever ran (e.g. eval interval > total steps); use last as best.
        model.save(str(best_path))

    metadata = {
        "seed": seed,
        "best_path": str(best_path),
        "last_path": str(last_path),
        "best_validation_score": best_score if best_score != -float("inf") else None,
        "eval_history": eval_history,
        "total_timesteps_run": int(callback.num_timesteps if eval_history else total_timesteps),
        "trained_at": datetime.utcnow().isoformat(),
        "feature_version": "features_v001_core_daily",
        "algorithm": cfg.algorithm,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    # FIX-#53: do NOT write the top-level model.zip alias here.
    # In a multi-seed run, each seed would overwrite the alias with
    # its own best checkpoint, leaving the LAST seed's model on
    # disk — not the best across seeds. The alias is now written by
    # train_from_experiment after all seeds finish, picking the
    # global best by best_validation_score.

    return metadata


# ---------------------------------------------------------------------
def _evaluate(
    model, env, *,
    window_start: date | None = None,
    window_end: date | None = None,
) -> tuple[float, dict]:
    """One full pass through the chronological validation env.

    FIX-#51: scores against the SAME metric the walk-forward report
    uses (``validation_composite_score_from_daily_pnl``). Pre-FIX-#51
    this used the legacy per-trade ``validation_composite_score``,
    so ``best.zip`` could be selected by a metric the final report
    no longer trusts. Now the entire training-time selection +
    final report agree.

    FIX-#56: ``window_start`` / ``window_end`` should be the
    validation window (cfg.validation_start / cfg.validation_end).
    Idle days are filled with zero P&L so a policy that's flat for
    parts of the window still gets those days included in
    Sharpe / max-DD.
    """
    from rl_swing.rl.validation.metrics import (
        validation_composite_score_from_daily_pnl,
    )
    from rl_swing.rl.validation.portfolio_pnl import TradeRecord

    obs, _ = env.reset()
    rewards: list[float] = []
    actions_taken: list[str] = []
    trade_records: list[TradeRecord] = []

    # FEAT-29: if the policy is a MaskablePPO, route action masks
    # through predict(). Detect by attribute on the model — vanilla
    # PPO/DQN don't accept ``action_masks`` kwarg, but MaskablePPO
    # does and the env exposes ``action_masks()`` regardless.
    is_maskable = type(model).__name__ == "MaskablePPO"

    done = False
    while not done:
        if is_maskable and hasattr(env, "action_masks"):
            mask = env.action_masks()
            action, _ = model.predict(
                obs, deterministic=True, action_masks=mask,
            )
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        rewards.append(float(reward))
        if info.get("action") == "take":
            actions_taken.append("take")
            # FIX-#51 + #36: build TradeRecord for daily-P&L scoring.
            entry = info.get("entry_date")
            exit_ = info.get("exit_date")
            if entry is not None and exit_ is not None:
                trade_records.append(TradeRecord(
                    entry_date=entry, exit_date=exit_,
                    return_pct=float(info.get("net_return", 0.0)),
                    size_pct=float(info.get("size_pct", 0.0)),
                ))
        elif info.get("action") == "skip":
            actions_taken.append("skip")
        done = bool(terminated) or bool(truncated)

    # FIX-#61: derive trading_days from val_env's bars so checkpoint
    # selection uses the SAME calendar walk-forward uses (handles
    # exchange holidays correctly, not just weekday-approx).
    trading_days = None
    bars_attr = getattr(env, "bars", None)
    if bars_attr is not None and hasattr(bars_attr, "by_symbol"):
        td_set = set()
        for sym_bars in bars_attr.by_symbol.values():
            for b in sym_bars:
                td_set.add(b.timestamp.date())
        if window_start is not None and window_end is not None:
            td_set = {d for d in td_set if window_start <= d <= window_end}
        trading_days = sorted(td_set) or None

    score, breakdown = validation_composite_score_from_daily_pnl(
        trades=trade_records,
        n_total_packs=len(actions_taken),
        rewards=rewards,
        actions=actions_taken,
        window_start=window_start,
        window_end=window_end,
        trading_days=trading_days,
    )
    return score, breakdown
