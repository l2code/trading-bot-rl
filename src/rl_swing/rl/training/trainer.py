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
from datetime import date, datetime
from pathlib import Path

import yaml

from rl_swing.adapters.data.synthetic_provider import SyntheticProvider
from rl_swing.features.pipelines import ALL_FEATURE_NAMES, CoreDailyPipeline
from rl_swing.rl.env.cost_model import EquityExecutionModel
from rl_swing.rl.env.reward_model import RewardModel
from rl_swing.rl.env.swing_env import SwingTradingEnv

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
) -> SwingTradingEnv:
    from rl_swing.domain import PortfolioState
    from rl_swing.strategies.aggregator import StrategyAggregator
    from rl_swing.strategies.breakout import BreakoutStrategy
    from rl_swing.strategies.mean_reversion import RsiMeanReversionStrategy
    from rl_swing.strategies.momentum import MomentumStrategy

    provider = _build_provider(provider_name)
    symbols = _load_universe_symbols(cfg.universe)
    bars = list(provider.get_bars(symbols, start, end, "1d", True))

    pipeline = CoreDailyPipeline()
    frames = list(pipeline.build(bars))

    portfolio = PortfolioState(
        as_of=datetime(end.year, end.month, end.day),
        cash=100_000.0, equity=100_000.0,
    )
    strategies = [
        MomentumStrategy(),
        RsiMeanReversionStrategy(),
        BreakoutStrategy(),
    ]
    candidates = list(StrategyAggregator(strategies).generate(frames, portfolio))

    cost = EquityExecutionModel(**cfg.cost_model) if cfg.cost_model else EquityExecutionModel()
    reward = RewardModel(
        target_risk_pct=0.02,
        drawdown_penalty_weight=cfg.reward.get("drawdown_penalty_weight", 0.10),
        turnover_penalty_weight=cfg.reward.get("turnover_penalty_weight", 0.02),
        holding_period_penalty_weight=cfg.reward.get("holding_period_penalty_weight", 0.05),
    )

    return SwingTradingEnv(
        bars=bars,
        candidates=candidates,
        feature_frames=frames,
        feature_names=ALL_FEATURE_NAMES,
        sampler_kind=sampler_kind,
        sampler_seed=seed,
        sampler_window_days=120,
        cost_model=cost,
        reward_model=reward,
    )


# ---------------------------------------------------------------------
def train_from_experiment(
    experiment_path: str | Path,
    *,
    total_timesteps_override: int | None = None,
    seed_override: int | None = None,
    data_provider_override: str | None = None,
    artifact_root_override: str | None = None,
    n_envs: int = 1,
) -> dict:
    cfg = _ExperimentCfg.from_yaml(experiment_path)
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
        train_env = SubprocVecEnv(factories, start_method="spawn")
    else:
        train_env = DummyVecEnv(factories)

    val_env = _build_env(
        cfg, start=cfg.validation_start, end=cfg.validation_end,
        sampler_kind="chronological", seed=seed,
        provider_name=provider_name,
    )

    if cfg.algorithm.upper() == "PPO":
        model = PPO(
            "MlpPolicy", train_env, seed=seed, verbose=0,
            **cfg.hyperparams,
        )
    elif cfg.algorithm.upper() == "DQN":
        model = DQN(
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
            score, breakdown = _evaluate(model, val_env)
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

    # Also write a top-level model.zip alias pointing at the best
    # checkpoint, which is what the registry/scorer adapters expect.
    alias = artifact_root / cfg.name / "model.zip"
    try:
        if alias.exists():
            alias.unlink()
        alias.write_bytes(best_path.read_bytes())
    except Exception as e:  # pragma: no cover
        _log.warning("failed to write model.zip alias: %s", e)

    return metadata


# ---------------------------------------------------------------------
def _evaluate(model, env) -> tuple[float, dict]:
    """One full pass through the chronological validation env."""
    from rl_swing.rl.validation.metrics import validation_composite_score

    obs, _ = env.reset()
    rewards: list[float] = []
    raw_returns: list[float] = []
    cost_drag_bps: list[float] = []
    holding_days: list[int] = []
    actions_taken: list[str] = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        rewards.append(float(reward))
        if info.get("action") == "take":
            raw_returns.append(float(info.get("net_return", 0.0)))
            cost_drag_bps.append(float(info.get("cost_bps", 0.0)))
            holding_days.append(int(info.get("holding_days", 0)))
            actions_taken.append("take")
        elif info.get("action") == "skip":
            actions_taken.append("skip")
        done = bool(terminated) or bool(truncated)

    score, breakdown = validation_composite_score(
        net_returns=raw_returns,
        cost_bps=cost_drag_bps,
        holding_days=holding_days,
        rewards=rewards,
        actions=actions_taken,
    )
    return score, breakdown
