"""Behavioral-cloning trainer for v003 chronological env (FEAT-32 M2).

Confirms the v003 env is learnable by fitting a small supervised
classifier to imitate a non-trivial state-dependent target policy.

Workflow:
  1. Build the v003 env on the experiment's TRAIN window (yfinance).
  2. Roll out ``BCTargetPortfolioPolicy`` end-to-end across multiple
     episodes (one per random window in the train range), collect
     (obs, action) pairs.
  3. Split (obs, action) train/val by held-out windows.
  4. Fit ``sklearn.ensemble.HistGradientBoostingClassifier`` with
     fixed seed and small max_iter for fast convergence.
  5. Report top-1 action accuracy + per-action confusion matrix on
     the held-out split.
  6. Save bundle at data/models/portfolio_baseline_bc/model.joblib;
     auto-included in v003 evaluate when artifact exists.

Acceptance (per the M2 plan):
  - ≥70% top-1 held-out accuracy → M2 PASS, env is learnable.
  - ≤50% → M2 FAIL, surface and pause.

Local-only; no Kaggle compute. Wall-time target: ~2-5 min on Loki.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from rl_swing.adapters.data.synthetic_provider import SyntheticProvider  # noqa: E402
from rl_swing.adapters.data.yfinance_provider import YFinanceProvider  # noqa: E402
from rl_swing.domain import PortfolioState  # noqa: E402
from rl_swing.features.pipelines import CoreDailyPipeline  # noqa: E402
from rl_swing.rl.agents.portfolio_baselines import (  # noqa: E402
    BCTargetPortfolioPolicy,
)
from rl_swing.rl.env.chronological_swing_env import ChronologicalSwingEnv  # noqa: E402
from rl_swing.rl.env.cost_model import EquityExecutionModel  # noqa: E402
from rl_swing.rl.env.reward_model import RewardModel  # noqa: E402
from rl_swing.strategies.breakout import BreakoutStrategy  # noqa: E402
from rl_swing.strategies.mean_reversion import RsiMeanReversionStrategy  # noqa: E402
from rl_swing.strategies.momentum import MomentumStrategy  # noqa: E402
from rl_swing.strategies.multi_strategy_packer import MultiStrategyPacker  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger("train_bc_v003")

_FEATURE_WARMUP_DAYS = int(252 * 1.5)


def _build_provider(name: str):
    if name == "yfinance_daily":
        return YFinanceProvider(auto_adjust=True)
    if name.startswith("synthetic_"):
        regime = name.replace("synthetic_", "")
        return SyntheticProvider(regime=regime, seed=11)
    raise ValueError(f"unknown data provider: {name}")


def _load_universe(name: str) -> list[str]:
    path = _REPO_ROOT / "configs" / "universes" / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    nested = (cfg.get("universe") or {}).get("symbols") or []
    flat = cfg.get("symbols") or []
    return list(nested or flat)


def _build_default_strategies():
    return [
        MomentumStrategy(min_relative_strength=-0.05, min_r20=-0.02, require_sma200_above=False),
        RsiMeanReversionStrategy(rsi_threshold=35.0),
        BreakoutStrategy(min_relative_volume=0.7, max_distance_below_high=-0.02),
    ]


def _collect_dataset(
    *, env: ChronologicalSwingEnv, target: BCTargetPortfolioPolicy,
    n_episodes: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out the target through the env across n_episodes random
    windows and collect (obs, action) pairs."""
    X: list[np.ndarray] = []
    y: list[int] = []
    rng = np.random.default_rng(seed)
    for _ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        obs, _info = env.reset(seed=ep_seed)
        done = False
        while not done:
            action = target.decide(obs)
            X.append(obs.copy())
            y.append(int(action))
            obs, _r, terminated, truncated, _info = env.step(action)
            done = bool(terminated) or bool(truncated)
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True)
    p.add_argument("--data-provider", default=None)
    p.add_argument("--n-train-episodes", type=int, default=80)
    p.add_argument("--n-val-episodes", type=int, default=20)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--output", default="data/models/portfolio_baseline_bc/model.joblib")
    args = p.parse_args()

    with open(args.experiment, encoding="utf-8") as f:
        exp = (yaml.safe_load(f) or {}).get("experiment") or {}
    universe = exp.get("universe", "starter_equities")
    provider_name = args.data_provider or exp.get("data_provider", "synthetic_momentum")
    train_start = date.fromisoformat(exp["train_start"])
    train_end = date.fromisoformat(exp["train_end"])
    cost_cfg = exp.get("cost_model") or {}
    reward_cfg = exp.get("reward") or {}
    max_top_k = int(exp.get("v003_max_top_k", 2))
    sampler_window = int(exp.get("v003_sampler_window_days", 60))

    # 1) Load bars + frames + packs over the train range with warmup.
    warmup_start = train_start - timedelta(days=_FEATURE_WARMUP_DAYS)
    provider = _build_provider(provider_name)
    symbols = _load_universe(universe)
    _log.info("loading bars provider=%s universe=%s window=%s..%s",
              provider_name, universe, train_start, train_end)
    t0 = time.time()
    bars = list(provider.get_bars(symbols, warmup_start, train_end, "1d", True))
    _log.info("loaded %d bars in %.1fs", len(bars), time.time() - t0)
    pipeline = CoreDailyPipeline()
    all_frames = list(pipeline.build(bars))
    frames = [f for f in all_frames if train_start <= f.as_of.date() <= train_end]
    portfolio = PortfolioState(
        as_of=datetime(train_end.year, train_end.month, train_end.day),
        cash=100_000.0, equity=100_000.0,
    )
    packer = MultiStrategyPacker(_build_default_strategies())
    packs = packer.pack(frames, portfolio)
    _log.info("built %d frames + %d packs", len(frames), len(packs))

    cost_model = EquityExecutionModel(**cost_cfg) if cost_cfg else EquityExecutionModel()
    reward_model = RewardModel(
        target_risk_pct=0.02,
        drawdown_penalty_weight=reward_cfg.get("drawdown_penalty_weight", 0.10),
        turnover_penalty_weight=reward_cfg.get("turnover_penalty_weight", 0.05),
        holding_period_penalty_weight=reward_cfg.get("holding_period_penalty_weight", 0.02),
        skip_counterfactual_scale=reward_cfg.get("skip_counterfactual_scale", 1.0),
    )

    n_actions = 1 + max_top_k
    target = BCTargetPortfolioPolicy(n_actions=n_actions)

    # 2) Collect train + val datasets via random windowed rollouts.
    env = ChronologicalSwingEnv(
        bars=bars, packs=packs, feature_frames=frames,
        cost_model=cost_model, reward_model=reward_model,
        max_top_k=max_top_k,
        sampler_kind="random", sampler_seed=args.seed,
        sampler_window_days=sampler_window,
    )
    _log.info("collecting train dataset (%d episodes)...", args.n_train_episodes)
    t0 = time.time()
    X_train, y_train = _collect_dataset(
        env=env, target=target, n_episodes=args.n_train_episodes, seed=args.seed,
    )
    _log.info("train: %d examples in %.1fs; class distribution: %s",
              len(y_train), time.time() - t0,
              {int(c): int((y_train == c).sum()) for c in np.unique(y_train)})

    _log.info("collecting val dataset (%d episodes, different seed)...", args.n_val_episodes)
    X_val, y_val = _collect_dataset(
        env=env, target=target, n_episodes=args.n_val_episodes,
        seed=args.seed + 1000,
    )
    _log.info("val: %d examples; class distribution: %s",
              len(y_val), {int(c): int((y_val == c).sum()) for c in np.unique(y_val)})

    # 3) Train.
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(
        max_iter=args.max_iter, max_depth=args.max_depth, random_state=args.seed,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    _log.info("fit took %.1fs", time.time() - t0)

    # 4) Evaluate.
    val_pred = model.predict(X_val)
    val_acc = float((val_pred == y_val).mean())
    train_acc = float((model.predict(X_train) == y_train).mean())
    _log.info("train top-1 accuracy: %.4f", train_acc)
    _log.info("val top-1 accuracy:   %.4f", val_acc)

    # Per-action confusion (predicted vs actual on val).
    confusion = np.zeros((n_actions, n_actions), dtype=int)
    for actual, pred in zip(y_val, val_pred, strict=False):
        if 0 <= int(actual) < n_actions and 0 <= int(pred) < n_actions:
            confusion[int(actual), int(pred)] += 1
    _log.info("val confusion (rows=actual, cols=predicted):")
    for i in range(n_actions):
        _log.info("  actual=%d  %s", i, list(map(int, confusion[i])))

    # M2 verdict per the plan.
    if val_acc >= 0.70:
        verdict = "PASS"
    elif val_acc <= 0.50:
        verdict = "FAIL"
    else:
        verdict = "MARGINAL"
    _log.info("M2 verdict: %s (val_acc=%.4f vs PASS threshold 0.70)", verdict, val_acc)

    # 5) Persist bundle.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    bundle = {
        "model": model,
        "n_actions": int(n_actions),
        "trained_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
        "n_train_examples": int(len(y_train)),
        "n_val_examples": int(len(y_val)),
        "train_acc": train_acc,
        "val_acc": val_acc,
        "verdict": verdict,
        "confusion_val": confusion.tolist(),
        "data_provider": provider_name,
        "universe": universe,
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "max_top_k": max_top_k,
        "max_iter": args.max_iter,
        "max_depth": args.max_depth,
        "seed": args.seed,
    }
    joblib.dump(bundle, str(out_path))
    meta_path = out_path.with_suffix(".metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in bundle.items() if k != "model"}, f, indent=2, default=str)
    _log.info("wrote %s and %s", out_path, meta_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
