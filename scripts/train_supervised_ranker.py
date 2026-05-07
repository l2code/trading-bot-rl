"""Offline trainer for the supervised ranker baseline (FEAT-30 / #30).

Walks the training window, simulates every fired (pack × strategy)
trade with the same ExecutionSimulator + cost model the v2 env uses,
records realized risk-adjusted returns as the regression target,
fits HistGradientBoostingRegressor, saves a joblib artifact + a
metadata.json side-by-side.

Usage:
    python scripts/train_supervised_ranker.py \\
        --experiment configs/experiments/ppo_selector_v002_masked.yaml \\
        --data-provider yfinance_daily \\
        --output data/models/selector_baseline_supervised/model.joblib

The artifact is then auto-discovered by the v002 / v002_masked
variants' evaluate() pipeline when ``"supervised"`` is in
``include_baselines`` (the default).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import yaml

# Make the repo's src tree importable when run as a plain script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from rl_swing.adapters.data.synthetic_provider import SyntheticProvider  # noqa: E402
from rl_swing.adapters.data.yfinance_provider import YFinanceProvider  # noqa: E402
from rl_swing.domain import PortfolioState  # noqa: E402
from rl_swing.features.pipelines import CoreDailyPipeline  # noqa: E402
from rl_swing.rl.agents.supervised_ranker_scorer import (  # noqa: E402
    PER_SLOT_FEATURE_NAMES,
    build_slot_features,
    write_metadata,
)
from rl_swing.rl.env.cost_model import EquityExecutionModel  # noqa: E402
from rl_swing.rl.env.execution_simulator import ExecutionSimulator  # noqa: E402
from rl_swing.strategies.breakout import BreakoutStrategy  # noqa: E402
from rl_swing.strategies.mean_reversion import RsiMeanReversionStrategy  # noqa: E402
from rl_swing.strategies.momentum import MomentumStrategy  # noqa: E402
from rl_swing.strategies.multi_strategy_packer import MultiStrategyPacker  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger("train_supervised_ranker")


# Calendar-day warmup so long-lookback features (sma_200, etc.)
# populate from the first in-window day. Mirrors trainer.py's
# FIX-#24 default.
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
    if not path.exists():
        raise FileNotFoundError(f"universe not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # YAML nests symbols under top-level "universe" key; fall back to
    # flat layout for forward-compat.
    nested = (cfg.get("universe") or {}).get("symbols") or []
    flat = cfg.get("symbols") or []
    return list(nested or flat)


def _build_default_strategies():
    """Same loose-config strategies as selector_v002 — keep in sync."""
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


def _simulate_outcome(candidate, frame, by_symbol, cost_model, sim):
    """Mirror of selector_v002._simulate_take so the training labels
    are computed by the SAME simulator + cost layer the eval-time
    environment uses. Any drift between train labels and eval
    rewards = the ranker silently optimizing the wrong target."""
    atr_pct = float(frame.values.get("atr_pct_14", 0.02))
    rv20 = float(frame.values.get("realized_vol_20", 0.20))
    vol_percentile = min(1.0, max(0.0, rv20 / 0.6))
    adv = float(frame.values.get("dollar_volume", 0.0))
    notional = 100_000.0 * candidate.base_size_pct
    cost_bps = cost_model.cost_bps(
        atr_pct=atr_pct, volatility_percentile=vol_percentile,
        in_event_window=False, notional=notional, avg_dollar_volume=adv,
    )
    bars = by_symbol.get(candidate.symbol) or []
    idx = -1
    for i, b in enumerate(bars):
        if b.timestamp == candidate.as_of:
            idx = i
            break
    if idx < 0:
        return None
    return sim.simulate(
        bars=bars, entry_index=idx,
        size_pct=candidate.base_size_pct,
        max_holding_days=candidate.max_holding_days,
        cost_bps=cost_bps, atr_pct=atr_pct,
        starting_equity=100_000.0,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True,
                   help="Experiment YAML (provides universe + train_start/end + cost_model).")
    p.add_argument("--data-provider", default=None,
                   help="Override data provider (e.g., yfinance_daily).")
    p.add_argument("--output", default="data/models/selector_baseline_supervised/model.joblib")
    p.add_argument("--target-risk-pct", type=float, default=0.02,
                   help="Denominator for risk_adj_return = net_return / target_risk_pct.")
    p.add_argument("--max-iter", type=int, default=200,
                   help="HistGradientBoostingRegressor max_iter.")
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()

    with open(args.experiment, encoding="utf-8") as f:
        exp = (yaml.safe_load(f) or {}).get("experiment") or {}

    universe = exp.get("universe", "starter_equities")
    provider_name = args.data_provider or exp.get("data_provider", "synthetic_momentum")
    train_start = date.fromisoformat(exp["train_start"])
    train_end = date.fromisoformat(exp["train_end"])

    cost_cfg = exp.get("cost_model") or {}
    cost_model = EquityExecutionModel(**cost_cfg)
    sim = ExecutionSimulator()

    # 1) Load bars with warmup so long-lookback features are populated.
    from datetime import timedelta
    warmup_start = train_start - timedelta(days=_FEATURE_WARMUP_DAYS)
    provider = _build_provider(provider_name)
    symbols = _load_universe(universe)
    _log.info("loading bars provider=%s universe=%s symbols=%d window=%s..%s (warmup=%s)",
              provider_name, universe, len(symbols), train_start, train_end, warmup_start)
    t0 = time.time()
    bars = list(provider.get_bars(symbols, warmup_start, train_end, "1d", True))
    _log.info("loaded %d bars in %.1fs", len(bars), time.time() - t0)

    # 2) Build features. Filter frames to the in-window region.
    pipeline = CoreDailyPipeline()
    all_frames = list(pipeline.build(bars))
    frames = [f for f in all_frames if train_start <= f.as_of.date() <= train_end]
    frames_by_key = {(f.symbol, f.as_of): f for f in frames}
    _log.info("built %d feature frames in window", len(frames))

    # 3) Pack candidates.
    portfolio = PortfolioState(
        as_of=datetime(train_end.year, train_end.month, train_end.day),
        cash=100_000.0, equity=100_000.0,
    )
    packer = MultiStrategyPacker(_build_default_strategies())
    packs = packer.pack(frames, portfolio)
    n_slots = packer.n_slots
    _log.info("packed %d packs (n_slots=%d)", len(packs), n_slots)

    # 4) For each fired (pack, slot), simulate the trade and emit a
    #    training row. Skip slots whose simulator returns None
    #    (boundary edge cases — same as the v2 env).
    by_symbol: dict[str, list] = {}
    for b in bars:
        by_symbol.setdefault(b.symbol, []).append(b)
    for sym in by_symbol:
        by_symbol[sym].sort(key=lambda b: b.timestamp)

    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    n_skipped_no_sim = 0
    for pack in packs:
        frame = frames_by_key.get((pack.symbol, pack.as_of))
        if frame is None:
            continue
        for slot_idx, candidate in enumerate(pack.candidates):
            if candidate is None:
                continue
            outcome = _simulate_outcome(candidate, frame, by_symbol, cost_model, sim)
            if outcome is None:
                n_skipped_no_sim += 1
                continue
            X_rows.append(build_slot_features(frame, slot_idx, candidate))
            y_rows.append(float(outcome.return_pct) / float(args.target_risk_pct))

    if not X_rows:
        _log.error("no training rows — abort")
        return 1
    X = np.vstack(X_rows)
    y = np.asarray(y_rows, dtype=np.float64)
    _log.info("training set: X=%s y=%s mean(y)=%.4f std(y)=%.4f skipped_no_sim=%d",
              X.shape, y.shape, float(np.mean(y)), float(np.std(y)), n_skipped_no_sim)

    # 5) Fit. HistGradientBoostingRegressor: fast on CPU, deterministic
    #    with random_state, no GPU required.
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        random_state=args.seed,
    )
    t0 = time.time()
    model.fit(X, y)
    _log.info("fit took %.1fs", time.time() - t0)

    # Quick in-sample sanity (NOT a generalization claim; just plumbing).
    y_hat = model.predict(X)
    in_sample_mse = float(np.mean((y - y_hat) ** 2))
    _log.info("in-sample MSE: %.5f", in_sample_mse)

    # 6) Persist artifact + metadata.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib  # type: ignore[import-untyped]
    bundle = {
        "model": model,
        "feature_names": PER_SLOT_FEATURE_NAMES,
        "n_strategies": n_slots,
        "target_risk_pct": float(args.target_risk_pct),
        "trained_at": datetime.utcnow().isoformat(),
        "n_train_examples": int(X.shape[0]),
        "in_sample_mse": in_sample_mse,
        "data_provider": provider_name,
        "universe": universe,
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
    }
    joblib.dump(bundle, str(out_path))
    write_metadata(out_path.with_suffix(".metadata.json"), {
        k: v for k, v in bundle.items() if k != "model"
    })
    _log.info("wrote %s and %s", out_path, out_path.with_suffix(".metadata.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
