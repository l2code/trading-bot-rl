"""Skip-threshold sweep for the supervised set ranker (FEAT-34 PR-1c / B1b).

Operator's B1b list, item 1: 'sweep skip_threshold'. The cheapest
intervention — needs no retraining, just varies the inference-time
threshold that controls how often the scorer skips.

Workflow (no test-set peeking):
  1. Load the trained set ranker artifact (PR-1b stabilized).
  2. Sweep skip_threshold ∈ {-1.5, -1.25, ..., +1.5} on the
     **validation** window (2021 yfinance) — out-of-sample to
     the test 2022 window.
  3. For each threshold compute the Phase-24 gate vs
     selector_baseline_random on validation.
  4. Pick the threshold that clears the gate (or has the best
     gate result if none clear).
  5. Run a single out-of-sample test on 2022 with that locked
     threshold; report the gate verdict.

Usage:
    python scripts/sweep_set_ranker_threshold.py \\
        --experiment configs/experiments/ppo_selector_v002_masked.yaml \\
        --data-provider yfinance_daily

Out: prints the sweep table + the test-2022 verdict at the picked
threshold; also writes a JSON sidecar with the full sweep.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from rl_swing.adapters.data.synthetic_provider import SyntheticProvider  # noqa: E402
from rl_swing.adapters.data.yfinance_provider import YFinanceProvider  # noqa: E402
from rl_swing.domain import PortfolioState  # noqa: E402
from rl_swing.features.pipelines import CoreDailyPipeline  # noqa: E402
from rl_swing.rl.agents.selector_scorers import RandomSelectorScorer  # noqa: E402
from rl_swing.rl.agents.set_ranker_scorer import SetRankerSelectorScorer  # noqa: E402
from rl_swing.rl.env.cost_model import EquityExecutionModel  # noqa: E402
from rl_swing.rl.env.execution_simulator import ExecutionSimulator  # noqa: E402
from rl_swing.rl.env.reward_model import RewardModel  # noqa: E402
from rl_swing.rl.validation.acceptance_gate import evaluate_gate  # noqa: E402
from rl_swing.strategies.breakout import BreakoutStrategy  # noqa: E402
from rl_swing.strategies.mean_reversion import RsiMeanReversionStrategy  # noqa: E402
from rl_swing.strategies.momentum import MomentumStrategy  # noqa: E402
from rl_swing.strategies.multi_strategy_packer import MultiStrategyPacker  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger("sweep_threshold")

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


def _evaluate_one_scorer_window(
    scorer, packs, frames_by_key, by_symbol, cost_model, sim, reward_model, n_slots,
    window_start: date, window_end: date, bars,
):
    """Mirror of selector_v002._evaluate_scorer. Returns the
    PolicyResult-equivalent dict that acceptance_gate can consume."""
    from rl_swing.rl.validation.metrics import (
        validation_composite_score_from_daily_pnl,
    )
    from rl_swing.rl.validation.portfolio_pnl import TradeRecord

    rewards: list[float] = []
    net_returns: list[float] = []
    cost_drag_bps: list[float] = []
    holding_days: list[int] = []
    actions_taken: list[str] = []
    per_strategy_take_counts = [0] * n_slots
    trade_records: list[TradeRecord] = []
    portfolio = PortfolioState(
        as_of=datetime.utcnow(), cash=100_000.0, equity=100_000.0,
    )

    for pack in sorted(packs, key=lambda p: (p.as_of, p.symbol)):
        frame = frames_by_key.get((pack.symbol, pack.as_of))
        if frame is None:
            continue
        action = scorer.select(pack, frame, portfolio)
        if action == 0:
            # Skip counterfactual (highest_signal mode)
            chosen = None
            best_strength = -1.0
            for c in pack.candidates:
                if c is None:
                    continue
                if c.signal_strength > best_strength:
                    chosen = c
                    best_strength = c.signal_strength
            cf = None
            if chosen is not None:
                cf = _simulate_take(chosen, frame, by_symbol, cost_model, sim)
            reward = reward_model.reward_for_skip(cf)
            rewards.append(reward)
            actions_taken.append("skip")
            continue
        idx = action - 1
        chosen = pack.candidates[idx] if 0 <= idx < n_slots else None
        if chosen is None:
            rewards.append(0.0)
            actions_taken.append("skip")
            continue
        outcome = _simulate_take(chosen, frame, by_symbol, cost_model, sim)
        if outcome is None:
            rewards.append(0.0)
            actions_taken.append("skip")
            continue
        reward = reward_model.reward_for_take(
            outcome, max_holding_days=chosen.max_holding_days,
        )
        rewards.append(reward)
        net_returns.append(outcome.return_pct)
        cost_drag_bps.append(outcome.cost_bps)
        holding_days.append(outcome.holding_days)
        actions_taken.append("take")
        per_strategy_take_counts[idx] += 1
        trade_records.append(TradeRecord(
            entry_date=outcome.entry_timestamp.date(),
            exit_date=outcome.exit_timestamp.date(),
            return_pct=outcome.return_pct,
            size_pct=chosen.base_size_pct,
        ))

    trading_days = sorted({
        b.timestamp.date() for b in bars
        if window_start <= b.timestamp.date() <= window_end
    })
    score, breakdown = validation_composite_score_from_daily_pnl(
        trades=trade_records, n_total_packs=len(actions_taken),
        rewards=rewards, actions=actions_taken,
        window_start=window_start, window_end=window_end,
        trading_days=trading_days or None,
    )
    return {
        "model_id": scorer.model_id,
        "validation_composite_score": float(score),
        "n_trades": int(breakdown.get("n_trades", 0)),
        "total_return": float(breakdown.get("total_return", 0.0)),
        "annualized_sharpe": float(breakdown.get("annualized_sharpe", 0.0)),
        "profit_factor": float(breakdown.get("profit_factor", 0.0)),
        "max_drawdown": float(breakdown.get("max_drawdown", 0.0)),
        "turnover_take_rate": float(breakdown.get("turnover_take_rate", 0.0)),
        "per_strategy_take_counts": list(per_strategy_take_counts),
    }


def _simulate_take(candidate, frame, by_symbol, cost_model, sim):
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


def _build_window(provider, symbols, start: date, end: date, cost_cfg, reward_cfg):
    """Load bars+frames+packs for one window. Returns the bundle
    needed to evaluate scorers on it."""
    from datetime import timedelta
    warmup_start = start - timedelta(days=_FEATURE_WARMUP_DAYS)
    bars = list(provider.get_bars(symbols, warmup_start, end, "1d", True))
    pipeline = CoreDailyPipeline()
    frames = [
        f for f in pipeline.build(bars)
        if start <= f.as_of.date() <= end
    ]
    frames_by_key = {(f.symbol, f.as_of): f for f in frames}
    portfolio = PortfolioState(
        as_of=datetime(end.year, end.month, end.day),
        cash=100_000.0, equity=100_000.0,
    )
    packer = MultiStrategyPacker(_build_default_strategies())
    packs = packer.pack(frames, portfolio)
    by_symbol: dict = {}
    for b in bars:
        by_symbol.setdefault(b.symbol, []).append(b)
    for sym in by_symbol:
        by_symbol[sym].sort(key=lambda b: b.timestamp)
    cost_model = EquityExecutionModel(**cost_cfg) if cost_cfg else EquityExecutionModel()
    reward_model = RewardModel(
        target_risk_pct=0.02,
        drawdown_penalty_weight=reward_cfg.get("drawdown_penalty_weight", 0.10),
        turnover_penalty_weight=reward_cfg.get("turnover_penalty_weight", 0.05),
        holding_period_penalty_weight=reward_cfg.get("holding_period_penalty_weight", 0.02),
        skip_counterfactual_scale=reward_cfg.get("skip_counterfactual_scale", 1.0),
    )
    sim = ExecutionSimulator()
    return {
        "bars": bars, "packs": packs, "frames_by_key": frames_by_key,
        "by_symbol": by_symbol, "cost_model": cost_model,
        "reward_model": reward_model, "sim": sim,
        "n_slots": packer.n_slots,
        "start": start, "end": end,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True)
    p.add_argument("--data-provider", default=None)
    p.add_argument("--ranker-artifact",
                   default="data/models/selector_baseline_set_ranker/model.pt")
    p.add_argument("--thresholds", type=str,
                   default="-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0,1.25,1.5",
                   help="Comma-separated skip_threshold values to sweep on val 2021.")
    p.add_argument("--out", default="data/reports/set_ranker_threshold_sweep.json")
    args = p.parse_args()

    with open(args.experiment, encoding="utf-8") as f:
        exp = (yaml.safe_load(f) or {}).get("experiment") or {}
    universe = exp.get("universe", "starter_equities")
    provider_name = args.data_provider or exp.get("data_provider", "synthetic_momentum")
    val_start = date.fromisoformat(exp["validation_start"])
    val_end = date.fromisoformat(exp["validation_end"])
    test_start = date.fromisoformat(exp["test_start"])
    test_end = date.fromisoformat(exp["test_end"])
    cost_cfg = exp.get("cost_model") or {}
    reward_cfg = exp.get("reward") or {}
    provider = _build_provider(provider_name)
    symbols = _load_universe(universe)

    _log.info("loading val 2021 window...")
    t0 = time.time()
    val = _build_window(provider, symbols, val_start, val_end, cost_cfg, reward_cfg)
    _log.info("val packs=%d (loaded in %.1fs)", len(val["packs"]), time.time() - t0)

    # Random baseline on val 2021 — single source of truth for the gate.
    random_v = RandomSelectorScorer(seed=42)
    _log.info("evaluating random baseline on val 2021...")
    val_random_row = _evaluate_one_scorer_window(
        random_v, val["packs"], val["frames_by_key"], val["by_symbol"],
        val["cost_model"], val["sim"], val["reward_model"], val["n_slots"],
        val_start, val_end, val["bars"],
    )
    _log.info("val random: score=%.4f ret=%+.4f sharpe=%+.3f dd=%.4f trades=%d",
              val_random_row["validation_composite_score"], val_random_row["total_return"],
              val_random_row["annualized_sharpe"], val_random_row["max_drawdown"],
              val_random_row["n_trades"])

    # Sweep on val 2021.
    thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]
    sweep_rows = []
    for thr in thresholds:
        scorer = SetRankerSelectorScorer(
            artifact_path=args.ranker_artifact,
            n_strategies=val["n_slots"],
            skip_threshold=thr,
        )
        row = _evaluate_one_scorer_window(
            scorer, val["packs"], val["frames_by_key"], val["by_symbol"],
            val["cost_model"], val["sim"], val["reward_model"], val["n_slots"],
            val_start, val_end, val["bars"],
        )
        gate = evaluate_gate(row, val_random_row)
        sweep_rows.append({"threshold": thr, "row": row, "gate": gate.to_dict()})
        _log.info(
            "thr=%+5.2f score=%.4f ret=%+.4f sharpe=%+.3f dd=%.4f take=%.4f "
            "trades=%4d per_strat=%s | gate=%s improved=%d mat_regress=%d",
            thr, row["validation_composite_score"], row["total_return"],
            row["annualized_sharpe"], row["max_drawdown"], row["turnover_take_rate"],
            row["n_trades"], row["per_strategy_take_counts"],
            gate.verdict, gate.n_improved, gate.n_regressed_materially,
        )

    # Pick best threshold by gate verdict.
    # Priority: GO > SHADOW_ONLY > NO_GO. Among same-verdict rows,
    # pick the one with most improved metrics, then lowest n_regressed_materially,
    # then highest composite as a tiebreaker.
    verdict_rank = {"GO": 3, "SHADOW_ONLY": 2, "NO_GO": 1}
    def sort_key(s):
        g = s["gate"]
        return (
            verdict_rank.get(g["verdict"], 0),
            g["n_improved"],
            -g["n_regressed_materially"],
            s["row"]["validation_composite_score"],
        )
    best = max(sweep_rows, key=sort_key)
    _log.info(
        "BEST val threshold=%+.2f verdict=%s improved=%d material_regress=%d",
        best["threshold"], best["gate"]["verdict"],
        best["gate"]["n_improved"], best["gate"]["n_regressed_materially"],
    )

    # Out-of-sample: evaluate the locked threshold on test 2022.
    _log.info("loading test 2022 window for OOS eval...")
    t0 = time.time()
    test = _build_window(provider, symbols, test_start, test_end, cost_cfg, reward_cfg)
    _log.info("test packs=%d (loaded in %.1fs)", len(test["packs"]), time.time() - t0)
    random_t = RandomSelectorScorer(seed=42)
    test_random_row = _evaluate_one_scorer_window(
        random_t, test["packs"], test["frames_by_key"], test["by_symbol"],
        test["cost_model"], test["sim"], test["reward_model"], test["n_slots"],
        test_start, test_end, test["bars"],
    )
    test_scorer = SetRankerSelectorScorer(
        artifact_path=args.ranker_artifact,
        n_strategies=test["n_slots"],
        skip_threshold=best["threshold"],
    )
    test_row = _evaluate_one_scorer_window(
        test_scorer, test["packs"], test["frames_by_key"], test["by_symbol"],
        test["cost_model"], test["sim"], test["reward_model"], test["n_slots"],
        test_start, test_end, test["bars"],
    )
    test_gate = evaluate_gate(test_row, test_random_row)
    _log.info(
        "TEST 2022 OOS at threshold=%+.2f: score=%.4f ret=%+.4f sharpe=%+.3f "
        "dd=%.4f take=%.4f trades=%d per_strat=%s | gate=%s improved=%d mat_regress=%d",
        best["threshold"], test_row["validation_composite_score"], test_row["total_return"],
        test_row["annualized_sharpe"], test_row["max_drawdown"], test_row["turnover_take_rate"],
        test_row["n_trades"], test_row["per_strategy_take_counts"],
        test_gate.verdict, test_gate.n_improved, test_gate.n_regressed_materially,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": str(args.experiment),
            "data_provider": provider_name,
            "ranker_artifact": args.ranker_artifact,
            "val_window": [val_start.isoformat(), val_end.isoformat()],
            "test_window": [test_start.isoformat(), test_end.isoformat()],
            "val_random": val_random_row,
            "test_random": test_random_row,
            "sweep": sweep_rows,
            "best_threshold": best["threshold"],
            "best_val_verdict": best["gate"]["verdict"],
            "test_oos_at_best_threshold": {
                "row": test_row,
                "gate": test_gate.to_dict(),
            },
        }, f, indent=2, default=str)
    _log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
