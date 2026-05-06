"""Walk-forward validation harness.

Runs an experiment's *test* window through every PolicyScorer baseline
plus the trained model and produces a comparison report. The full
spec calls for multi-cycle walk-forward (train 2014-19/val 20/test 21,
shifted forward each cycle); this implementation does one cycle and
the caller can loop over experiment configs to do multiple. Each
cycle's report is independent.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

import yaml

from rl_swing.adapters.data.synthetic_provider import SyntheticProvider
from rl_swing.domain import (
    CandidateTrade,
    FeatureFrame,
    MarketBar,
    PortfolioState,
)
from rl_swing.features.pipelines import CoreDailyPipeline
from rl_swing.ports import PolicyScorer
from rl_swing.rl.agents.baseline_scorers import (
    AlwaysTakePolicyScorer,
    NeverTakePolicyScorer,
    RandomPolicyScorer,
)
from rl_swing.rl.env.cost_model import EquityExecutionModel
from rl_swing.rl.env.execution_simulator import ExecutionSimulator
from rl_swing.rl.env.reward_model import RewardModel
from rl_swing.rl.validation.baselines import buy_and_hold_return
from rl_swing.rl.validation.metrics import validation_composite_score
from rl_swing.strategies.aggregator import StrategyAggregator
from rl_swing.strategies.breakout import BreakoutStrategy
from rl_swing.strategies.mean_reversion import RsiMeanReversionStrategy
from rl_swing.strategies.momentum import MomentumStrategy

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
def _load_universe(name: str) -> list[str]:
    path = Path("configs/universes") / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        return list((yaml.safe_load(f) or {}).get("universe", {}).get("symbols", []))


def _build_provider(provider_name: str):
    if provider_name.startswith("synthetic_"):
        regime = provider_name.split("_", 1)[1]
        return SyntheticProvider(regime=regime, seed=11)
    if provider_name == "yfinance_daily":
        from rl_swing.adapters.data.yfinance_provider import YFinanceProvider
        return YFinanceProvider()
    if provider_name == "wrds_parquet":
        from rl_swing.adapters.data.wrds_parquet_provider import WrdsParquetProvider
        return WrdsParquetProvider()
    raise ValueError(f"unknown data provider: {provider_name!r}")


# ---------------------------------------------------------------------
def evaluate_policy(
    scorer: PolicyScorer,
    bars: Sequence[MarketBar],
    candidates: Sequence[CandidateTrade],
    feature_frames: Sequence[FeatureFrame],
    *,
    cost_model: EquityExecutionModel,
    reward_model: RewardModel,
    cost_stress_multiplier: float = 1.0,
) -> dict:
    """Replay every candidate through the same pipeline the env would,
    using ``scorer`` as the policy."""
    cost_model.cost_stress_multiplier = float(cost_stress_multiplier)
    sim = ExecutionSimulator()

    by_symbol: dict[str, list[MarketBar]] = {}
    for b in bars:
        by_symbol.setdefault(b.symbol, []).append(b)
    for s in by_symbol:
        by_symbol[s].sort(key=lambda b: b.timestamp)

    frames_by_key = {(f.symbol, f.as_of): f for f in feature_frames}
    portfolio = PortfolioState(
        as_of=datetime.utcnow(), cash=100_000.0, equity=100_000.0,
    )

    rewards: list[float] = []
    net_returns: list[float] = []
    cost_drag_bps: list[float] = []
    holding_days: list[int] = []
    actions: list[str] = []
    decisions: list[dict] = []

    for c in sorted(candidates, key=lambda c: (c.as_of, c.symbol)):
        frame = frames_by_key.get((c.symbol, c.as_of))
        if frame is None:
            continue
        try:
            decision = scorer.score(c, frame, portfolio)
        except FileNotFoundError as e:
            _log.warning("scorer %s missing artifact: %s", scorer.model_id, e)
            return {"error": str(e), "model_id": scorer.model_id}

        action = decision.action
        size_mult = {"skip": 0.0, "take_25": 0.25, "take_50": 0.5, "take_100": 1.0}[action]
        atr_pct = float(frame.values.get("atr_pct_14", 0.02))
        rv20 = float(frame.values.get("realized_vol_20", 0.20))
        adv = float(frame.values.get("dollar_volume", 0.0))
        notional = 100_000.0 * c.base_size_pct * size_mult
        bps = cost_model.cost_bps(
            atr_pct=atr_pct,
            volatility_percentile=min(1.0, rv20 / 0.6),
            in_event_window=False,
            notional=notional,
            avg_dollar_volume=adv,
        )

        bars_for_sym = by_symbol.get(c.symbol) or []
        # binary search for entry index
        from bisect import bisect_left
        timestamps = [b.timestamp for b in bars_for_sym]
        idx = bisect_left(timestamps, c.as_of)
        if idx == len(bars_for_sym) or bars_for_sym[idx].timestamp != c.as_of:
            continue

        outcome = sim.simulate(
            bars=bars_for_sym, entry_index=idx,
            size_pct=c.base_size_pct * size_mult if size_mult > 0 else c.base_size_pct,
            max_holding_days=c.max_holding_days, cost_bps=bps,
            atr_pct=atr_pct,
            starting_equity=100_000.0,
        )

        if size_mult > 0 and outcome is not None:
            r = reward_model.reward_for_take(outcome, c.max_holding_days)
            net_returns.append(outcome.return_pct)
            cost_drag_bps.append(outcome.cost_bps)
            holding_days.append(outcome.holding_days)
            actions.append("take")
            rewards.append(r)
            decisions.append({
                "candidate_id": c.candidate_id, "symbol": c.symbol,
                "as_of": c.as_of.isoformat(),
                "strategy_id": c.strategy_id,
                "action": action,
                "raw_return": outcome.raw_return_pct,
                "net_return": outcome.return_pct,
                "exit_reason": outcome.exit_reason,
                "cost_bps": outcome.cost_bps,
                "holding_days": outcome.holding_days,
            })
        else:
            r = reward_model.reward_for_skip(outcome)
            actions.append("skip")
            rewards.append(r)
            decisions.append({
                "candidate_id": c.candidate_id, "symbol": c.symbol,
                "as_of": c.as_of.isoformat(),
                "strategy_id": c.strategy_id,
                "action": "skip",
                "counterfactual_return": outcome.return_pct if outcome else None,
            })

    score, breakdown = validation_composite_score(
        net_returns=net_returns,
        cost_bps=cost_drag_bps,
        holding_days=holding_days,
        rewards=rewards,
        actions=actions,
    )
    return {
        "model_id": scorer.model_id,
        "validation_composite_score": score,
        **breakdown,
        "cost_stress_multiplier": cost_stress_multiplier,
        "decisions": decisions,
    }


# ---------------------------------------------------------------------
def validate_from_experiment(
    experiment_path: str | Path,
    *,
    model_id: str | None = None,
    report_dir: Path | None = None,
    data_provider_override: str | None = None,
    artifact_root_override: str | Path | None = None,
    include_baselines: tuple[str, ...] = (
        "random", "always_take_100", "always_take_50", "never_take",
    ),
    include_cost_stress: bool = True,
) -> dict:
    """Run walk-forward validation for one experiment cycle."""
    with open(experiment_path, encoding="utf-8") as f:
        exp = (yaml.safe_load(f) or {}).get("experiment") or {}

    universe_name = exp.get("universe", "synthetic")
    test_start = date.fromisoformat(exp["test_start"])
    test_end = date.fromisoformat(exp["test_end"])
    provider_name = data_provider_override or exp.get("data_provider", "synthetic_momentum")

    provider = _build_provider(provider_name)
    symbols = _load_universe(universe_name)
    bars = list(provider.get_bars(symbols, test_start, test_end, "1d", True))
    pipeline = CoreDailyPipeline()
    frames = list(pipeline.build(bars))

    portfolio = PortfolioState(
        as_of=datetime(test_end.year, test_end.month, test_end.day),
        cash=100_000.0, equity=100_000.0,
    )
    # Loose candidate config — must match trainer.py to ensure the
    # trained model is evaluated on the same candidate distribution
    # it learned from.
    candidates = list(StrategyAggregator([
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
    ]).generate(frames, portfolio))

    cost_cfg = exp.get("cost_model") or {}
    cost_model = EquityExecutionModel(**cost_cfg) if cost_cfg else EquityExecutionModel()
    reward_cfg = exp.get("reward") or {}
    reward_model = RewardModel(
        target_risk_pct=0.02,
        drawdown_penalty_weight=reward_cfg.get("drawdown_penalty_weight", 0.10),
        turnover_penalty_weight=reward_cfg.get("turnover_penalty_weight", 0.30),
        holding_period_penalty_weight=reward_cfg.get("holding_period_penalty_weight", 0.05),
        skip_counterfactual_scale=reward_cfg.get("skip_counterfactual_scale", 1.0),
    )

    # Baselines
    scorers: list[PolicyScorer] = []
    if "random" in include_baselines:
        scorers.append(RandomPolicyScorer(model_id="baseline_random", seed=42))
    if "always_take_100" in include_baselines:
        scorers.append(AlwaysTakePolicyScorer(
            model_id="baseline_always_take_100", action="take_100"))
    if "always_take_50" in include_baselines:
        scorers.append(AlwaysTakePolicyScorer(
            model_id="baseline_always_take_50", action="take_50"))
    if "never_take" in include_baselines:
        scorers.append(NeverTakePolicyScorer(model_id="baseline_never_take"))

    # Trained model. Resolution order:
    #   1. caller-supplied ``artifact_root_override`` (Kaggle/Colab pass
    #      this in, since they write to /kaggle/working/artifacts).
    #   2. experiment YAML's ``artifact_root``.
    #   3. ``data/models/`` (local dev default).
    if artifact_root_override is not None:
        artifact_root = Path(artifact_root_override) / exp["name"]
    else:
        artifact_root = Path(exp.get("artifact_root", "data/models/")) / exp["name"]
    model_artifact = artifact_root / "model.zip"
    rl_added = False
    if model_artifact.exists():
        from rl_swing.rl.agents.dqn_scorer import DqnPolicyScorer
        from rl_swing.rl.agents.ppo_scorer import PpoPolicyScorer
        AlgoCls = PpoPolicyScorer if exp["algorithm"].upper() == "PPO" else DqnPolicyScorer
        scorers.append(AlgoCls(
            model_id=model_id or exp["name"],
            artifact_path=str(model_artifact),
            feature_version="features_v001_core_daily",
        ))
        rl_added = True

    results = []
    for s in scorers:
        res = evaluate_policy(
            s, bars, candidates, frames,
            cost_model=cost_model, reward_model=reward_model,
            cost_stress_multiplier=1.0,
        )
        results.append(res)
        if include_cost_stress:
            res2 = evaluate_policy(
                s, bars, candidates, frames,
                cost_model=cost_model, reward_model=reward_model,
                cost_stress_multiplier=2.0,
            )
            res2["model_id"] = res2["model_id"] + "_cost2x"
            results.append(res2)

    # Buy-and-hold (per benchmark symbol, if present)
    bnh: dict[str, float] = {}
    for sym in ("SPY", "QQQ"):
        if sym in symbols:
            bnh[sym] = buy_and_hold_return(bars, sym, test_start, test_end)

    summary = {
        "experiment": exp["name"],
        "test_start": test_start.isoformat(),
        "test_end": test_end.isoformat(),
        "n_candidates": len(candidates),
        "rl_model_present": rl_added,
        "buy_and_hold": bnh,
        "policies": [
            {k: v for k, v in r.items() if k != "decisions"} for r in results
        ],
    }

    report_dir = Path(report_dir) if report_dir else Path("data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"walkforward_{exp['name']}_{test_start.isoformat()}_{test_end.isoformat()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    _log.info("Wrote walk-forward report to %s", out_path)
    return summary
