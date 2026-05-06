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
from rl_swing.rl.env.cost_model import EquityExecutionModel
from rl_swing.rl.env.execution_simulator import ExecutionSimulator
from rl_swing.rl.env.reward_model import RewardModel
from rl_swing.rl.validation.baselines import buy_and_hold_return
from rl_swing.rl.validation.metrics import validation_composite_score

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
    # FIX-#36: TradeRecords for date-ordered daily-P&L metrics.
    from rl_swing.rl.validation.portfolio_pnl import TradeRecord
    trade_records: list[TradeRecord] = []

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
            trade_records.append(TradeRecord(
                entry_date=outcome.entry_timestamp.date(),
                exit_date=outcome.exit_timestamp.date(),
                return_pct=outcome.return_pct,
                size_pct=c.base_size_pct * size_mult,
            ))
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

    # FIX-#36: primary metrics now come from the date-ordered daily-
    # P&L path. Legacy per-trade metrics still computed for the
    # ``legacy_*`` keys so we can A/B and verify the magnitude of
    # the change.
    from rl_swing.rl.validation.metrics import (
        validation_composite_score_from_daily_pnl,
    )
    score, breakdown = validation_composite_score_from_daily_pnl(
        trades=trade_records,
        n_total_packs=len(actions),
        rewards=rewards,
        actions=actions,
    )
    legacy_score, legacy_breakdown = validation_composite_score(
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
        "legacy_per_trade_score": legacy_score,
        "legacy_per_trade_breakdown": {
            k: legacy_breakdown[k]
            for k in ("total_return", "annualized_sharpe", "profit_factor",
                      "max_drawdown", "turnover_take_rate")
        },
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
    """Run walk-forward validation for one experiment cycle.

    Dispatches to the experiment's RL variant for env-specific
    aggregation + evaluation; this function only handles the common
    work (loading bars/frames, resolving the artifact path, picking
    benchmarks, writing the report).
    """
    from rl_swing.rl.variants import EvaluationContext
    from rl_swing.rl.variants.base import load_variant

    with open(experiment_path, encoding="utf-8") as f:
        exp = (yaml.safe_load(f) or {}).get("experiment") or {}

    universe_name = exp.get("universe", "synthetic")
    test_start = date.fromisoformat(exp["test_start"])
    test_end = date.fromisoformat(exp["test_end"])
    provider_name = data_provider_override or exp.get("data_provider", "synthetic_momentum")

    provider = _build_provider(provider_name)
    symbols = _load_universe(universe_name)
    # FIX-24: load bars with warmup before test_start so long-lookback
    # features (sma_200, return_60d, atr_pct_14, etc.) are populated
    # on the first test day. Without warmup, ~200 of the 252 test
    # trading days had degraded features and silently corrupted every
    # diary's metrics.
    from rl_swing.rl.training.trainer import (
        _filter_frames_to_window,
        _load_bars_with_warmup,
    )
    bars, _warmup_start = _load_bars_with_warmup(provider, symbols, test_start, test_end)
    pipeline = CoreDailyPipeline()
    all_frames = list(pipeline.build(bars))
    # Filter frames to the test window so candidates only fire in
    # [test_start, test_end]. Bars stay full so the simulator can
    # access prior-day context for ATR / stop calculations.
    frames = _filter_frames_to_window(all_frames, test_start, test_end)

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

    # Resolve artifact path:
    #   1. caller-supplied ``artifact_root_override`` (Kaggle/Colab).
    #   2. experiment YAML's ``artifact_root``.
    #   3. ``data/models/`` (local dev default).
    if artifact_root_override is not None:
        artifact_root = Path(artifact_root_override) / exp["name"]
    else:
        artifact_root = Path(exp.get("artifact_root", "data/models/")) / exp["name"]
    model_artifact: Path | None = artifact_root / "model.zip"
    rl_added = bool(model_artifact and model_artifact.exists())

    # Dispatch to the variant for evaluation. Variants own how to map
    # the ``include_baselines`` tuple to their action space — filter
    # variants take {random, always_take_100, always_take_50,
    # never_take}, selector variants take {random, always_skip,
    # first_fired, highest_signal}.
    variant_name = exp.get("rl_variant", "filter_v001")
    variant = load_variant(variant_name)
    eval_ctx = EvaluationContext(
        bars=bars, frames=frames,
        test_start=test_start, test_end=test_end,
        cost_model=cost_model, reward_model=reward_model,
        artifact_path=model_artifact,
        model_id=model_id or exp["name"],
        include_baselines=tuple(include_baselines),
        include_cost_stress=include_cost_stress,
        experiment_config=dict(exp),
    )
    policy_results = variant.evaluate(eval_ctx)

    # Buy-and-hold (per benchmark symbol, if present)
    bnh: dict[str, float] = {}
    for sym in ("SPY", "QQQ"):
        if sym in symbols:
            bnh[sym] = buy_and_hold_return(bars, sym, test_start, test_end)

    # ``n_candidates`` interpretation differs across variants —
    # filter_v001 reports deduped candidates, selector_v002 reports
    # packs. We surface what the variant actually saw.
    n_units = sum(1 for r in policy_results if r.cost_stress_multiplier == 1.0)
    summary = {
        "experiment": exp["name"],
        "rl_variant": variant_name,
        "test_start": test_start.isoformat(),
        "test_end": test_end.isoformat(),
        "n_policies": n_units,
        "rl_model_present": rl_added,
        "buy_and_hold": bnh,
        "policies": [r.to_dict() for r in policy_results],
    }

    report_dir = Path(report_dir) if report_dir else Path("data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"walkforward_{exp['name']}_{test_start.isoformat()}_{test_end.isoformat()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    _log.info("Wrote walk-forward report to %s", out_path)
    return summary
