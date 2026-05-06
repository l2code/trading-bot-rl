"""SelectorV002Variant — v2 multi-strategy selector.

Per-(symbol, date) decisions: agent sees the full slate of strategy
proposals and chooses among ``Discrete(N+1)`` actions (skip or take
strategy k). Strategy proposals are kept separate (no dedupe), so the
agent can condition on cross-strategy agreement.

Baselines:
    selector_baseline_random              — random fired strategy or skip
    selector_baseline_always_skip         — always 0
    selector_baseline_first_fired         — first fired strategy
    selector_baseline_highest_signal      — fired strategy with max signal_strength
                                            (matches v1's pre-dedupe winner)
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime

import gymnasium as gym

from rl_swing.domain import (
    CandidateTrade,
    MarketBar,
    PortfolioState,
)
from rl_swing.features.pipelines import ALL_FEATURE_NAMES
from rl_swing.rl.agents.selector_scorers import (
    AlwaysFirstFiredSelectorScorer,
    AlwaysSkipSelectorScorer,
    HighestSignalSelectorScorer,
    PpoSelectorScorer,
    RandomSelectorScorer,
    SelectorScorer,
)
from rl_swing.rl.env.cost_model import EquityExecutionModel
from rl_swing.rl.env.execution_simulator import ExecutionSimulator
from rl_swing.rl.env.multi_strategy_env import MultiStrategySwingTradingEnv
from rl_swing.rl.validation.metrics import validation_composite_score
from rl_swing.rl.variants.base import (
    EnvBuildContext,
    EvaluationContext,
    PolicyResult,
)
from rl_swing.strategies.breakout import BreakoutStrategy
from rl_swing.strategies.mean_reversion import RsiMeanReversionStrategy
from rl_swing.strategies.momentum import MomentumStrategy
from rl_swing.strategies.multi_strategy_packer import (
    MultiStrategyPacker,
    StrategyPack,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
def _build_default_strategies() -> list:
    """Same loose defaults as the v1 filter, so candidate sets match
    1:1 — that gives us a clean apples-to-apples comparison.
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


def _pack_candidates(frames, portfolio):
    packer = MultiStrategyPacker(_build_default_strategies())
    packs = packer.pack(frames, portfolio)
    return packs, packer.n_slots


# ---------------------------------------------------------------------
class SelectorV002Variant:
    name: str = "selector_v002"

    # ---- env -----------------------------------------------------
    def build_env(self, ctx: EnvBuildContext) -> gym.Env:
        packs, n_slots = _pack_candidates(ctx.frames, ctx.portfolio)
        # FIX-26: skip_counterfactual_mode controls how the skip
        # reward's counterfactual is chosen. Default (and recommended
        # for new experiments) is "highest_signal" — uses prior info
        # only, no hindsight peek. The legacy "max" mode has a max-
        # over-noise bias that grows with N strategies.
        reward_cfg = (ctx.experiment_config or {}).get("reward") or {}
        skip_cf_mode = reward_cfg.get(
            "skip_counterfactual_mode", "highest_signal"
        )
        return MultiStrategySwingTradingEnv(
            bars=ctx.bars,
            packs=packs,
            feature_frames=ctx.frames,
            feature_names=ALL_FEATURE_NAMES,
            n_strategies=n_slots,
            sampler_kind=ctx.sampler_kind,
            sampler_seed=ctx.seed,
            sampler_window_days=120,
            cost_model=ctx.cost_model,
            reward_model=ctx.reward_model,
            skip_counterfactual_mode=skip_cf_mode,
        )

    # ---- evaluation ---------------------------------------------
    def evaluate(self, ctx: EvaluationContext) -> list[PolicyResult]:
        portfolio = PortfolioState(
            as_of=datetime(ctx.test_end.year, ctx.test_end.month, ctx.test_end.day),
            cash=100_000.0, equity=100_000.0,
        )
        packs, n_slots = _pack_candidates(ctx.frames, portfolio)

        scorers: list[SelectorScorer] = []
        if "random" in ctx.include_baselines:
            scorers.append(RandomSelectorScorer(seed=42))
        if "always_skip" in ctx.include_baselines or "never_take" in ctx.include_baselines:
            scorers.append(AlwaysSkipSelectorScorer())
        if "first_fired" in ctx.include_baselines:
            scorers.append(AlwaysFirstFiredSelectorScorer())
        if "highest_signal" in ctx.include_baselines:
            scorers.append(HighestSignalSelectorScorer())

        rl_added = False
        if ctx.artifact_path is not None and ctx.artifact_path.exists():
            scorers.append(PpoSelectorScorer(
                model_id=ctx.model_id,
                artifact_path=str(ctx.artifact_path),
                n_strategies=n_slots,
            ))
            rl_added = True

        results: list[PolicyResult] = []
        for s in scorers:
            res = self._evaluate_scorer(
                s, packs, ctx, n_slots, cost_stress_multiplier=1.0,
            )
            results.append(res)
            if ctx.include_cost_stress:
                res2 = self._evaluate_scorer(
                    s, packs, ctx, n_slots, cost_stress_multiplier=2.0,
                )
                results.append(PolicyResult(
                    **{**res2.to_dict(), "model_id": res2.model_id + "_cost2x"}
                ))

        for r in results:
            r.extras.setdefault("rl_model_present", rl_added)
        return results

    # ---- internals ----------------------------------------------
    def _evaluate_scorer(
        self, scorer: SelectorScorer,
        packs: Sequence[StrategyPack],
        ctx: EvaluationContext,
        n_slots: int,
        *,
        cost_stress_multiplier: float,
    ) -> PolicyResult:
        ctx.cost_model.cost_stress_multiplier = float(cost_stress_multiplier)
        sim = ExecutionSimulator()

        # FIX-26: mirror the train-time env's skip_counterfactual_mode
        # so eval-time skip rewards use the same convention.
        reward_cfg = (ctx.experiment_config or {}).get("reward") or {}
        skip_cf_mode = reward_cfg.get("skip_counterfactual_mode", "highest_signal")

        by_symbol: dict[str, list[MarketBar]] = {}
        for b in ctx.bars:
            by_symbol.setdefault(b.symbol, []).append(b)
        for sym in by_symbol:
            by_symbol[sym].sort(key=lambda b: b.timestamp)
        frames_by_key = {(f.symbol, f.as_of): f for f in ctx.frames}
        portfolio = PortfolioState(
            as_of=datetime.utcnow(), cash=100_000.0, equity=100_000.0,
        )

        rewards: list[float] = []
        net_returns: list[float] = []
        cost_drag_bps: list[float] = []
        holding_days: list[int] = []
        actions: list[str] = []
        per_strategy_take_counts = [0] * n_slots
        # FIX-#36: TradeRecords for date-ordered daily-P&L metrics.
        from rl_swing.rl.validation.portfolio_pnl import TradeRecord
        trade_records: list[TradeRecord] = []

        for pack in sorted(packs, key=lambda p: (p.as_of, p.symbol)):
            frame = frames_by_key.get((pack.symbol, pack.as_of))
            if frame is None:
                continue
            try:
                action = scorer.select(pack, frame, portfolio)
            except FileNotFoundError as e:
                _log.warning("scorer %s missing artifact: %s", scorer.model_id, e)
                return PolicyResult(
                    model_id=scorer.model_id,
                    n_trades=0, total_return=0.0, annualized_sharpe=0.0,
                    profit_factor=0.0, max_drawdown=0.0,
                    turnover_take_rate=0.0, mean_reward=0.0,
                    validation_composite_score=0.0,
                    components={}, cost_stress_multiplier=cost_stress_multiplier,
                )
            if action == 0:
                cf = self._best_counterfactual(
                    pack, frame, by_symbol, ctx.cost_model, sim,
                    mode=skip_cf_mode,
                )
                reward = ctx.reward_model.reward_for_skip(cf)
                rewards.append(reward)
                actions.append("skip")
                continue
            idx = action - 1
            chosen = pack.candidates[idx] if 0 <= idx < n_slots else None
            if chosen is None:
                rewards.append(0.0)
                actions.append("skip")  # treat illegal as skip for metrics
                continue
            outcome = self._simulate_take(
                chosen, frame, by_symbol, ctx.cost_model, sim,
            )
            if outcome is None:
                rewards.append(0.0)
                actions.append("skip")
                continue
            reward = ctx.reward_model.reward_for_take(
                outcome, max_holding_days=chosen.max_holding_days,
            )
            rewards.append(reward)
            net_returns.append(outcome.return_pct)
            cost_drag_bps.append(outcome.cost_bps)
            holding_days.append(outcome.holding_days)
            actions.append("take")
            per_strategy_take_counts[idx] += 1
            trade_records.append(TradeRecord(
                entry_date=outcome.entry_timestamp.date(),
                exit_date=outcome.exit_timestamp.date(),
                return_pct=outcome.return_pct,
                size_pct=chosen.base_size_pct,
            ))

        # FIX-#36: primary score is daily-P&L based.
        # FIX-#52: idle days fill as zero so Sharpe/DD aren't biased.
        from rl_swing.rl.validation.metrics import (
            validation_composite_score_from_daily_pnl,
        )
        if trade_records:
            win_start = min(t.entry_date for t in trade_records)
            win_end = max(t.exit_date for t in trade_records)
        else:
            win_start = win_end = None
        score, breakdown = validation_composite_score_from_daily_pnl(
            trades=trade_records,
            n_total_packs=len(actions),
            rewards=rewards,
            actions=actions,
            window_start=win_start,
            window_end=win_end,
        )
        legacy_score, _legacy_breakdown = validation_composite_score(
            net_returns=net_returns,
            cost_bps=cost_drag_bps,
            holding_days=holding_days,
            rewards=rewards,
            actions=actions,
        )

        extras = {
            "per_strategy_take_counts": list(per_strategy_take_counts),
            "metric_basis": breakdown.get("metric_basis"),
            "legacy_per_trade_score": float(legacy_score),
            "n_trading_days": breakdown.get("n_trading_days", 0),
        }

        return PolicyResult(
            model_id=scorer.model_id,
            n_trades=int(breakdown.get("n_trades", 0)),
            total_return=float(breakdown.get("total_return", 0.0)),
            annualized_sharpe=float(breakdown.get("annualized_sharpe", 0.0)),
            profit_factor=float(breakdown.get("profit_factor", 0.0)),
            max_drawdown=float(breakdown.get("max_drawdown", 0.0)),
            turnover_take_rate=float(breakdown.get("turnover_take_rate", 0.0)),
            mean_reward=float(breakdown.get("mean_reward", 0.0)),
            validation_composite_score=float(score),
            components=dict(breakdown.get("components", {})),
            cost_stress_multiplier=float(cost_stress_multiplier),
            extras=extras,
        )

    # ---- simulation helpers (mirror v1's per-candidate path) ---
    def _simulate_take(
        self, candidate: CandidateTrade, frame, by_symbol,
        cost_model: EquityExecutionModel, sim: ExecutionSimulator,
    ):
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
        # Find index by timestamp
        idx = -1
        for i, b in enumerate(bars):
            if b.timestamp == candidate.as_of:
                idx = i
                break
        return sim.simulate(
            bars=bars, entry_index=idx,
            size_pct=candidate.base_size_pct,
            max_holding_days=candidate.max_holding_days,
            cost_bps=cost_bps, atr_pct=atr_pct,
            starting_equity=100_000.0,
        )

    def _best_counterfactual(
        self, pack, frame, by_symbol,
        cost_model: EquityExecutionModel, sim: ExecutionSimulator,
        mode: str = "highest_signal",
    ):
        """FIX-26: mode-aware skip counterfactual. Mirrors the env's
        ``_skip_counterfactual`` semantics so train-time and eval-time
        skip rewards use the same convention. Default
        ``highest_signal`` uses prior info only (no hindsight peek)."""
        if mode == "none":
            return None
        if mode == "highest_signal":
            chosen = None
            best_strength = -1.0
            for c in pack.candidates:
                if c is None:
                    continue
                if c.signal_strength > best_strength:
                    chosen = c
                    best_strength = c.signal_strength
            if chosen is None:
                return None
            return self._simulate_take(chosen, frame, by_symbol, cost_model, sim)
        if mode == "max":
            best = None
            for c in pack.candidates:
                if c is None:
                    continue
                outcome = self._simulate_take(c, frame, by_symbol, cost_model, sim)
                if outcome is None:
                    continue
                if best is None or outcome.return_pct > best.return_pct:
                    best = outcome
            return best
        if mode == "mean":
            outcomes = []
            for c in pack.candidates:
                if c is None:
                    continue
                o = self._simulate_take(c, frame, by_symbol, cost_model, sim)
                if o is not None:
                    outcomes.append(o)
            if not outcomes:
                return None
            from dataclasses import replace
            mean_ret = sum(o.return_pct for o in outcomes) / len(outcomes)
            mean_asset = sum(o.asset_return_pct for o in outcomes) / len(outcomes)
            return replace(
                outcomes[0],
                return_pct=mean_ret,
                raw_return_pct=mean_asset,
                asset_return_pct=mean_asset,
            )
        raise ValueError(f"unknown skip_counterfactual_mode: {mode!r}")
