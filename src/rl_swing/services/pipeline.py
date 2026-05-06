"""DecisionPipeline — the 12-step pipeline shared by all runtime modes.

See ADR 0006. This service is the only place that orchestrates the
entire flow. It does not know about specific data providers, brokers,
or RL algorithms — it talks to the ports.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from rl_swing.domain import (
    BrokerOrder,
    CandidateTrade,
    EventType,
    FeatureFrame,
    MarketBar,
    OrderIntent,
    PolicyDecision,
    PortfolioState,
    RiskDecision,
)
from rl_swing.runtime.dependency_container import Container

_log = logging.getLogger(__name__)


@dataclass
class PipelineRunOutput:
    run_id: str
    bars_count: int = 0
    feature_frames: list[FeatureFrame] = field(default_factory=list)
    candidates: list[CandidateTrade] = field(default_factory=list)
    policy_decisions: list[PolicyDecision] = field(default_factory=list)
    risk_decisions: list[RiskDecision] = field(default_factory=list)
    order_intents: list[OrderIntent] = field(default_factory=list)
    broker_orders: list[BrokerOrder] = field(default_factory=list)


class DecisionPipeline:
    def __init__(self, container: Container, *, default_lookback_days: int = 365) -> None:
        self.container = container
        self.lookback_days = int(default_lookback_days)

    # ------------------------------------------------------------------
    def run_once(
        self,
        as_of: date | None = None,
        symbols: list[str] | None = None,
    ) -> PipelineRunOutput:
        cfg = self.container.config
        run_id = self.container.run_id
        as_of = as_of or date.today()
        symbols = symbols or self._resolve_universe(cfg.universe)

        out = PipelineRunOutput(run_id=run_id)

        # 1-3: bars
        start = as_of - timedelta(days=self.lookback_days)
        bars: list[MarketBar] = list(
            self.container.market_data.get_bars(symbols, start, as_of, "1d", True)
        )
        out.bars_count = len(bars)
        self.container.emit(
            EventType.MARKET_DATA_LOADED, run_id,
            {"symbols": symbols, "bars": len(bars)},
        )

        # 4: features
        frames = list(self.container.feature_pipeline.build(bars))
        out.feature_frames = frames
        self.container.emit(
            EventType.FEATURES_BUILT, run_id,
            {"feature_frames": len(frames),
             "feature_version": getattr(self.container.feature_pipeline, "feature_version", "?")},
        )

        # 5: candidates from each strategy
        portfolio_state = self._read_portfolio_state(as_of)
        all_candidates: list[CandidateTrade] = []
        for strat in self.container.strategies:
            cands = list(strat.generate(frames, portfolio_state))
            all_candidates.extend(cands)
            self.container.emit(
                EventType.CANDIDATE_GENERATED, run_id,
                {"strategy_id": getattr(strat, "strategy_id", strat.__class__.__name__),
                 "count": len(cands)},
            )
        out.candidates = all_candidates

        # 6: score candidates
        latest_frames: dict[tuple[str, date], FeatureFrame] = {
            (f.symbol, f.as_of.date()): f for f in frames
        }
        for cand in all_candidates:
            frame = latest_frames.get((cand.symbol, cand.as_of.date()))
            if frame is None:
                _log.debug("no frame for candidate %s — skipping", cand.candidate_id)
                continue
            decision = self.container.policy_scorer.score(cand, frame, portfolio_state)
            out.policy_decisions.append(decision)
            self.container.emit(
                EventType.POLICY_SCORED, decision.candidate_id,
                {"action": decision.action,
                 "model_id": decision.model_id,
                 "target_size_pct": decision.target_size_pct,
                 "confidence": decision.confidence},
            )

        # 7-9: risk + order routing left as no-ops here in the MVP.
        # The walk-forward harness drives risk/execution separately
        # (see rl/validation/walk_forward.py).

        self.container.emit(
            EventType.PIPELINE_COMPLETED, run_id,
            {"candidates": len(out.candidates),
             "policy_decisions": len(out.policy_decisions)},
        )
        return out

    # ------------------------------------------------------------------
    def _resolve_universe(self, universe: str) -> list[str]:
        from pathlib import Path

        import yaml
        candidate_paths = [
            Path("configs/universes") / f"{universe}.yaml",
            Path(universe),
        ]
        for p in candidate_paths:
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                u = data.get("universe") or {}
                return list(u.get("symbols") or [])
        # Fallback: assume the string is a comma-separated list.
        return [s.strip().upper() for s in universe.split(",") if s.strip()]

    def _read_portfolio_state(self, as_of: date) -> PortfolioState:
        try:
            account = self.container.broker.get_account_snapshot()
            positions = tuple(self.container.broker.list_positions())
            return PortfolioState(
                as_of=datetime.combine(as_of, datetime.min.time()),
                cash=account.cash,
                equity=account.equity,
                positions=positions,
                open_positions_count=len(positions),
            )
        except NotImplementedError:
            # Stubbed broker (e.g. AlpacaPaperBrokerAdapter without keys).
            return PortfolioState(
                as_of=datetime.combine(as_of, datetime.min.time()),
                cash=0.0, equity=0.0, positions=(),
                open_positions_count=0,
            )
