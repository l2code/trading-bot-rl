"""Risk policies — composable rules that the risk engine evaluates.

Each rule returns a ``RiskRuleResult``:
    * ``approves=False`` blocks the trade.
    * ``size_multiplier < 1`` scales it down.
    * Rules never raise size.

The order of policies in the configured profile matters because
blocked reasons are accumulated in evaluation order.
"""
from __future__ import annotations

from rl_swing.domain import (
    CandidateTrade,
    PolicyDecision,
    PortfolioState,
    RiskRuleResult,
)
from rl_swing.ports.risk_policy import MarketState


class _Base:
    rule_id: str

    def __init__(self, rule_id: str) -> None:
        self.rule_id = rule_id


class MaxSinglePositionPolicy(_Base):
    """No single position may exceed ``max_pct`` of equity."""

    def __init__(self, rule_id: str, max_pct: float) -> None:
        super().__init__(rule_id)
        self.max_pct = float(max_pct)

    def evaluate(
        self,
        candidate: CandidateTrade,
        policy_decision: PolicyDecision,
        portfolio_state: PortfolioState,
        market_state: MarketState,
    ) -> RiskRuleResult:
        target = policy_decision.target_size_pct
        if target <= self.max_pct:
            return RiskRuleResult(rule_id=self.rule_id, approves=True, size_multiplier=1.0)
        if self.max_pct <= 0:
            return RiskRuleResult(
                rule_id=self.rule_id, approves=False,
                block_reason=f"max_single_position={self.max_pct:.2%}",
            )
        ratio = self.max_pct / target
        return RiskRuleResult(
            rule_id=self.rule_id,
            approves=True,
            size_multiplier=min(1.0, ratio),
            note=f"scaled to max_pct={self.max_pct:.2%}",
        )


class MaxPortfolioExposurePolicy(_Base):
    def __init__(self, rule_id: str, max_pct: float) -> None:
        super().__init__(rule_id)
        self.max_pct = float(max_pct)

    def evaluate(
        self,
        candidate: CandidateTrade,
        policy_decision: PolicyDecision,
        portfolio_state: PortfolioState,
        market_state: MarketState,
    ) -> RiskRuleResult:
        existing = portfolio_state.gross_exposure_pct
        proposed = existing + policy_decision.target_size_pct
        if proposed <= self.max_pct:
            return RiskRuleResult(rule_id=self.rule_id, approves=True)
        headroom = max(0.0, self.max_pct - existing)
        if headroom <= 0:
            return RiskRuleResult(
                rule_id=self.rule_id, approves=False,
                block_reason=f"max_portfolio_exposure={self.max_pct:.2%}",
            )
        if policy_decision.target_size_pct <= 0:
            return RiskRuleResult(rule_id=self.rule_id, approves=True)
        return RiskRuleResult(
            rule_id=self.rule_id,
            approves=True,
            size_multiplier=min(1.0, headroom / policy_decision.target_size_pct),
            note=f"scaled to leave gross_exposure<={self.max_pct:.2%}",
        )


class MaxDailyLossPolicy(_Base):
    def __init__(self, rule_id: str, max_pct: float) -> None:
        super().__init__(rule_id)
        self.max_pct = float(max_pct)

    def evaluate(
        self,
        candidate: CandidateTrade,
        policy_decision: PolicyDecision,
        portfolio_state: PortfolioState,
        market_state: MarketState,
    ) -> RiskRuleResult:
        if portfolio_state.daily_loss_pct >= self.max_pct:
            return RiskRuleResult(
                rule_id=self.rule_id, approves=False,
                block_reason=f"daily_loss>={self.max_pct:.2%}",
            )
        return RiskRuleResult(rule_id=self.rule_id, approves=True)


class MaxOpenPositionsPolicy(_Base):
    def __init__(self, rule_id: str, max_positions: int) -> None:
        super().__init__(rule_id)
        self.max_positions = int(max_positions)

    def evaluate(
        self,
        candidate: CandidateTrade,
        policy_decision: PolicyDecision,
        portfolio_state: PortfolioState,
        market_state: MarketState,
    ) -> RiskRuleResult:
        if portfolio_state.position_for(candidate.symbol) is not None:
            return RiskRuleResult(rule_id=self.rule_id, approves=True)
        if portfolio_state.open_positions_count >= self.max_positions:
            return RiskRuleResult(
                rule_id=self.rule_id, approves=False,
                block_reason=f"max_open_positions={self.max_positions}",
            )
        return RiskRuleResult(rule_id=self.rule_id, approves=True)


class MaxDailyNewPositionsPolicy(_Base):
    """Limits NEW positions opened today (not size; count)."""

    def __init__(self, rule_id: str, max_positions: int) -> None:
        super().__init__(rule_id)
        self.max_positions = int(max_positions)
        self._opened_today: int = 0
        self._today: str | None = None

    def evaluate(
        self,
        candidate: CandidateTrade,
        policy_decision: PolicyDecision,
        portfolio_state: PortfolioState,
        market_state: MarketState,
    ) -> RiskRuleResult:
        today = candidate.as_of.date().isoformat()
        if today != self._today:
            self._today = today
            self._opened_today = 0
        already_held = portfolio_state.position_for(candidate.symbol) is not None
        if already_held or policy_decision.target_size_pct <= 0:
            return RiskRuleResult(rule_id=self.rule_id, approves=True)
        if self._opened_today >= self.max_positions:
            return RiskRuleResult(
                rule_id=self.rule_id, approves=False,
                block_reason=f"max_daily_new_positions={self.max_positions}",
            )
        self._opened_today += 1
        return RiskRuleResult(rule_id=self.rule_id, approves=True)


class LiquidityPolicy(_Base):
    """Blocks trades on illiquid symbols.

    Reads ``avg_dollar_volume`` from the candidate's metadata. The
    candidate strategy is responsible for populating this; missing means
    "unknown", which we treat as a soft pass with a note.
    """

    def __init__(self, rule_id: str, min_avg_dollar_volume: float) -> None:
        super().__init__(rule_id)
        self.min_avg_dollar_volume = float(min_avg_dollar_volume)

    def evaluate(
        self,
        candidate: CandidateTrade,
        policy_decision: PolicyDecision,
        portfolio_state: PortfolioState,
        market_state: MarketState,
    ) -> RiskRuleResult:
        adv = float(candidate.metadata.get("avg_dollar_volume", -1.0))
        if adv < 0:
            return RiskRuleResult(
                rule_id=self.rule_id, approves=True,
                note="liquidity unknown — treated as pass",
            )
        if adv < self.min_avg_dollar_volume:
            return RiskRuleResult(
                rule_id=self.rule_id, approves=False,
                block_reason=f"liquidity_below_min adv={adv:.0f}",
            )
        return RiskRuleResult(rule_id=self.rule_id, approves=True)


class DuplicateOrderPolicy(_Base):
    """Blocks two orders for the same candidate within a short window.

    The risk engine populates ``portfolio_state.open_positions_count``
    and the per-rule state below tracks recent candidate ids.
    """

    def __init__(self, rule_id: str) -> None:
        super().__init__(rule_id)
        self._seen: set[str] = set()

    def evaluate(
        self,
        candidate: CandidateTrade,
        policy_decision: PolicyDecision,
        portfolio_state: PortfolioState,
        market_state: MarketState,
    ) -> RiskRuleResult:
        if candidate.candidate_id in self._seen:
            return RiskRuleResult(
                rule_id=self.rule_id, approves=False,
                block_reason="duplicate_candidate",
            )
        self._seen.add(candidate.candidate_id)
        return RiskRuleResult(rule_id=self.rule_id, approves=True)


class KillSwitchPolicy(_Base):
    def evaluate(
        self,
        candidate: CandidateTrade,
        policy_decision: PolicyDecision,
        portfolio_state: PortfolioState,
        market_state: MarketState,
    ) -> RiskRuleResult:
        if market_state.is_kill_switch_active:
            return RiskRuleResult(
                rule_id=self.rule_id, approves=False,
                block_reason="kill_switch_active",
            )
        return RiskRuleResult(rule_id=self.rule_id, approves=True)


class LiveTradingApprovalPolicy(_Base):
    """In live mode, requires the ``RL_SWING_LIVE_APPROVAL_TOKEN`` env."""

    def evaluate(
        self,
        candidate: CandidateTrade,
        policy_decision: PolicyDecision,
        portfolio_state: PortfolioState,
        market_state: MarketState,
    ) -> RiskRuleResult:
        import os
        if not os.environ.get("RL_SWING_LIVE_APPROVAL_TOKEN"):
            return RiskRuleResult(
                rule_id=self.rule_id, approves=False,
                block_reason="missing_live_approval_token",
            )
        return RiskRuleResult(rule_id=self.rule_id, approves=True)
