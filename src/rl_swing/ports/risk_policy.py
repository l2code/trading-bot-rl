"""RiskPolicy port.

A ``RiskEngine`` composes many ``RiskPolicy`` rules. Rules can block
outright or scale size down. They never raise size — only the policy/
candidate request larger size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rl_swing.domain import (
    CandidateTrade,
    PolicyDecision,
    PortfolioState,
    RiskRuleResult,
)


@dataclass(frozen=True, slots=True)
class MarketState:
    """Lightweight market view passed to risk rules.

    Some rules (kill switch, daily loss) need market-level context.
    """
    is_market_open: bool
    is_kill_switch_active: bool = False
    daily_pnl_pct: float = 0.0
    weekly_pnl_pct: float = 0.0


@runtime_checkable
class RiskPolicy(Protocol):
    rule_id: str

    def evaluate(
        self,
        candidate: CandidateTrade,
        policy_decision: PolicyDecision,
        portfolio_state: PortfolioState,
        market_state: MarketState,
    ) -> RiskRuleResult: ...
