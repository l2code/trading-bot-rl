"""Policy and risk decision types.

The contract is: ``PolicyDecision`` proposes; ``RiskDecision`` disposes.
The risk engine has the final word — ``approved=False`` blocks the
trade regardless of what the policy said.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

PolicyAction = Literal["skip", "take_25", "take_50", "take_100"]

# Mapping action -> size multiplier. ActionMapper / RiskService both rely
# on this single source of truth.
ACTION_TO_SIZE: dict[PolicyAction, float] = {
    "skip": 0.0,
    "take_25": 0.25,
    "take_50": 0.50,
    "take_100": 1.00,
}


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    decision_id: str
    candidate_id: str
    as_of: datetime
    model_id: str
    action: PolicyAction
    confidence: float | None
    target_size_pct: float           # candidate.base_size_pct * action multiplier
    raw_action: int | float          # whatever came out of the network
    observation_hash: str
    explanation: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RiskDecision:
    decision_id: str
    candidate_id: str
    policy_decision_id: str
    approved: bool
    final_size_pct: float            # may be 0 even if approved (e.g. dust)
    blocked_reasons: tuple[str, ...] = ()
    applied_rules: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RiskRuleResult:
    """One rule's vote within the risk engine.

    Rules can either block outright or scale the size down. They never
    scale size up — only the candidate/policy can request larger size.
    """

    rule_id: str
    approves: bool
    size_multiplier: float = 1.0     # 0..1; 1.0 = no change
    block_reason: str | None = None
    note: str | None = None
