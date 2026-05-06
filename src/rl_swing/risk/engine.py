"""RiskEngine — composes a stack of ``RiskPolicy`` rules.

The engine is the *only* place that converts (candidate, policy
decision) into a final approve/deny. It walks rules in order; the
first ``approves=False`` rule blocks. Multiple rules can scale size
down, multiplying their multipliers together.
"""
from __future__ import annotations

import importlib
import uuid
from collections.abc import Iterable
from pathlib import Path

import yaml

from rl_swing.domain import (
    CandidateTrade,
    PolicyDecision,
    PortfolioState,
    RiskDecision,
    RiskRuleResult,
)
from rl_swing.ports.risk_policy import MarketState, RiskPolicy


class RiskEngine:
    def __init__(self, policies: Iterable[RiskPolicy]) -> None:
        self.policies: list[RiskPolicy] = list(policies)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RiskEngine:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        profile = data.get("risk_profile") or data
        policies = []
        for entry in profile.get("policies", []):
            cls_path = entry["class"]
            params = entry.get("params") or {}
            module_path, _, class_name = cls_path.rpartition(".")
            module = importlib.import_module(module_path)
            policy_cls = getattr(module, class_name)
            policies.append(policy_cls(**params))
        return cls(policies)

    def evaluate(
        self,
        candidate: CandidateTrade,
        policy_decision: PolicyDecision,
        portfolio_state: PortfolioState,
        market_state: MarketState,
    ) -> RiskDecision:
        size_multiplier = 1.0
        applied: list[str] = []
        blocked: list[str] = []
        for rule in self.policies:
            result: RiskRuleResult = rule.evaluate(
                candidate, policy_decision, portfolio_state, market_state
            )
            applied.append(result.rule_id)
            if not result.approves:
                blocked.append(result.block_reason or result.rule_id)
                # First block stops the trade. We continue iteration so the
                # blocked_reasons list captures every offending rule, which
                # makes diagnostics easier.
                continue
            if result.size_multiplier < 1.0:
                size_multiplier *= max(0.0, result.size_multiplier)

        approved = len(blocked) == 0 and policy_decision.target_size_pct > 0
        final_size = (
            policy_decision.target_size_pct * size_multiplier if approved else 0.0
        )
        return RiskDecision(
            decision_id=f"risk-{uuid.uuid4().hex[:12]}",
            candidate_id=candidate.candidate_id,
            policy_decision_id=policy_decision.decision_id,
            approved=approved,
            final_size_pct=final_size,
            blocked_reasons=tuple(blocked),
            applied_rules=tuple(applied),
        )
