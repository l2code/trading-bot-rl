"""Baseline policy scorers — policies as `PolicyScorer`s, not separate
backtest paths. The walk-forward harness can compare RL against any of
these by swapping the registry entry.

Implementations:
    * RandomPolicyScorer    — uniform over {skip, take_25, take_50, take_100}
    * AlwaysTakePolicyScorer — fires the configured action for every candidate
    * NeverTakePolicyScorer  — always skip; useful as a floor benchmark
"""
from __future__ import annotations

import hashlib
import random
import uuid
from dataclasses import dataclass
from typing import cast

from rl_swing.domain import (
    ACTION_TO_SIZE,
    CandidateTrade,
    FeatureFrame,
    PolicyAction,
    PolicyDecision,
    PortfolioState,
)


def _decision(
    candidate: CandidateTrade,
    model_id: str,
    action: PolicyAction,
    raw_action: int | float,
    confidence: float | None,
    obs_hash: str | None,
    explanation: dict | None = None,
) -> PolicyDecision:
    target = candidate.base_size_pct * ACTION_TO_SIZE[action]
    return PolicyDecision(
        decision_id=f"dec-{uuid.uuid4().hex[:12]}",
        candidate_id=candidate.candidate_id,
        as_of=candidate.as_of,
        model_id=model_id,
        action=action,
        confidence=confidence,
        target_size_pct=target,
        raw_action=raw_action,
        observation_hash=obs_hash or "",
        explanation=explanation or {},
    )


@dataclass
class RandomPolicyScorer:
    model_id: str = "baseline_random"
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def score(
        self,
        candidate: CandidateTrade,
        features: FeatureFrame,
        portfolio_state: PortfolioState,
    ) -> PolicyDecision:
        a = self._rng.randint(0, 3)
        action = cast(PolicyAction, ["skip", "take_25", "take_50", "take_100"][a])
        return _decision(
            candidate, self.model_id, action, raw_action=a,
            confidence=0.25,
            obs_hash=hashlib.sha1(candidate.candidate_id.encode()).hexdigest()[:12],
        )


@dataclass
class AlwaysTakePolicyScorer:
    model_id: str = "baseline_always_take"
    action: PolicyAction = "take_100"

    def score(
        self,
        candidate: CandidateTrade,
        features: FeatureFrame,
        portfolio_state: PortfolioState,
    ) -> PolicyDecision:
        return _decision(
            candidate, self.model_id, self.action,
            raw_action={"skip": 0, "take_25": 1, "take_50": 2, "take_100": 3}[self.action],
            confidence=1.0,
            obs_hash=hashlib.sha1(candidate.candidate_id.encode()).hexdigest()[:12],
        )


@dataclass
class NeverTakePolicyScorer:
    model_id: str = "baseline_never_take"

    def score(
        self,
        candidate: CandidateTrade,
        features: FeatureFrame,
        portfolio_state: PortfolioState,
    ) -> PolicyDecision:
        return _decision(
            candidate, self.model_id, "skip", raw_action=0,
            confidence=1.0,
            obs_hash=hashlib.sha1(candidate.candidate_id.encode()).hexdigest()[:12],
        )
