"""Candidate-trade domain types.

A ``CandidateTrade`` is the output of a rule-based ``CandidateStrategy``
and the input to a ``PolicyScorer``. Every field below is the *intent*,
not the *decision* — the policy and risk layers may reject, scale, or
defer it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

Direction = Literal["long", "short"]
EntryTiming = Literal["next_open", "next_close", "limit"]


@dataclass(frozen=True, slots=True)
class CandidateTrade:
    candidate_id: str
    as_of: datetime
    symbol: str
    strategy_id: str
    direction: Direction
    entry_timing: EntryTiming
    base_size_pct: float          # default risk-capped fraction of account
    max_holding_days: int
    stop_rule_id: str | None
    exit_rule_id: str
    signal_strength: float        # 0..1; provider-defined
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.base_size_pct <= 1.0:
            raise ValueError(
                f"base_size_pct out of range [0,1]: {self.base_size_pct}"
            )
        if not 0.0 <= self.signal_strength <= 1.0:
            raise ValueError(
                f"signal_strength out of range [0,1]: {self.signal_strength}"
            )
        if self.max_holding_days <= 0:
            raise ValueError("max_holding_days must be > 0")
