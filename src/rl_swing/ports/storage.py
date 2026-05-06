"""Repository ports.

These are intentionally narrow. Adapters: SQLite, Postgres, parquet.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Protocol, runtime_checkable

from rl_swing.domain import (
    AuditEvent,
    BrokerOrder,
    CandidateTrade,
    FillEvent,
    MarketBar,
    PolicyDecision,
    ReconciliationBreak,
    RiskDecision,
)


@runtime_checkable
class BarRepository(Protocol):
    def save_bars(self, bars: Iterable[MarketBar]) -> int: ...
    def load_bars(
        self,
        symbols: list[str],
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> list[MarketBar]: ...


@runtime_checkable
class CandidateRepository(Protocol):
    def save_candidates(self, candidates: Iterable[CandidateTrade]) -> int: ...
    def load_candidates(self, run_id: str) -> list[CandidateTrade]: ...


@runtime_checkable
class DecisionRepository(Protocol):
    def save_policy_decisions(self, decisions: Iterable[PolicyDecision]) -> int: ...
    def save_risk_decisions(self, decisions: Iterable[RiskDecision]) -> int: ...
    def load_policy_decisions(self, run_id: str) -> list[PolicyDecision]: ...


@runtime_checkable
class OrderRepository(Protocol):
    def save_orders(self, orders: Iterable[BrokerOrder]) -> int: ...
    def save_fills(self, fills: Iterable[FillEvent]) -> int: ...


@runtime_checkable
class ReconciliationRepository(Protocol):
    def save_breaks(self, breaks: Iterable[ReconciliationBreak]) -> int: ...


@runtime_checkable
class AuditRepository(Protocol):
    def append_events(self, events: Iterable[AuditEvent]) -> int: ...
    def replay(self, run_id: str) -> list[AuditEvent]: ...
