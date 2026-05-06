"""Reconciliation domain types."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

Severity = Literal["INFO", "WARN", "ERROR", "CRITICAL"]
BreakType = Literal[
    "missing_position",
    "unexpected_position",
    "quantity_mismatch",
    "market_value_drift",
    "open_order_mismatch",
    "duplicate_order",
    "cash_mismatch",
    "feature_model_version_mismatch",
    "feature_freshness",
    "data_quality",
]


@dataclass(frozen=True, slots=True)
class ReconciliationBreak:
    recon_id: str
    recon_at: datetime
    environment: str
    break_type: BreakType
    severity: Severity
    description: str
    expected: dict = field(default_factory=dict)
    actual: dict = field(default_factory=dict)
    resolved_at: datetime | None = None
