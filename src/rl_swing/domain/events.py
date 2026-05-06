"""Audit-event domain types.

Every meaningful state transition emits an immutable ``AuditEvent``.
This is the single replay/debugging substrate for the system.

The ``correlation_id`` chain ties one logical decision together:

    candidate_id
      -> policy_decision_id
        -> risk_decision_id
          -> order_intent_id
            -> broker_order_id
              -> fill_id
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class EventType(str, Enum):
    MARKET_DATA_LOADED = "MarketDataLoaded"
    FEATURES_BUILT = "FeaturesBuilt"
    CANDIDATE_GENERATED = "CandidateGenerated"
    POLICY_SCORED = "PolicyScored"
    RISK_EVALUATED = "RiskEvaluated"
    ORDER_INTENT_CREATED = "OrderIntentCreated"
    ORDER_SUBMITTED = "OrderSubmitted"
    ORDER_FILLED = "OrderFilled"
    ORDER_REJECTED = "OrderRejected"
    POSITION_UPDATED = "PositionUpdated"
    RECONCILIATION_COMPLETED = "ReconciliationCompleted"
    RECONCILIATION_BREAK_DETECTED = "ReconciliationBreakDetected"
    RISK_LIMIT_BREACHED = "RiskLimitBreached"
    MODEL_PROMOTED = "ModelPromoted"
    KILL_SWITCH_ACTIVATED = "KillSwitchActivated"
    PIPELINE_STARTED = "PipelineStarted"
    PIPELINE_COMPLETED = "PipelineCompleted"


@dataclass(frozen=True, slots=True)
class AuditEvent:
    event_id: str
    event_type: EventType
    timestamp: datetime
    correlation_id: str
    payload: dict
    run_id: str
    environment: str
    schema_version: str = "v1"
    tags: tuple[str, ...] = ()
