"""Port (interface) definitions.

Importing from ``rl_swing.ports`` should never trigger heavy imports.
Adapters live under ``rl_swing.adapters.*`` and are wired in via the
component registry — never imported directly by service code.
"""
from .broker import BrokerAdapter
from .event_bus import EventBus, EventListener
from .feature_pipeline import FeaturePipeline
from .market_data import MarketDataProvider
from .model_registry import ApprovalStatus, ModelEntry, ModelRegistry
from .policy_scorer import PolicyScorer
from .risk_policy import MarketState, RiskPolicy
from .storage import (
    AuditRepository,
    BarRepository,
    CandidateRepository,
    DecisionRepository,
    OrderRepository,
    ReconciliationRepository,
)
from .strategy import CandidateStrategy

__all__ = [
    "ApprovalStatus",
    "AuditRepository",
    "BarRepository",
    "BrokerAdapter",
    "CandidateRepository",
    "CandidateStrategy",
    "DecisionRepository",
    "EventBus",
    "EventListener",
    "FeaturePipeline",
    "MarketDataProvider",
    "MarketState",
    "ModelEntry",
    "ModelRegistry",
    "OrderRepository",
    "PolicyScorer",
    "ReconciliationRepository",
    "RiskPolicy",
]
