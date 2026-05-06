"""Domain dataclasses — the stable contracts everything else depends on.

Importing from ``rl_swing.domain`` should never pull in heavy
dependencies (pandas/torch/etc). Keep this layer pure.
"""
from .candidates import CandidateTrade, Direction, EntryTiming
from .decisions import (
    ACTION_TO_SIZE,
    PolicyAction,
    PolicyDecision,
    RiskDecision,
    RiskRuleResult,
)
from .events import AuditEvent, EventType
from .features import FeatureFrame, FeatureSnapshot
from .market import MarketBar, MarketSnapshot
from .orders import (
    BrokerOrder,
    Environment,
    FillEvent,
    OrderIntent,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)
from .portfolio import AccountSnapshot, PortfolioState, PositionSnapshot
from .reconciliation import BreakType, ReconciliationBreak, Severity

__all__ = [
    "ACTION_TO_SIZE",
    "AccountSnapshot",
    "AuditEvent",
    "BreakType",
    "BrokerOrder",
    "CandidateTrade",
    "Direction",
    "EntryTiming",
    "Environment",
    "EventType",
    "FeatureFrame",
    "FeatureSnapshot",
    "FillEvent",
    "MarketBar",
    "MarketSnapshot",
    "OrderIntent",
    "OrderStatus",
    "OrderType",
    "PolicyAction",
    "PolicyDecision",
    "PortfolioState",
    "PositionSnapshot",
    "ReconciliationBreak",
    "RiskDecision",
    "RiskRuleResult",
    "Severity",
    "Side",
    "TimeInForce",
]
