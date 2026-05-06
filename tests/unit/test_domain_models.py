"""Domain dataclass invariants and helpers."""
from __future__ import annotations

from datetime import datetime

import pytest

from rl_swing.domain import (
    ACTION_TO_SIZE,
    AccountSnapshot,
    AuditEvent,
    BrokerOrder,
    CandidateTrade,
    EventType,
    FeatureFrame,
    FeatureSnapshot,
    FillEvent,
    MarketBar,
    MarketSnapshot,
    OrderIntent,
    PolicyDecision,
    PortfolioState,
    PositionSnapshot,
    ReconciliationBreak,
    RiskDecision,
    RiskRuleResult,
)


def _bar(symbol: str = "AAPL", flag: tuple[str, ...] = ()) -> MarketBar:
    return MarketBar(
        symbol=symbol, timestamp=datetime(2024, 1, 2), timeframe="1d",
        open=100.0, high=101.0, low=99.0, close=100.5, volume=1_000_000,
        adjusted_close=100.5, source="test",
        quality_flags=flag,
    )


def test_market_bar_with_quality_flag_is_idempotent():
    b = _bar()
    b1 = b.with_quality_flag("warn")
    b2 = b1.with_quality_flag("warn")
    assert b1 is not b
    assert b2 is b1
    assert b1.quality_flags == ("warn",)


def test_market_bar_with_quality_flag_sorted():
    b = _bar()
    out = b.with_quality_flag("zeta").with_quality_flag("alpha")
    assert out.quality_flags == ("alpha", "zeta")


def test_market_snapshot_metadata_default():
    snap = MarketSnapshot(
        snapshot_id="abc", provider_id="x",
        timeframe="1d", universe_version="u",
        start=datetime(2020, 1, 1), end=datetime(2020, 12, 31),
    )
    assert snap.metadata == {}


def test_feature_frame_rejects_missing_value():
    with pytest.raises(ValueError):
        FeatureFrame(
            as_of=datetime(2024, 1, 2),
            symbol="AAPL",
            feature_version="v1",
            values={"a": 1.0},
            feature_names=("a", "b"),
            source_snapshot_id="s",
        )


def test_feature_frame_vector_in_declared_order():
    f = FeatureFrame(
        as_of=datetime(2024, 1, 2),
        symbol="AAPL",
        feature_version="v1",
        values={"b": 2.0, "a": 1.0},
        feature_names=("a", "b"),
        source_snapshot_id="s",
    )
    assert f.vector() == [1.0, 2.0]


def test_feature_snapshot_default_metadata():
    fs = FeatureSnapshot(
        snapshot_id="x", feature_version="v", market_snapshot_id="m",
    )
    assert fs.metadata == {}


def test_candidate_trade_validates_size_and_strength():
    base = dict(
        candidate_id="c1", as_of=datetime(2024, 1, 2),
        symbol="AAPL", strategy_id="m", direction="long",
        entry_timing="next_open", base_size_pct=0.1, max_holding_days=10,
        stop_rule_id=None, exit_rule_id="x", signal_strength=0.5,
    )
    CandidateTrade(**base)

    with pytest.raises(ValueError):
        CandidateTrade(**{**base, "base_size_pct": 1.5})
    with pytest.raises(ValueError):
        CandidateTrade(**{**base, "signal_strength": -0.1})
    with pytest.raises(ValueError):
        CandidateTrade(**{**base, "max_holding_days": 0})


def test_action_to_size_table_complete():
    assert ACTION_TO_SIZE["skip"] == 0.0
    assert ACTION_TO_SIZE["take_25"] == 0.25
    assert ACTION_TO_SIZE["take_50"] == 0.50
    assert ACTION_TO_SIZE["take_100"] == 1.00


def test_policy_decision_carries_target_size():
    pd = PolicyDecision(
        decision_id="d", candidate_id="c", as_of=datetime(2024, 1, 2),
        model_id="m", action="take_50", confidence=0.5,
        target_size_pct=0.05, raw_action=2,
        observation_hash="abc",
    )
    assert pd.action == "take_50"
    assert pd.target_size_pct == 0.05
    assert pd.explanation == {}


def test_risk_decision_default_tuples_are_empty():
    rd = RiskDecision(
        decision_id="rd", candidate_id="c", policy_decision_id="d",
        approved=True, final_size_pct=0.05,
    )
    assert rd.blocked_reasons == ()
    assert rd.applied_rules == ()


def test_risk_rule_result_defaults():
    r = RiskRuleResult(rule_id="x", approves=True)
    assert r.size_multiplier == 1.0
    assert r.block_reason is None


def test_order_intent_carries_environment():
    oi = OrderIntent(
        intent_id="i", as_of=datetime(2024, 1, 2),
        symbol="AAPL", side="buy", quantity=10,
        order_type="market", time_in_force="day",
        limit_price=None, source_decision_id="rd",
        environment="backtest", client_order_id="abc-123",
    )
    assert oi.environment == "backtest"


def test_broker_order_status_round_trip():
    bo = BrokerOrder(
        internal_order_id="i", broker_order_id=None, client_order_id="c",
        environment="shadow", symbol="AAPL", side="buy",
        order_type="market", time_in_force="day",
        requested_qty=10, limit_price=None, status="ACCEPTED",
        submitted_at=datetime(2024, 1, 2), updated_at=datetime(2024, 1, 2),
    )
    assert bo.status == "ACCEPTED"
    assert bo.raw_request == {}


def test_fill_event_default_raw():
    f = FillEvent(
        fill_id="f", internal_order_id="i", broker_order_id="b",
        symbol="AAPL", side="buy", filled_qty=5, filled_avg_price=100.0,
        filled_at=datetime(2024, 1, 2),
    )
    assert f.raw_fill == {}


def test_position_snapshot_optional_fields():
    p = PositionSnapshot(
        as_of=datetime(2024, 1, 2), source="simulated",
        symbol="AAPL", quantity=10, market_value=1000.0,
    )
    assert p.avg_entry_price is None
    assert p.unrealized_pnl is None
    assert p.days_held == 0


def test_portfolio_state_gross_exposure_and_lookup():
    p1 = PositionSnapshot(
        as_of=datetime(2024, 1, 2), source="simulated",
        symbol="AAPL", quantity=10, market_value=2000.0,
    )
    p2 = PositionSnapshot(
        as_of=datetime(2024, 1, 2), source="simulated",
        symbol="MSFT", quantity=5, market_value=1500.0,
    )
    s = PortfolioState(
        as_of=datetime(2024, 1, 2), cash=10000.0, equity=10000.0,
        positions=(p1, p2), open_positions_count=2,
    )
    assert s.gross_exposure_pct == pytest.approx(0.35)
    assert s.position_for("AAPL") is p1
    assert s.position_for("UNKNOWN") is None


def test_portfolio_state_zero_equity_protects_against_div_zero():
    s = PortfolioState(as_of=datetime(2024, 1, 2), cash=0, equity=0)
    assert s.gross_exposure_pct == 0.0


def test_account_snapshot_default_raw():
    a = AccountSnapshot(
        as_of=datetime(2024, 1, 2), source="simulated",
        cash=1000, buying_power=1000, equity=1000, portfolio_value=1000,
    )
    assert a.raw == {}


def test_audit_event_tags_default_empty():
    ev = AuditEvent(
        event_id="e", event_type=EventType.PIPELINE_STARTED,
        timestamp=datetime(2024, 1, 2), correlation_id="c",
        payload={}, run_id="r", environment="research",
    )
    assert ev.tags == ()
    assert ev.schema_version == "v1"


def test_reconciliation_break_resolved_optional():
    b = ReconciliationBreak(
        recon_id="x", recon_at=datetime(2024, 1, 2),
        environment="paper", break_type="quantity_mismatch",
        severity="WARN", description="off by 1",
    )
    assert b.resolved_at is None
    assert b.expected == {}


def test_event_type_values_match_strings():
    assert EventType.MARKET_DATA_LOADED.value == "MarketDataLoaded"
    assert EventType.RISK_LIMIT_BREACHED.value == "RiskLimitBreached"


def test_market_snapshot_metadata_carries_dict():
    snap = MarketSnapshot(
        snapshot_id="abc", provider_id="x",
        timeframe="1d", universe_version="u",
        start=datetime(2020, 1, 1), end=datetime(2020, 12, 31),
        metadata={"k": 1},
    )
    assert snap.metadata == {"k": 1}
