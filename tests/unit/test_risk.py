"""Risk policies + engine."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
import yaml

from rl_swing.domain import (
    CandidateTrade,
    PolicyDecision,
    PortfolioState,
    PositionSnapshot,
)
from rl_swing.ports.risk_policy import MarketState
from rl_swing.risk.engine import RiskEngine
from rl_swing.risk.policies import (
    DuplicateOrderPolicy,
    KillSwitchPolicy,
    LiquidityPolicy,
    LiveTradingApprovalPolicy,
    MaxDailyLossPolicy,
    MaxDailyNewPositionsPolicy,
    MaxOpenPositionsPolicy,
    MaxPortfolioExposurePolicy,
    MaxSinglePositionPolicy,
)


def _candidate(symbol: str = "AAPL", **kw) -> CandidateTrade:
    base = dict(
        candidate_id=f"c-{symbol}", as_of=datetime(2024, 1, 2), symbol=symbol,
        strategy_id="momentum_20_60", direction="long",
        entry_timing="next_open", base_size_pct=0.10, max_holding_days=10,
        stop_rule_id=None, exit_rule_id="x", signal_strength=0.5, metadata={},
    )
    base.update(kw)
    return CandidateTrade(**base)


def _decision(target: float = 0.10, action: str = "take_100") -> PolicyDecision:
    return PolicyDecision(
        decision_id="d", candidate_id="c-AAPL", as_of=datetime(2024, 1, 2),
        model_id="m", action=action, confidence=1.0,
        target_size_pct=target, raw_action=3, observation_hash="h",
    )


def _portfolio(**kw) -> PortfolioState:
    base = dict(as_of=datetime(2024, 1, 2), cash=10_000, equity=10_000)
    base.update(kw)
    return PortfolioState(**base)


_MARKET = MarketState(is_market_open=True, is_kill_switch_active=False)


# --- single-position -----------------------------------------------------
def test_max_single_position_allows_under_cap():
    rule = MaxSinglePositionPolicy("r", max_pct=0.10)
    r = rule.evaluate(_candidate(), _decision(target=0.05), _portfolio(), _MARKET)
    assert r.approves
    assert r.size_multiplier == 1.0


def test_max_single_position_scales_when_over_cap():
    rule = MaxSinglePositionPolicy("r", max_pct=0.05)
    r = rule.evaluate(_candidate(), _decision(target=0.10), _portfolio(), _MARKET)
    assert r.approves
    assert r.size_multiplier == pytest.approx(0.5)


def test_max_single_position_blocks_when_cap_zero():
    rule = MaxSinglePositionPolicy("r", max_pct=0.0)
    r = rule.evaluate(_candidate(), _decision(target=0.10), _portfolio(), _MARKET)
    assert not r.approves


# --- portfolio exposure --------------------------------------------------
def test_max_portfolio_exposure_passes_when_room():
    rule = MaxPortfolioExposurePolicy("r", max_pct=0.50)
    r = rule.evaluate(_candidate(), _decision(target=0.10), _portfolio(), _MARKET)
    assert r.approves


def test_max_portfolio_exposure_blocks_when_already_full():
    pos = PositionSnapshot(
        as_of=datetime(2024, 1, 2), source="simulated",
        symbol="X", quantity=1, market_value=10_000,  # 100% exposure
    )
    rule = MaxPortfolioExposurePolicy("r", max_pct=0.50)
    p = _portfolio(positions=(pos,))
    r = rule.evaluate(_candidate(), _decision(target=0.10), p, _MARKET)
    assert not r.approves


def test_max_portfolio_exposure_scales_to_remaining_room():
    pos = PositionSnapshot(
        as_of=datetime(2024, 1, 2), source="simulated",
        symbol="X", quantity=1, market_value=4_000,  # 40% exposure
    )
    rule = MaxPortfolioExposurePolicy("r", max_pct=0.50)
    p = _portfolio(positions=(pos,))
    # room = 10%; want 20% -> mult = 0.5
    r = rule.evaluate(_candidate(), _decision(target=0.20), p, _MARKET)
    assert r.approves
    assert r.size_multiplier == pytest.approx(0.5)


def test_max_portfolio_exposure_passes_zero_size_decision():
    rule = MaxPortfolioExposurePolicy("r", max_pct=0.50)
    r = rule.evaluate(_candidate(), _decision(target=0.0, action="skip"),
                      _portfolio(), _MARKET)
    assert r.approves


# --- daily loss ----------------------------------------------------------
def test_daily_loss_blocks_after_breach():
    rule = MaxDailyLossPolicy("r", max_pct=0.01)
    p = _portfolio(daily_loss_pct=0.02)
    r = rule.evaluate(_candidate(), _decision(), p, _MARKET)
    assert not r.approves


def test_daily_loss_allows_under_threshold():
    rule = MaxDailyLossPolicy("r", max_pct=0.01)
    p = _portfolio(daily_loss_pct=0.001)
    r = rule.evaluate(_candidate(), _decision(), p, _MARKET)
    assert r.approves


# --- max open positions --------------------------------------------------
def test_max_open_positions_blocks_when_full():
    rule = MaxOpenPositionsPolicy("r", max_positions=2)
    p = _portfolio(open_positions_count=2)
    r = rule.evaluate(_candidate(symbol="NEW"), _decision(), p, _MARKET)
    assert not r.approves


def test_max_open_positions_allows_existing_symbol():
    pos = PositionSnapshot(
        as_of=datetime(2024, 1, 2), source="simulated",
        symbol="AAPL", quantity=1, market_value=1000,
    )
    rule = MaxOpenPositionsPolicy("r", max_positions=2)
    p = _portfolio(open_positions_count=2, positions=(pos,))
    r = rule.evaluate(_candidate(symbol="AAPL"), _decision(), p, _MARKET)
    assert r.approves


# --- max daily new positions ---------------------------------------------
def test_max_daily_new_positions_resets_per_day():
    rule = MaxDailyNewPositionsPolicy("r", max_positions=1)
    c1 = _candidate(symbol="A")
    c2 = _candidate(symbol="B")
    assert rule.evaluate(c1, _decision(), _portfolio(), _MARKET).approves
    blocked = rule.evaluate(c2, _decision(), _portfolio(), _MARKET)
    assert not blocked.approves
    # Next day -> counter resets.
    c3 = CandidateTrade(
        candidate_id="c3", as_of=datetime(2024, 1, 3),
        symbol="C", strategy_id="m", direction="long",
        entry_timing="next_open", base_size_pct=0.10, max_holding_days=10,
        stop_rule_id=None, exit_rule_id="x", signal_strength=0.5,
    )
    assert rule.evaluate(c3, _decision(), _portfolio(), _MARKET).approves


def test_max_daily_new_positions_skips_size_zero():
    rule = MaxDailyNewPositionsPolicy("r", max_positions=1)
    r = rule.evaluate(_candidate(), _decision(target=0.0, action="skip"),
                      _portfolio(), _MARKET)
    assert r.approves


def test_max_daily_new_positions_passes_existing_position():
    pos = PositionSnapshot(
        as_of=datetime(2024, 1, 2), source="simulated",
        symbol="AAPL", quantity=1, market_value=1000,
    )
    rule = MaxDailyNewPositionsPolicy("r", max_positions=0)
    r = rule.evaluate(_candidate(symbol="AAPL"), _decision(),
                      _portfolio(positions=(pos,)), _MARKET)
    assert r.approves


# --- liquidity -----------------------------------------------------------
def test_liquidity_passes_when_unknown():
    rule = LiquidityPolicy("r", min_avg_dollar_volume=1e8)
    r = rule.evaluate(_candidate(), _decision(), _portfolio(), _MARKET)
    assert r.approves
    assert "unknown" in (r.note or "")


def test_liquidity_blocks_below_threshold():
    rule = LiquidityPolicy("r", min_avg_dollar_volume=1e8)
    c = _candidate(metadata={"avg_dollar_volume": 1e6})
    r = rule.evaluate(c, _decision(), _portfolio(), _MARKET)
    assert not r.approves


def test_liquidity_passes_above_threshold():
    rule = LiquidityPolicy("r", min_avg_dollar_volume=1e6)
    c = _candidate(metadata={"avg_dollar_volume": 1e9})
    r = rule.evaluate(c, _decision(), _portfolio(), _MARKET)
    assert r.approves


# --- duplicate order -----------------------------------------------------
def test_duplicate_order_blocks_second_evaluation():
    rule = DuplicateOrderPolicy("r")
    c = _candidate()
    assert rule.evaluate(c, _decision(), _portfolio(), _MARKET).approves
    assert not rule.evaluate(c, _decision(), _portfolio(), _MARKET).approves


# --- kill switch ---------------------------------------------------------
def test_kill_switch_blocks_when_active():
    rule = KillSwitchPolicy("r")
    state = MarketState(is_market_open=True, is_kill_switch_active=True)
    r = rule.evaluate(_candidate(), _decision(), _portfolio(), state)
    assert not r.approves


def test_kill_switch_passes_when_inactive():
    rule = KillSwitchPolicy("r")
    r = rule.evaluate(_candidate(), _decision(), _portfolio(), _MARKET)
    assert r.approves


# --- live approval -------------------------------------------------------
def test_live_approval_blocks_without_token(monkeypatch):
    monkeypatch.delenv("RL_SWING_LIVE_APPROVAL_TOKEN", raising=False)
    rule = LiveTradingApprovalPolicy("r")
    r = rule.evaluate(_candidate(), _decision(), _portfolio(), _MARKET)
    assert not r.approves


def test_live_approval_passes_with_token(monkeypatch):
    monkeypatch.setenv("RL_SWING_LIVE_APPROVAL_TOKEN", "yes")
    rule = LiveTradingApprovalPolicy("r")
    r = rule.evaluate(_candidate(), _decision(), _portfolio(), _MARKET)
    assert r.approves


# --- engine --------------------------------------------------------------
def test_risk_engine_composes_and_blocks():
    engine = RiskEngine([
        MaxSinglePositionPolicy("a", max_pct=0.20),
        DuplicateOrderPolicy("b"),
    ])
    c = _candidate()
    rd1 = engine.evaluate(c, _decision(), _portfolio(), _MARKET)
    assert rd1.approved
    # Same candidate again: dup rule blocks.
    rd2 = engine.evaluate(c, _decision(), _portfolio(), _MARKET)
    assert not rd2.approved
    assert "duplicate_candidate" in rd2.blocked_reasons


def test_risk_engine_chains_size_multipliers():
    engine = RiskEngine([
        MaxSinglePositionPolicy("a", max_pct=0.05),  # 0.5x
        MaxPortfolioExposurePolicy("b", max_pct=1.0),  # 1.0x
    ])
    rd = engine.evaluate(_candidate(), _decision(target=0.10),
                         _portfolio(), _MARKET)
    assert rd.approved
    assert rd.final_size_pct == pytest.approx(0.05)


def test_risk_engine_blocks_when_target_zero():
    engine = RiskEngine([MaxSinglePositionPolicy("a", max_pct=1.0)])
    rd = engine.evaluate(
        _candidate(), _decision(target=0.0, action="skip"),
        _portfolio(), _MARKET,
    )
    assert not rd.approved
    assert rd.final_size_pct == 0.0


def test_risk_engine_from_yaml(tmp_path: Path):
    yml = tmp_path / "profile.yaml"
    yml.write_text(yaml.safe_dump({
        "risk_profile": {
            "name": "test",
            "policies": [
                {"class": "rl_swing.risk.policies.MaxSinglePositionPolicy",
                 "params": {"rule_id": "r", "max_pct": 0.10}},
                {"class": "rl_swing.risk.policies.KillSwitchPolicy",
                 "params": {"rule_id": "k"}},
            ],
        }
    }))
    engine = RiskEngine.from_yaml(yml)
    assert len(engine.policies) == 2
    rd = engine.evaluate(_candidate(), _decision(target=0.05), _portfolio(), _MARKET)
    assert rd.approved
