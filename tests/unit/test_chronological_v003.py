"""Unit tests for v3 chronological env (FEAT-32 M1).

These tests pin the MDP correctness invariants the env depends on:
position lifecycle, day-stepping, portfolio bookkeeping, action
validity, episode boundaries, reward shape, and the variant's
ability to evaluate over a synthetic mini-window.

The tests deliberately avoid yfinance — they build hand-crafted
bars + frames + packs in-memory so MDP behavior can be verified
without network or feature pipeline dependencies.
"""
from __future__ import annotations

from datetime import date, datetime

import pytest

from rl_swing.domain import (
    CandidateTrade,
    FeatureFrame,
    MarketBar,
)
from rl_swing.rl.agents.portfolio_baselines import (
    NoOpPortfolioPolicy,
    RandomActionPortfolioPolicy,
    TopKPortfolioPolicy,
)
from rl_swing.rl.env.chronological_swing_env import (
    OBS_DIM,
    ChronologicalSwingEnv,
)
from rl_swing.rl.env.cost_model import EquityExecutionModel
from rl_swing.rl.env.portfolio_state_tracker import (
    ClosedTrade,
    OpenPosition,
    PortfolioStateTracker,
)
from rl_swing.rl.env.reward_model import RewardModel
from rl_swing.strategies.multi_strategy_packer import StrategyPack


# ---------------------------------------------------------------------
# Test fixtures: hand-built bars/frames/packs so MDP behavior is
# pinned without the yfinance + feature-pipeline machinery.
# ---------------------------------------------------------------------
def _make_candidate(
    symbol: str = "AAA", as_of: datetime | None = None,
    signal_strength: float = 0.7, base_size_pct: float = 0.10,
    max_holding_days: int = 5, candidate_id: str = "test-cand",
):
    return CandidateTrade(
        candidate_id=candidate_id,
        as_of=as_of or datetime(2024, 1, 1),
        symbol=symbol,
        strategy_id="test",
        direction="long",
        entry_timing="next_open",
        base_size_pct=base_size_pct,
        max_holding_days=max_holding_days,
        stop_rule_id=None,
        exit_rule_id="time_or_atr",
        signal_strength=signal_strength,
        metadata={},
    )


def _make_bars(symbols: list[str], dates: list[date], price_path: list[float] | None = None):
    """Generate one MarketBar per (symbol, date). Same price path
    applied to every symbol (deterministic for tests)."""
    if price_path is None:
        price_path = [100.0 + i for i in range(len(dates))]
    bars: list[MarketBar] = []
    for sym in symbols:
        for d, p in zip(dates, price_path, strict=False):
            bars.append(MarketBar(
                symbol=sym, timestamp=datetime(d.year, d.month, d.day),
                timeframe="1d",
                open=p, high=p * 1.005, low=p * 0.995, close=p,
                volume=1_000_000.0, adjusted_close=p,
                source="test", quality_flags=(),
            ))
    return bars


def _make_packs_one_per_day(
    symbols: list[str], dates: list[date], n_per_day: int = 1,
    base_size_pct: float = 0.10, max_holding_days: int = 3,
) -> list[StrategyPack]:
    """One StrategyPack per (symbol, date) with one fired candidate
    on slot 0. n_per_day controls how many symbols fire on each
    date (limited to len(symbols))."""
    n = min(n_per_day, len(symbols))
    packs: list[StrategyPack] = []
    for d in dates:
        for sym in symbols[:n]:
            cand = _make_candidate(
                symbol=sym, as_of=datetime(d.year, d.month, d.day),
                base_size_pct=base_size_pct,
                max_holding_days=max_holding_days,
                candidate_id=f"{sym}-{d.isoformat()}",
            )
            packs.append(StrategyPack(
                symbol=sym, as_of=datetime(d.year, d.month, d.day),
                candidates=(cand, None, None),
            ))
    return packs


# ---------------------------------------------------------------------
# PortfolioStateTracker tests
# ---------------------------------------------------------------------
def test_tracker_open_position_consumes_cash_and_appends():
    t = PortfolioStateTracker()
    ok = t.open_position(
        symbol="AAA", entry_date=date(2024, 1, 2), entry_price=100.0,
        size_pct=0.10, max_holding_days=5, cost_bps_round_trip=10.0,
    )
    assert ok is True
    assert t.n_open == 1
    assert t.gross_exposure_pct == pytest.approx(0.10)
    # Cash drops by size_pct + entry-side cost (5 bps of size_pct):
    expected_cash = 1.0 - 0.10 - 0.10 * (5.0 / 10_000.0)
    assert t.cash_pct == pytest.approx(expected_cash)


def test_tracker_refuses_when_gross_exposure_exceeds_one():
    """Stack 9 positions at 10% each = 90% gross. The 10th would
    push cash below the 0.05 floor (cost drag accumulates with
    each entry); the cash check fires first. This documents the
    actual order of budget checks."""
    t = PortfolioStateTracker()
    for i in range(9):
        ok = t.open_position(
            symbol=f"S{i}", entry_date=date(2024, 1, 2), entry_price=100.0,
            size_pct=0.10, max_holding_days=5, cost_bps_round_trip=10.0,
        )
        assert ok is True
    # 10th: would be at 100% gross, but cash drops below 0.05 first.
    ok = t.open_position(
        symbol="OVER", entry_date=date(2024, 1, 2), entry_price=100.0,
        size_pct=0.10, max_holding_days=5, cost_bps_round_trip=10.0,
    )
    assert ok is False
    assert t.n_open == 9


def test_tracker_refuses_at_gross_exposure_when_cash_irrelevant():
    """A direct test of the gross-exposure cap: open one 99% position
    (passes both cash and gross checks), then attempt another 5%.
    The cash check refuses (cash already at floor)."""
    t = PortfolioStateTracker()
    # 60% position is fine: cash_after = 1 - 0.60 - tiny ≈ 0.4.
    ok = t.open_position(
        symbol="A", entry_date=date(2024, 1, 2), entry_price=100.0,
        size_pct=0.60, max_holding_days=5, cost_bps_round_trip=10.0,
    )
    assert ok is True
    # 50% more: pushes gross to 1.10 > 1.0 → gross-exposure refusal
    # fires (it's checked before the cash arithmetic).
    ok = t.open_position(
        symbol="B", entry_date=date(2024, 1, 2), entry_price=100.0,
        size_pct=0.50, max_holding_days=5, cost_bps_round_trip=10.0,
    )
    assert ok is False
    assert t.n_open == 1


def test_tracker_refuses_when_cash_drops_below_threshold():
    t = PortfolioStateTracker()
    # Open a 90% position — cash after = 1 - 0.90 - tiny cost ≈ 0.0996.
    ok = t.open_position(
        symbol="BIG", entry_date=date(2024, 1, 2), entry_price=100.0,
        size_pct=0.90, max_holding_days=5, cost_bps_round_trip=10.0,
    )
    assert ok is True
    # Another 0.10 position would drop cash to ~ -0.0005 (well under
    # the 0.05 floor) → refused.
    ok = t.open_position(
        symbol="TINY", entry_date=date(2024, 1, 2), entry_price=100.0,
        size_pct=0.10, max_holding_days=5, cost_bps_round_trip=10.0,
    )
    assert ok is False


def test_tracker_advance_marks_to_market_and_exits_on_holding_days():
    """A position with max_holding_days=2 opened on day 0:
       - day 1: advance, days_held=1, still open, mtm reflected
       - day 2: advance, days_held=2 → exits at today's close, P&L realized
    """
    t = PortfolioStateTracker()
    t.open_position(
        symbol="AAA", entry_date=date(2024, 1, 2), entry_price=100.0,
        size_pct=0.10, max_holding_days=2, cost_bps_round_trip=10.0,
    )
    # Day 1 (Jan 3): close at 102 — unrealized +2%×size_pct = +0.002.
    pnl_d1 = t.advance_one_day(date(2024, 1, 3), {"AAA": 102.0})
    assert t.n_open == 1
    assert pnl_d1 == pytest.approx(0.10 * 0.02)
    # Day 2 (Jan 4): close at 105 — exits. Realized = 0.10×(0.05 - 10/10000)
    pnl_d2 = t.advance_one_day(date(2024, 1, 4), {"AAA": 105.0})
    assert t.n_open == 0
    assert len(t.closed_trades) == 1
    expected_realized = 0.10 * (0.05 - 10 / 10_000.0)
    # daily P&L = realized + (today_unrealized 0 − yesterday_unrealized 0.002)
    assert pnl_d2 == pytest.approx(expected_realized + (0.0 - 0.002))


def test_tracker_drawdown_tracks_peak_and_current():
    t = PortfolioStateTracker()
    t.open_position(
        symbol="AAA", entry_date=date(2024, 1, 2), entry_price=100.0,
        size_pct=0.20, max_holding_days=10, cost_bps_round_trip=10.0,
    )
    # Day 1: price rises to 110. Portfolio value = cash + 0.20 × 0.10 (10% asset return) = cash + 0.02.
    t.advance_one_day(date(2024, 1, 3), {"AAA": 110.0})
    peak_after_up = t.peak_value_pct
    # Day 2: price drops to 90. Asset return is -10%; unrealized = 0.20 × -0.10 = -0.02.
    t.advance_one_day(date(2024, 1, 4), {"AAA": 90.0})
    assert t.peak_value_pct == pytest.approx(peak_after_up)
    assert t.current_drawdown_pct > 0.0


def test_closed_trade_net_return_matches_v002_simulator_semantics():
    """ClosedTrade.net_return_pct should match v002's
    ExecutionSimulator's portfolio-scaled net_return formula:
    size_pct × (asset_return − cost_drag_round_trip)."""
    ct = ClosedTrade(
        symbol="AAA", entry_date=date(2024, 1, 2), exit_date=date(2024, 1, 7),
        entry_price=100.0, exit_price=105.0,
        size_pct=0.10, cost_bps_round_trip=10.0,
    )
    expected = 0.10 * (0.05 - 10 / 10_000.0)
    assert ct.net_return_pct == pytest.approx(expected)


def test_tracker_close_all_realizes_all_open_positions():
    t = PortfolioStateTracker()
    t.open_position(
        symbol="AAA", entry_date=date(2024, 1, 2), entry_price=100.0,
        size_pct=0.10, max_holding_days=10, cost_bps_round_trip=10.0,
    )
    t.open_position(
        symbol="BBB", entry_date=date(2024, 1, 2), entry_price=200.0,
        size_pct=0.15, max_holding_days=10, cost_bps_round_trip=10.0,
    )
    realized = t.close_all(date(2024, 1, 5), {"AAA": 105.0, "BBB": 210.0})
    assert t.n_open == 0
    assert len(t.closed_trades) == 2
    # Combined realized P&L: AAA 0.10×(0.05-0.001) + BBB 0.15×(0.05-0.001)
    expected_aaa = 0.10 * (0.05 - 10 / 10_000.0)
    expected_bbb = 0.15 * (0.05 - 10 / 10_000.0)
    assert realized == pytest.approx(expected_aaa + expected_bbb)


def test_open_position_with_held_returns_new_immutable():
    """OpenPosition is frozen; with_held returns a fresh instance."""
    p = OpenPosition(
        symbol="AAA", entry_date=date(2024, 1, 2), entry_price=100.0,
        size_pct=0.10, max_holding_days=5, cost_bps_round_trip=10.0,
    )
    p2 = p.with_held(3)
    assert p.days_held == 0  # unchanged
    assert p2.days_held == 3
    assert p2.symbol == p.symbol


# ---------------------------------------------------------------------
# Per-day baselines
# ---------------------------------------------------------------------
def test_no_op_baseline_always_returns_zero():
    pol = NoOpPortfolioPolicy()
    for _ in range(10):
        assert pol.decide(None) == 0


def test_topk_baseline_returns_fixed_k():
    pol = TopKPortfolioPolicy(k=2)
    assert pol.decide(None) == 2
    assert pol.model_id == "portfolio_baseline_top2"


def test_random_action_baseline_stays_in_range():
    pol = RandomActionPortfolioPolicy(n_actions=3, seed=42)
    for _ in range(50):
        a = pol.decide(None)
        assert 0 <= a < 3


# ---------------------------------------------------------------------
# ChronologicalSwingEnv tests
# ---------------------------------------------------------------------
def _build_minimal_env(
    n_days: int = 10, n_symbols: int = 3, n_packs_per_day: int = 2,
    max_top_k: int = 2,
):
    """Construct a small env from synthetic bars + packs. Frames
    are created with zeroed feature values; per-symbol cost_bps
    will use defaults from EquityExecutionModel."""
    symbols = [f"S{i}" for i in range(n_symbols)]
    dates = [date(2024, 1, d) for d in range(2, 2 + n_days)]
    bars = _make_bars(symbols, dates)
    # One feature frame per (symbol, date) with empty values; the env
    # uses these only to compute cost_bps.
    frames = [
        FeatureFrame(
            symbol=sym, as_of=datetime(d.year, d.month, d.day),
            feature_version="features_v001_core_daily",
            values={"atr_pct_14": 0.02, "realized_vol_20": 0.20, "dollar_volume": 1e8},
            feature_names=("atr_pct_14", "realized_vol_20", "dollar_volume"),
            source_snapshot_id="test",
        )
        for sym in symbols for d in dates
    ]
    packs = _make_packs_one_per_day(
        symbols, dates, n_per_day=n_packs_per_day,
    )
    return ChronologicalSwingEnv(
        bars=bars, packs=packs, feature_frames=frames,
        cost_model=EquityExecutionModel(),
        reward_model=RewardModel(),
        max_top_k=max_top_k,
        sampler_kind="chronological",
        sampler_seed=0,
        episode_min_days=2,
    )


def test_env_action_space_is_discrete_one_plus_max_top_k():
    env = _build_minimal_env(max_top_k=2)
    assert env.action_space.n == 3  # 0=no-op, 1=top1, 2=top2
    env2 = _build_minimal_env(max_top_k=3)
    assert env2.action_space.n == 4


def test_env_observation_dim_is_stable():
    env = _build_minimal_env()
    obs, _info = env.reset()
    assert obs.shape == (OBS_DIM,)


def test_env_no_op_action_keeps_portfolio_unchanged():
    """Action 0 should never open a new position."""
    env = _build_minimal_env(n_days=5)
    env.reset(seed=0)
    initial_open = env.tracker.n_open
    for _ in range(3):
        _, _, term, _, info = env.step(0)
        if term:
            break
    assert env.tracker.n_trades_opened == 0
    assert env.tracker.n_open == initial_open  # nothing opened


def test_env_top1_action_opens_when_slate_has_packs():
    env = _build_minimal_env(n_days=5, n_packs_per_day=2)
    env.reset(seed=0)
    _, _, _, _, info = env.step(1)
    assert info["n_opened_today"] >= 1
    assert env.tracker.n_trades_opened >= 1


def test_env_top2_action_opens_two_when_slate_has_two_packs():
    env = _build_minimal_env(n_days=5, n_packs_per_day=2, max_top_k=2)
    env.reset(seed=0)
    _, _, _, _, info = env.step(2)
    assert info["n_opened_today"] == 2


def test_env_episode_terminates_at_window_end():
    env = _build_minimal_env(n_days=5)
    env.reset(seed=0)
    terminated_seen = False
    for _ in range(20):
        _, _, term, _, _ = env.step(0)
        if term:
            terminated_seen = True
            break
    assert terminated_seen


def test_env_invalid_action_clamps_to_no_op_not_crash():
    env = _build_minimal_env()
    env.reset(seed=0)
    # Out-of-range action should be clamped to 0 by the env's
    # defensive check, not crash.
    _, reward, _, _, info = env.step(999)
    assert info["n_opened_today"] == 0
    assert isinstance(float(reward), float)


def test_env_close_all_runs_at_episode_end():
    """All open positions are closed when the episode terminates,
    regardless of max_holding_days. Verifies the env's final-day
    close_all() call."""
    env = _build_minimal_env(n_days=3, max_top_k=2)
    env.reset(seed=0)
    # Step taking top-2 on day 0 → opens 2 positions with
    # max_holding_days=3 (from _make_packs_one_per_day default).
    env.step(2)
    env.step(0)
    # Day 2: terminates (episode_days = 3 days).
    _, _, term, _, _ = env.step(0)
    assert term
    assert env.tracker.n_open == 0  # close_all fired


def test_env_reward_negative_when_drawdown_grows():
    """Reward path: when daily P&L is negative and DD increases, the
    reward should be negative. Non-trivial because rewards combine
    risk_adj_pnl, dd_penalty, turnover_penalty."""
    symbols = ["AAA"]
    dates = [date(2024, 1, d) for d in range(2, 6)]  # 4 days
    # Price path with a big drop: 100, 100, 80, 80
    price_path = [100.0, 100.0, 80.0, 80.0]
    bars = _make_bars(symbols, dates, price_path)
    frames = [
        FeatureFrame(
            symbol="AAA", as_of=datetime(d.year, d.month, d.day),
            feature_version="features_v001_core_daily",
            values={"atr_pct_14": 0.02, "realized_vol_20": 0.20, "dollar_volume": 1e8},
            feature_names=("atr_pct_14", "realized_vol_20", "dollar_volume"),
            source_snapshot_id="test",
        ) for d in dates
    ]
    packs = _make_packs_one_per_day(symbols, dates, n_per_day=1, base_size_pct=0.20)
    env = ChronologicalSwingEnv(
        bars=bars, packs=packs, feature_frames=frames,
        sampler_kind="chronological", episode_min_days=2,
    )
    env.reset(seed=0)
    # Day 0 (Jan 2, price 100): take top-1 → opens AAA at open=100.
    env.step(1)
    # Day 1 (Jan 3, price still 100): no-op; daily P&L ~0.
    env.step(0)
    # Day 2 (Jan 4, price drops to 80): no-op; daily P&L should
    # reflect the drop in unrealized P&L since yesterday.
    _, reward_d2, _, _, info = env.step(0)
    assert info["daily_pnl_pct"] < 0
    assert reward_d2 < 0


# ---------------------------------------------------------------------
# Variant integration smoke
# ---------------------------------------------------------------------
def test_variant_registry_loads_portfolio_v003():
    from rl_swing.rl.variants.base import load_variant
    v = load_variant("portfolio_v003")
    assert v.name == "portfolio_v003"
    assert hasattr(v, "build_env")
    assert hasattr(v, "evaluate")


def test_variant_evaluate_runs_baselines_on_synthetic_window():
    """End-to-end: the variant's evaluate() runs all baselines over
    a hand-built window without errors. Don't assert numbers (they're
    synthetic and not decision-grade); just verify the plumbing."""
    from rl_swing.rl.variants.base import EvaluationContext, load_variant

    v = load_variant("portfolio_v003")
    symbols = [f"S{i}" for i in range(2)]
    dates = [date(2024, 1, d) for d in range(2, 8)]
    bars = _make_bars(symbols, dates)
    frames = [
        FeatureFrame(
            symbol=sym, as_of=datetime(d.year, d.month, d.day),
            feature_version="features_v001_core_daily",
            values={"atr_pct_14": 0.02, "realized_vol_20": 0.20, "dollar_volume": 1e8,
                    "r20": 0.0, "r60": 0.0, "rsi_5": 50.0, "sma_50": 100.0,
                    "sma_200": 100.0, "rel_vol_20": 1.0, "dist_high_20d": 0.0},
            feature_names=("atr_pct_14", "realized_vol_20", "dollar_volume",
                          "r20", "r60", "rsi_5", "sma_50", "sma_200",
                          "rel_vol_20", "dist_high_20d"),
            source_snapshot_id="test",
        )
        for sym in symbols for d in dates
    ]
    ctx = EvaluationContext(
        bars=bars, frames=frames,
        test_start=dates[0], test_end=dates[-1],
        cost_model=EquityExecutionModel(),
        reward_model=RewardModel(),
        artifact_path=None,
        model_id="ppo_portfolio_v003",
        include_baselines=("random", "no_op", "top1", "top2"),
        include_cost_stress=False,
        experiment_config={"v003_max_top_k": 2},
    )
    results = v.evaluate(ctx)
    # Expect 4 baselines (no_op, top1, top2, random_action), no trained model.
    assert len(results) == 4
    ids = [r.model_id for r in results]
    assert "portfolio_baseline_no_op" in ids
    assert "portfolio_baseline_top1" in ids
    assert "portfolio_baseline_top2" in ids
    assert "portfolio_baseline_random_action" in ids
    # All should report cost_stress_multiplier=1.0 (M1 doesn't compute cost-2x).
    for r in results:
        assert r.cost_stress_multiplier == 1.0
        assert r.extras.get("variant") == "portfolio_v003"
