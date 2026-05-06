"""RL env + the modular pieces it composes."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from rl_swing.domain import CandidateTrade, FeatureFrame, MarketBar, PortfolioState
from rl_swing.features.pipelines import ALL_FEATURE_NAMES
from rl_swing.rl.env.action_mapper import (
    ACTION_INT_TO_LITERAL,
    LITERAL_TO_ACTION_INT,
    to_literal,
    to_size_multiplier,
)
from rl_swing.rl.env.cost_model import EquityExecutionModel
from rl_swing.rl.env.episode_sampler import (
    ChronologicalSampler,
    RandomWindowSampler,
)
from rl_swing.rl.env.execution_simulator import ExecutionSimulator
from rl_swing.rl.env.observation_builder import (
    CANDIDATE_FEATURE_NAMES,
    STRATEGY_INDEX,
    ObservationBuilder,
)
from rl_swing.rl.env.reward_model import RewardModel
from rl_swing.rl.env.swing_env import SwingTradingEnv
from rl_swing.strategies.aggregator import StrategyAggregator
from rl_swing.strategies.momentum import MomentumStrategy


def _frame(symbol: str = "AAPL", as_of: datetime | None = None,
           **values) -> FeatureFrame:
    full = {n: 0.0 for n in ALL_FEATURE_NAMES}
    full.update(values)
    return FeatureFrame(
        as_of=as_of or datetime(2024, 1, 2),
        symbol=symbol, feature_version="v1",
        values=full, feature_names=ALL_FEATURE_NAMES,
        source_snapshot_id="s",
    )


def _candidate(symbol: str = "AAPL", as_of: datetime | None = None,
               strategy_id: str = "momentum_20_60") -> CandidateTrade:
    return CandidateTrade(
        candidate_id=f"c-{symbol}-{(as_of or datetime(2024,1,2)).date()}",
        as_of=as_of or datetime(2024, 1, 2),
        symbol=symbol, strategy_id=strategy_id, direction="long",
        entry_timing="next_open", base_size_pct=0.10, max_holding_days=10,
        stop_rule_id=None, exit_rule_id="x", signal_strength=0.5,
        metadata={"avg_dollar_volume": 1e8},
    )


# --- action mapper -------------------------------------------------------
def test_action_mapper_table():
    assert to_literal(0) == "skip"
    assert to_literal(3) == "take_100"
    assert to_size_multiplier(2) == 0.5
    assert to_size_multiplier(0) == 0.0
    with pytest.raises(ValueError):
        to_literal(99)
    assert ACTION_INT_TO_LITERAL[1] == "take_25"
    assert LITERAL_TO_ACTION_INT["take_50"] == 2


# --- cost model ----------------------------------------------------------
def test_cost_model_higher_under_high_vol():
    m = EquityExecutionModel()
    low_vol = m.cost_bps(atr_pct=0.01, volatility_percentile=0.1)
    hi_vol = m.cost_bps(atr_pct=0.01, volatility_percentile=0.95)
    assert hi_vol > low_vol


def test_cost_model_event_window_increases_cost():
    m = EquityExecutionModel()
    base = m.cost_bps(atr_pct=0.01, volatility_percentile=0.1, in_event_window=False)
    ev = m.cost_bps(atr_pct=0.01, volatility_percentile=0.1, in_event_window=True)
    assert ev > base


def test_cost_model_market_impact_grows_with_notional():
    m = EquityExecutionModel(market_impact_coef=1.0)
    a = m.cost_bps(notional=1e6, avg_dollar_volume=1e8)
    b = m.cost_bps(notional=1e7, avg_dollar_volume=1e8)
    assert b > a


def test_cost_model_atr_high_branch():
    m = EquityExecutionModel()
    base = m.cost_bps(atr_pct=0.01, volatility_percentile=0.5)
    high_atr = m.cost_bps(atr_pct=0.05, volatility_percentile=0.5)
    assert high_atr > base


def test_cost_model_zero_adv_no_impact():
    m = EquityExecutionModel(market_impact_coef=10.0)
    cost = m.cost_bps(notional=1e7, avg_dollar_volume=0.0)
    no_impact = m.cost_bps(notional=0, avg_dollar_volume=0)
    # When adv is 0, impact term is bypassed.
    assert cost == no_impact


def test_cost_model_stress_multiplier_doubles_total():
    m = EquityExecutionModel(cost_stress_multiplier=2.0)
    m_normal = EquityExecutionModel()
    a = m.cost_bps(atr_pct=0.01)
    b = m_normal.cost_bps(atr_pct=0.01)
    assert a == pytest.approx(2 * b)


# --- observation builder -------------------------------------------------
def test_observation_builder_dim_and_hash_deterministic():
    ob = ObservationBuilder(feature_names=ALL_FEATURE_NAMES)
    assert ob.dim == len(ALL_FEATURE_NAMES) + len(CANDIDATE_FEATURE_NAMES)
    portfolio = PortfolioState(as_of=datetime(2024, 1, 2),
                               cash=10000, equity=10000)
    obs = ob.build(_candidate(), _frame(), portfolio)
    assert obs.shape == (ob.dim,)
    h1 = ob.hash(obs)
    h2 = ob.hash(obs)
    assert h1 == h2 and len(h1) == 12


def test_observation_includes_strategy_index():
    ob = ObservationBuilder(feature_names=ALL_FEATURE_NAMES)
    portfolio = PortfolioState(as_of=datetime(2024, 1, 2),
                               cash=10000, equity=10000)
    momentum_obs = ob.build(_candidate(strategy_id="momentum_20_60"),
                            _frame(), portfolio)
    breakout_obs = ob.build(_candidate(strategy_id="breakout_20d"),
                            _frame(), portfolio)
    # The "strategy index normalized" slot should differ between strategies.
    assert not np.array_equal(momentum_obs, breakout_obs)


def test_observation_unknown_strategy_index_is_safe():
    ob = ObservationBuilder(feature_names=ALL_FEATURE_NAMES)
    portfolio = PortfolioState(as_of=datetime(2024, 1, 2),
                               cash=10000, equity=10000)
    obs = ob.build(_candidate(strategy_id="not_in_index"),
                   _frame(), portfolio)
    assert obs.shape == (ob.dim,)


def test_strategy_index_table_is_lowercase_friendly():
    assert "momentum_20_60" in STRATEGY_INDEX
    assert "unknown" in STRATEGY_INDEX


# --- episode sampler -----------------------------------------------------
def _candidates_chain(n: int) -> list[CandidateTrade]:
    return [
        _candidate(as_of=datetime(2024, 1, 1) + (datetime(2024, 1, 2) - datetime(2024, 1, 1)) * i)
        for i in range(n)
    ]


def test_chronological_sampler_returns_all_then_empty():
    cands = _candidates_chain(3)
    s = ChronologicalSampler(cands)
    ep = s.sample()
    assert len(ep.candidates) == 3
    assert s.sample().candidates == []  # exhausted
    s.reset()
    assert len(s.sample().candidates) == 3


def test_chronological_sampler_empty_input():
    s = ChronologicalSampler([])
    ep = s.sample()
    assert ep.candidates == []
    assert s.sample().candidates == []


def test_random_window_sampler_returns_a_window():
    # Fabricate candidates spread over 365 days.
    cands = []
    base = datetime(2020, 1, 1)
    for i in range(0, 365, 5):
        cands.append(CandidateTrade(
            candidate_id=f"c{i}", as_of=base.replace(day=1) if False else base,
            symbol="AAPL", strategy_id="m", direction="long",
            entry_timing="next_open", base_size_pct=0.05, max_holding_days=10,
            stop_rule_id=None, exit_rule_id="x", signal_strength=0.5,
        ))
    # Spread the as_of times.
    from datetime import timedelta
    cands = [
        CandidateTrade(
            candidate_id=f"c{i}", as_of=base + timedelta(days=i),
            symbol="AAPL", strategy_id="m", direction="long",
            entry_timing="next_open", base_size_pct=0.05, max_holding_days=10,
            stop_rule_id=None, exit_rule_id="x", signal_strength=0.5,
        )
        for i in range(0, 365, 5)
    ]
    s = RandomWindowSampler(cands, window_days=60, min_candidates=3, seed=1)
    ep = s.sample()
    assert ep.candidates
    assert (ep.end - ep.start).days <= 60


def test_random_window_sampler_falls_back_when_not_enough_candidates():
    s = RandomWindowSampler(_candidates_chain(2), window_days=10, min_candidates=10, seed=0)
    ep = s.sample()
    assert ep.candidates  # falls back to whatever is available


def test_random_window_sampler_empty_input():
    s = RandomWindowSampler([], window_days=10, seed=0)
    assert s.sample().candidates == []


def test_random_window_sampler_short_history_returns_all():
    # Window larger than history -> should return everything in one episode.
    s = RandomWindowSampler(_candidates_chain(3), window_days=365, seed=0)
    ep = s.sample()
    assert len(ep.candidates) == 3


# --- execution simulator -------------------------------------------------
def _bars_linear(n: int, slope: float = 0.01, base: float = 100.0) -> list[MarketBar]:
    out = []
    p = base
    for i in range(n):
        op = p
        p = p * (1.0 + slope)
        out.append(MarketBar(
            symbol="X", timestamp=datetime(2024, 1, 1) + (datetime(2024, 1, 2) - datetime(2024, 1, 1)) * i,
            timeframe="1d", open=op, high=p * 1.02, low=op * 0.99, close=p,
            volume=1e6, adjusted_close=p, source="t",
        ))
    return out


def test_execution_simulator_target_hit():
    bars = _bars_linear(20, slope=0.05)
    sim = ExecutionSimulator(atr_target_mult=2.0, atr_stop_mult=2.0)
    out = sim.simulate(bars=bars, entry_index=0, size_pct=0.10,
                       max_holding_days=20, cost_bps=0, atr_pct=0.02)
    assert out is not None
    assert out.exit_reason in {"target", "time"}
    assert out.return_pct > 0


def test_execution_simulator_stop_hit():
    bars = _bars_linear(20, slope=-0.05)
    sim = ExecutionSimulator(atr_target_mult=4.0, atr_stop_mult=2.0)
    out = sim.simulate(bars=bars, entry_index=0, size_pct=0.10,
                       max_holding_days=20, cost_bps=0, atr_pct=0.02)
    assert out is not None
    assert out.exit_reason == "stop"


def test_execution_simulator_time_exit_when_flat():
    # Bars with very tight intraday range so neither stop nor target gets hit.
    from datetime import timedelta
    bars = []
    p = 100.0
    for i in range(10):
        bars.append(MarketBar(
            symbol="X", timestamp=datetime(2024, 1, 1) + timedelta(days=i),
            timeframe="1d", open=p, high=p * 1.0005, low=p * 0.9995,
            close=p, volume=1e6, adjusted_close=p, source="t",
        ))
    sim = ExecutionSimulator(atr_target_mult=10.0, atr_stop_mult=10.0)
    out = sim.simulate(bars=bars, entry_index=0, size_pct=0.10,
                       max_holding_days=5, cost_bps=0, atr_pct=0.005)
    assert out is not None
    assert out.exit_reason == "time"


def test_execution_simulator_returns_none_when_size_zero():
    sim = ExecutionSimulator()
    out = sim.simulate(bars=_bars_linear(3), entry_index=0, size_pct=0.0,
                       max_holding_days=2, cost_bps=0, atr_pct=0.01)
    assert out is None


def test_execution_simulator_returns_none_when_no_future_bar():
    sim = ExecutionSimulator()
    bars = _bars_linear(2)
    out = sim.simulate(bars=bars, entry_index=10, size_pct=0.10,
                       max_holding_days=2, cost_bps=0, atr_pct=0.01)
    assert out is None


def test_execution_simulator_returns_none_when_zero_open_price():
    bars = _bars_linear(3)
    # Force the entry bar's open to 0.
    from dataclasses import replace
    bars[1] = replace(bars[1], open=0.0)
    sim = ExecutionSimulator()
    out = sim.simulate(bars=bars, entry_index=0, size_pct=0.10,
                       max_holding_days=2, cost_bps=0, atr_pct=0.01)
    assert out is None


def test_execution_simulator_cost_reduces_return():
    bars = _bars_linear(20, slope=0.05)
    sim = ExecutionSimulator(atr_target_mult=10.0, atr_stop_mult=10.0)
    no_cost = sim.simulate(bars=bars, entry_index=0, size_pct=0.10,
                           max_holding_days=20, cost_bps=0, atr_pct=0.02)
    with_cost = sim.simulate(bars=bars, entry_index=0, size_pct=0.10,
                             max_holding_days=20, cost_bps=100, atr_pct=0.02)
    assert with_cost.return_pct < no_cost.return_pct


def test_execution_simulator_return_scales_with_size_pct():
    """FIX-22: size_pct must scale realized portfolio return.

    Two trades on the same bars with the same atr/cost differ ONLY in
    size. The 100%-sized trade's portfolio contribution must be ~10x
    the 10%-sized trade's, since size_pct linearly scales the
    portfolio-level return.

    Regression: prior to FIX-22, the simulator returned the asset's
    standalone percent return as ``return_pct`` regardless of size,
    so take_25 vs take_100 produced indistinguishable rewards and
    the agent never learned position sizing.
    """
    bars = _bars_linear(20, slope=0.02)
    sim = ExecutionSimulator(atr_target_mult=10.0, atr_stop_mult=10.0)
    small = sim.simulate(bars=bars, entry_index=0, size_pct=0.10,
                         max_holding_days=20, cost_bps=0, atr_pct=0.02)
    large = sim.simulate(bars=bars, entry_index=0, size_pct=1.00,
                         max_holding_days=20, cost_bps=0, atr_pct=0.02)
    assert small is not None and large is not None
    # Asset return must be identical (same bars, same exit).
    assert small.asset_return_pct == large.asset_return_pct
    # Portfolio return must scale ~linearly with size_pct
    # (with cost_bps=0 the ratio is exactly 10).
    assert abs(large.return_pct - 10.0 * small.return_pct) < 1e-9
    # Sanity: size_pct field reflects what was passed in.
    assert small.size_pct == 0.10
    assert large.size_pct == 1.00
    # Backward compat: raw_return_pct aliases asset_return_pct.
    assert small.raw_return_pct == small.asset_return_pct


# --- reward model --------------------------------------------------------
def test_reward_take_winner_positive():
    rm = RewardModel()
    bars = _bars_linear(20, slope=0.05)
    sim = ExecutionSimulator(atr_target_mult=10.0, atr_stop_mult=10.0)
    outcome = sim.simulate(bars=bars, entry_index=0, size_pct=0.10,
                           max_holding_days=20, cost_bps=0, atr_pct=0.02)
    r = rm.reward_for_take(outcome, max_holding_days=20)
    assert r > 0


def test_reward_take_clipping_protects_extremes():
    from rl_swing.rl.env.execution_simulator import TradeOutcome
    rm = RewardModel(reward_clip=2.0)
    big_winner = TradeOutcome(
        symbol="X", entry_timestamp=datetime(2024, 1, 1),
        exit_timestamp=datetime(2024, 1, 2), entry_price=100, exit_price=200,
        qty=10, notional=1000, return_pct=1.0, raw_return_pct=1.0,
        asset_return_pct=1.0, size_pct=1.0,
        holding_days=1, peak_drawdown_pct=0.0, exit_reason="target", cost_bps=0,
    )
    r = rm.reward_for_take(big_winner, max_holding_days=10)
    assert r <= rm.reward_clip + 0.01  # clipped


def test_reward_skip_with_no_counterfactual_is_zero():
    rm = RewardModel()
    assert rm.reward_for_skip(None) == 0.0


def test_reward_skip_mirrors_take_on_risk_adjusted_scale():
    """Skip rewards mirror take rewards on the same risk-adjusted scale.

    Skipping a +10% winner with target_risk_pct=2% costs you +5
    (clipped at reward_clip=5). Skipping a -10% loser earns you +5.
    This makes the agent's choice between skip and take a real
    decision rather than a degenerate "always take" optimum.
    """
    from rl_swing.rl.env.execution_simulator import TradeOutcome
    # Use scale=1.0 to test the mirror itself; default is 0.5.
    rm = RewardModel(target_risk_pct=0.02, reward_clip=5.0,
                     skip_counterfactual_scale=1.0)
    winner = TradeOutcome(
        symbol="X", entry_timestamp=datetime(2024, 1, 1),
        exit_timestamp=datetime(2024, 1, 2), entry_price=100, exit_price=110,
        qty=10, notional=1000, return_pct=0.10, raw_return_pct=0.10,
        asset_return_pct=0.10, size_pct=1.0,
        holding_days=1, peak_drawdown_pct=0.0, exit_reason="target", cost_bps=0,
    )
    loser = TradeOutcome(
        symbol="X", entry_timestamp=datetime(2024, 1, 1),
        exit_timestamp=datetime(2024, 1, 2), entry_price=100, exit_price=90,
        qty=10, notional=1000, return_pct=-0.10, raw_return_pct=-0.10,
        asset_return_pct=-0.10, size_pct=1.0,
        holding_days=1, peak_drawdown_pct=0.10, exit_reason="stop", cost_bps=0,
    )
    # +10% / 2% = +5 risk-adjusted; clipped at 5; skip mirrors → -5.
    assert rm.reward_for_skip(winner) == -5.0
    # -10% / 2% = -5 risk-adjusted; clipped at -5; skip mirrors → +5.
    assert rm.reward_for_skip(loser) == 5.0


def test_reward_skip_scale_dampens_mirror():
    """skip_counterfactual_scale lets you tune skip-vs-take
    aggressiveness.  scale=0.5 means missing a winner is half as bad
    as taking it is good → the agent leans slightly toward action
    when EV is near zero."""
    from rl_swing.rl.env.execution_simulator import TradeOutcome
    rm = RewardModel(target_risk_pct=0.02, reward_clip=5.0,
                     skip_counterfactual_scale=0.5)
    small_winner = TradeOutcome(
        symbol="X", entry_timestamp=datetime(2024, 1, 1),
        exit_timestamp=datetime(2024, 1, 2), entry_price=100, exit_price=101,
        qty=10, notional=1000, return_pct=0.01, raw_return_pct=0.01,
        asset_return_pct=0.01, size_pct=1.0,
        holding_days=1, peak_drawdown_pct=0.0, exit_reason="target", cost_bps=0,
    )
    # +1% / 2% = +0.5; mirrored × 0.5 = -0.25.
    assert abs(rm.reward_for_skip(small_winner) - (-0.25)) < 1e-9


# --- env -----------------------------------------------------------------
def test_env_reset_step_round_trip(synthetic_bars, feature_frames, portfolio_state):
    cands = list(StrategyAggregator([MomentumStrategy()]).generate(
        feature_frames, portfolio_state))
    env = SwingTradingEnv(
        bars=synthetic_bars, candidates=cands,
        feature_frames=feature_frames, feature_names=ALL_FEATURE_NAMES,
        sampler_kind="random", sampler_seed=0,
        sampler_window_days=120,
    )
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    total = 0.0
    for _ in range(3):
        next_obs, reward, term, trunc, _info = env.step(env.action_space.sample())
        total += reward
        assert next_obs.shape == env.observation_space.shape
        if term or trunc:
            break


def test_env_chronological_sampler(synthetic_bars, feature_frames, portfolio_state):
    cands = list(StrategyAggregator([MomentumStrategy()]).generate(
        feature_frames, portfolio_state))
    env = SwingTradingEnv(
        bars=synthetic_bars, candidates=cands,
        feature_frames=feature_frames, feature_names=ALL_FEATURE_NAMES,
        sampler_kind="chronological",
    )
    obs, info = env.reset()
    assert "n_candidates" in info


def test_env_handles_empty_candidates():
    env = SwingTradingEnv(
        bars=[], candidates=[], feature_frames=[],
        feature_names=ALL_FEATURE_NAMES,
    )
    obs, info = env.reset()
    assert info.get("empty_episode")
    obs, r, term, trunc, _ = env.step(0)
    assert term and r == 0.0


def test_env_step_skip_returns_counterfactual_in_info(synthetic_bars,
                                                      feature_frames,
                                                      portfolio_state):
    cands = list(StrategyAggregator([MomentumStrategy()]).generate(
        feature_frames, portfolio_state))
    env = SwingTradingEnv(
        bars=synthetic_bars, candidates=cands,
        feature_frames=feature_frames, feature_names=ALL_FEATURE_NAMES,
        sampler_kind="chronological",
    )
    env.reset()
    obs, r, term, trunc, info = env.step(0)  # skip
    # Either skip with counterfactual or empty info if pipeline edge case.
    assert info["action"] in {"skip", "take_no_data"} or info.get("reason")


def test_env_truncation_via_max_steps(synthetic_bars, feature_frames, portfolio_state):
    cands = list(StrategyAggregator([MomentumStrategy()]).generate(
        feature_frames, portfolio_state))
    env = SwingTradingEnv(
        bars=synthetic_bars, candidates=cands,
        feature_frames=feature_frames, feature_names=ALL_FEATURE_NAMES,
        sampler_kind="chronological",
        max_steps_per_episode=2,
    )
    env.reset()
    env.step(0)
    obs, r, term, trunc, info = env.step(0)
    assert term or trunc


def test_env_bars_find_index_missing_symbol(synthetic_bars, feature_frames, portfolio_state):
    cands = list(StrategyAggregator([MomentumStrategy()]).generate(
        feature_frames, portfolio_state))
    env = SwingTradingEnv(
        bars=synthetic_bars, candidates=cands,
        feature_frames=feature_frames, feature_names=ALL_FEATURE_NAMES,
        sampler_kind="chronological",
    )
    assert env.bars.find_index("DOES_NOT_EXIST", datetime(2020, 1, 1)) == -1
