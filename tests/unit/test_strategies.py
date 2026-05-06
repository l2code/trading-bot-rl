"""Strategy unit tests — fires when configured, doesn't fire otherwise."""
from __future__ import annotations

from datetime import datetime

import pytest

from rl_swing.domain import FeatureFrame, PortfolioState
from rl_swing.features.pipelines import ALL_FEATURE_NAMES
from rl_swing.strategies.aggregator import StrategyAggregator
from rl_swing.strategies.breakout import BreakoutStrategy
from rl_swing.strategies.mean_reversion import RsiMeanReversionStrategy
from rl_swing.strategies.momentum import MomentumStrategy
from rl_swing.strategies.trend_following import TrendFollowingStrategy


def _frame(symbol: str = "AAPL", **values) -> FeatureFrame:
    full = {n: 0.0 for n in ALL_FEATURE_NAMES}
    full.update(values)
    return FeatureFrame(
        as_of=datetime(2024, 1, 2),
        symbol=symbol, feature_version="v1",
        values=full, feature_names=ALL_FEATURE_NAMES,
        source_snapshot_id="s",
    )


@pytest.fixture
def empty_portfolio():
    return PortfolioState(
        as_of=datetime(2024, 1, 2), cash=100_000, equity=100_000,
    )


# --- momentum -------------------------------------------------------------
def test_momentum_fires_when_uptrend_clean(empty_portfolio):
    s = MomentumStrategy()
    f = _frame(
        return_20d=0.10, close_vs_sma_50=0.05,
        close_vs_sma_200=0.10, spy_return_20d=0.01,
    )
    out = list(s.generate([f], empty_portfolio))
    assert len(out) == 1
    cand = out[0]
    assert cand.strategy_id == "momentum_20_60"
    assert cand.direction == "long"
    assert 0 < cand.signal_strength <= 1


def test_momentum_skips_when_below_sma200(empty_portfolio):
    s = MomentumStrategy()
    f = _frame(return_20d=0.10, close_vs_sma_50=0.05, close_vs_sma_200=-0.05)
    assert list(s.generate([f], empty_portfolio)) == []


def test_momentum_skips_when_below_sma50(empty_portfolio):
    s = MomentumStrategy()
    f = _frame(return_20d=0.10, close_vs_sma_50=-0.05, close_vs_sma_200=0.05)
    assert list(s.generate([f], empty_portfolio)) == []


def test_momentum_skips_when_negative_return(empty_portfolio):
    s = MomentumStrategy()
    f = _frame(return_20d=-0.05)
    assert list(s.generate([f], empty_portfolio)) == []


def test_momentum_min_relative_strength(empty_portfolio):
    s = MomentumStrategy(min_relative_strength=0.05)
    f = _frame(return_20d=0.02, close_vs_sma_50=0.01, close_vs_sma_200=0.01,
               spy_return_20d=0.01)
    # rs = 0.02 - 0.01 = 0.01 < 0.05 -> skip
    assert list(s.generate([f], empty_portfolio)) == []


# --- mean reversion -------------------------------------------------------
def test_mean_reversion_fires_when_oversold(empty_portfolio):
    s = RsiMeanReversionStrategy(rsi_window=5)
    f = _frame(rsi_5=20.0, close_vs_sma_50=0.02, spy_above_sma_200=1.0)
    out = list(s.generate([f], empty_portfolio))
    assert len(out) == 1


def test_mean_reversion_skips_when_market_below_sma200(empty_portfolio):
    s = RsiMeanReversionStrategy(rsi_window=5)
    f = _frame(rsi_5=20.0, close_vs_sma_50=0.02, spy_above_sma_200=0.0)
    assert list(s.generate([f], empty_portfolio)) == []


def test_mean_reversion_skips_when_not_oversold(empty_portfolio):
    s = RsiMeanReversionStrategy(rsi_window=5, rsi_threshold=25.0)
    f = _frame(rsi_5=40.0, close_vs_sma_50=0.02, spy_above_sma_200=1.0)
    assert list(s.generate([f], empty_portfolio)) == []


def test_mean_reversion_skips_when_downtrend(empty_portfolio):
    s = RsiMeanReversionStrategy(rsi_window=5, require_uptrend=True)
    f = _frame(rsi_5=20.0, close_vs_sma_50=-0.05, spy_above_sma_200=1.0)
    assert list(s.generate([f], empty_portfolio)) == []


def test_mean_reversion_unknown_window_falls_back_to_rsi5(empty_portfolio):
    s = RsiMeanReversionStrategy(rsi_window=7)  # not in (2,5,14)
    f = _frame(rsi_5=20.0, close_vs_sma_50=0.02, spy_above_sma_200=1.0)
    assert list(s.generate([f], empty_portfolio))


# --- breakout -------------------------------------------------------------
def test_breakout_fires_at_new_high(empty_portfolio):
    s = BreakoutStrategy()
    f = _frame(distance_from_20d_high=0.0,
               relative_volume_20=1.5, spy_above_sma_50=1.0)
    out = list(s.generate([f], empty_portfolio))
    assert len(out) == 1


def test_breakout_skips_far_from_high(empty_portfolio):
    s = BreakoutStrategy()
    f = _frame(distance_from_20d_high=-0.05,
               relative_volume_20=1.5, spy_above_sma_50=1.0)
    assert list(s.generate([f], empty_portfolio)) == []


def test_breakout_skips_low_volume(empty_portfolio):
    s = BreakoutStrategy(min_relative_volume=1.5)
    f = _frame(distance_from_20d_high=0.0,
               relative_volume_20=0.5, spy_above_sma_50=1.0)
    assert list(s.generate([f], empty_portfolio)) == []


def test_breakout_skips_in_bear_regime(empty_portfolio):
    s = BreakoutStrategy()
    f = _frame(distance_from_20d_high=0.0,
               relative_volume_20=2.0, spy_above_sma_50=0.0)
    assert list(s.generate([f], empty_portfolio)) == []


# --- trend following ------------------------------------------------------
def test_trend_following_fires_in_uptrend_pullback_resolution(empty_portfolio):
    s = TrendFollowingStrategy()
    f = _frame(
        close_vs_sma_50=0.05, close_vs_sma_200=0.10,
        return_5d=0.02, atr_pct_14=0.02,
    )
    out = list(s.generate([f], empty_portfolio))
    assert len(out) == 1


def test_trend_following_skips_when_atr_too_high(empty_portfolio):
    s = TrendFollowingStrategy(max_atr_pct=0.04)
    f = _frame(
        close_vs_sma_50=0.05, close_vs_sma_200=0.10,
        return_5d=0.02, atr_pct_14=0.10,
    )
    assert list(s.generate([f], empty_portfolio)) == []


def test_trend_following_skips_when_no_uptrend(empty_portfolio):
    s = TrendFollowingStrategy()
    f = _frame(close_vs_sma_50=-0.01, close_vs_sma_200=0.10, return_5d=0.02)
    assert list(s.generate([f], empty_portfolio)) == []


def test_trend_following_skips_when_negative_recent_return(empty_portfolio):
    s = TrendFollowingStrategy()
    f = _frame(close_vs_sma_50=0.05, close_vs_sma_200=0.10, return_5d=-0.02)
    assert list(s.generate([f], empty_portfolio)) == []


# --- aggregator -----------------------------------------------------------
def test_aggregator_dedupes_by_symbol_date(empty_portfolio):
    f = _frame(
        return_20d=0.10, close_vs_sma_50=0.05, close_vs_sma_200=0.10,
        rsi_5=20.0, spy_above_sma_200=1.0,
    )
    agg = StrategyAggregator([
        MomentumStrategy(),
        RsiMeanReversionStrategy(),
    ])
    out = list(agg.generate([f], empty_portfolio))
    assert len(out) == 1, "should keep only the higher-strength candidate"


def test_aggregator_keeps_distinct_dates(empty_portfolio):
    f1 = _frame(return_20d=0.10, close_vs_sma_50=0.05, close_vs_sma_200=0.10)
    f2 = FeatureFrame(
        as_of=datetime(2024, 1, 3),
        symbol="AAPL", feature_version="v1",
        values=f1.values, feature_names=ALL_FEATURE_NAMES,
        source_snapshot_id="s",
    )
    agg = StrategyAggregator([MomentumStrategy()])
    out = list(agg.generate([f1, f2], empty_portfolio))
    assert len(out) == 2
