"""Feature-pipeline + technical helpers + leakage checks."""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest

from rl_swing.domain import FeatureFrame, MarketBar
from rl_swing.features import technical as tf
from rl_swing.features.leakage_checks import (
    LeakageError,
    assert_features_finite,
    assert_no_future_bars,
)
from rl_swing.features.pipelines import (
    ALL_FEATURE_NAMES,
    REGIME_FEATURE_NAMES,
    TECHNICAL_FEATURE_NAMES,
    CoreDailyPipeline,
)


# --- technical helpers ----------------------------------------------------
def test_returns_zero_when_window_exceeds_length():
    arr = np.array([100.0, 101.0])
    assert (tf.returns(arr, 10) == 0).all()


def test_returns_basic():
    arr = np.array([100.0, 110.0, 121.0])
    out = tf.returns(arr, 1)
    assert out[1] == pytest.approx(0.1)
    assert out[2] == pytest.approx(0.1)


def test_sma_handles_short_window():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = tf.sma(arr, 3)
    assert out[2] == pytest.approx(2.0)
    assert out[4] == pytest.approx(4.0)


def test_sma_window_zero_returns_copy():
    arr = np.array([1.0, 2.0])
    assert (tf.sma(arr, 0) == arr).all()


def test_rsi_short_input():
    arr = np.array([100.0, 101.0])
    out = tf.rsi(arr, 14)
    assert (out == 50.0).all()


def test_rsi_pure_uptrend():
    arr = np.linspace(100, 200, 30)
    out = tf.rsi(arr, 14)
    assert out[-1] > 90  # pure up = saturated


def test_rsi_pure_downtrend():
    arr = np.linspace(200, 100, 30)
    out = tf.rsi(arr, 14)
    assert out[-1] < 10


def test_atr_well_defined():
    n = 30
    high = np.linspace(100, 110, n)
    low = np.linspace(99, 109, n)
    close = (high + low) / 2.0
    out = tf.atr(high, low, close, 14)
    assert out[-1] > 0
    assert math.isfinite(out[-1])


def test_realized_vol_zero_for_flat_prices():
    arr = np.full(30, 100.0)
    out = tf.realized_vol(arr, 20, annualize=False)
    assert out[-1] == 0.0


def test_zscore_zero_when_std_zero():
    arr = np.full(30, 100.0)
    out = tf.zscore(arr, 20)
    assert out[-1] == 0.0


def test_relative_volume_handles_zero_average():
    vol = np.array([0.0, 0.0, 0.0, 100.0, 100.0])
    out = tf.relative_volume(vol, 3)
    assert math.isfinite(out[-1])


def test_distance_from_high_negative_at_pullback():
    arr = np.array([100.0, 110.0, 120.0, 115.0, 110.0, 105.0])
    out = tf.distance_from_high(arr, 5)
    assert out[-1] < 0


def test_distance_from_low_positive_after_bounce():
    arr = np.array([110.0, 100.0, 95.0, 105.0, 115.0])
    out = tf.distance_from_low(arr, 5)
    assert out[-1] > 0


# --- pipeline -------------------------------------------------------------
def test_core_daily_feature_names_exposed():
    pipe = CoreDailyPipeline()
    assert pipe.feature_version == "features_v001_core_daily"
    assert "return_20d" in pipe.feature_names
    for n in TECHNICAL_FEATURE_NAMES:
        assert n in pipe.feature_names
    for n in REGIME_FEATURE_NAMES:
        assert n in pipe.feature_names
    assert pipe.feature_names == ALL_FEATURE_NAMES


def test_core_daily_skips_short_history():
    # Fewer than 60 bars -> no frames produced.
    bars = [
        MarketBar(symbol="X", timestamp=datetime(2024, 1, i + 1),
                  timeframe="1d", open=100, high=101, low=99, close=100.5,
                  volume=1e6, adjusted_close=100.5, source="t")
        for i in range(30)
    ]
    pipe = CoreDailyPipeline()
    assert list(pipe.build(bars)) == []


def test_core_daily_emits_frames_with_finite_values(synthetic_bars):
    pipe = CoreDailyPipeline()
    frames = list(pipe.build(synthetic_bars))
    assert frames
    assert_features_finite(frames)


def test_core_daily_produces_regime_features_when_spy_present(synthetic_bars):
    pipe = CoreDailyPipeline()
    frames = list(pipe.build(synthetic_bars))
    seen_nonzero_spy = any(
        f.values["spy_return_20d"] != 0.0 for f in frames if f.symbol != "SPY"
    )
    assert seen_nonzero_spy


def test_core_daily_falls_back_when_no_spy_bars():
    # Drop SPY entirely.
    from datetime import date

    from rl_swing.adapters.data.synthetic_provider import SyntheticProvider
    prov = SyntheticProvider(regime="momentum", seed=11)
    bars = list(prov.get_bars(["AAPL", "MSFT"], date(2018, 1, 1), date(2020, 12, 31)))
    pipe = CoreDailyPipeline()
    frames = list(pipe.build(bars))
    assert frames
    # When SPY absent, regime features should default to 0.0.
    assert all(f.values["spy_return_20d"] == 0.0 for f in frames)


# --- leakage --------------------------------------------------------------
def test_leakage_detect_no_source_bars():
    f = FeatureFrame(
        as_of=datetime(2024, 1, 2),
        symbol="GHOST", feature_version="v",
        values={"a": 1.0}, feature_names=("a",),
        source_snapshot_id="x",
    )
    with pytest.raises(LeakageError):
        assert_no_future_bars([f], [])


def test_leakage_detect_pre_dating_bars():
    bar = MarketBar(
        symbol="A", timestamp=datetime(2024, 6, 1), timeframe="1d",
        open=1, high=1, low=1, close=1, volume=1, adjusted_close=1, source="t",
    )
    bad_frame = FeatureFrame(
        as_of=datetime(2020, 1, 1),
        symbol="A", feature_version="v",
        values={"a": 1.0}, feature_names=("a",),
        source_snapshot_id="x",
    )
    with pytest.raises(LeakageError):
        assert_no_future_bars([bad_frame], [bar])


def test_leakage_pass_when_frames_inside_bars():
    bar = MarketBar(
        symbol="A", timestamp=datetime(2024, 1, 1), timeframe="1d",
        open=1, high=1, low=1, close=1, volume=1, adjusted_close=1, source="t",
    )
    frame = FeatureFrame(
        as_of=datetime(2024, 1, 1),
        symbol="A", feature_version="v",
        values={"a": 1.0}, feature_names=("a",),
        source_snapshot_id="x",
    )
    assert_no_future_bars([frame], [bar])


def test_finite_check_blocks_nan():
    f = FeatureFrame(
        as_of=datetime(2024, 1, 1),
        symbol="A", feature_version="v",
        values={"a": float("nan")}, feature_names=("a",),
        source_snapshot_id="x",
    )
    with pytest.raises(LeakageError):
        assert_features_finite([f])


def test_finite_check_blocks_inf():
    f = FeatureFrame(
        as_of=datetime(2024, 1, 1),
        symbol="A", feature_version="v",
        values={"a": float("inf")}, feature_names=("a",),
        source_snapshot_id="x",
    )
    with pytest.raises(LeakageError):
        assert_features_finite([f])
