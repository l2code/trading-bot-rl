"""FIX-24 (P1): walk-forward / trainer load bars with feature-warmup.

The pipeline includes long-lookback features (sma_200, return_60d,
atr_pct_14, etc.). Loading bars over [start, end] alone leaves the
first ~200 days of any test window with degraded features. Fix:
load (start - warmup, end), build features over the extended
window, then filter frames to the original [start, end] before
candidate generation.
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def test_load_bars_with_warmup_extends_start():
    """The helper must request bars from BEFORE the requested
    start to give long-lookback features room to populate."""
    from rl_swing.rl.training.trainer import (
        _FEATURE_WARMUP_DAYS,
        _load_bars_with_warmup,
    )

    captured = {}

    class _MockProvider:
        def get_bars(self, symbols, start_date, end_date, freq, adjusted):
            captured["start_date"] = start_date
            captured["end_date"] = end_date
            return []

    test_start = date(2022, 1, 1)
    test_end = date(2022, 12, 31)
    _bars, warmup_start = _load_bars_with_warmup(
        _MockProvider(), ["AAPL"], test_start, test_end,
    )
    # Provider was asked for bars from BEFORE test_start.
    assert captured["start_date"] < test_start
    assert captured["end_date"] == test_end
    # Warmup is large enough for sma_200 (200 trading days ≈ 280
    # calendar days; we use 1.5 trading years ≈ 378 calendar days).
    days_back = (test_start - warmup_start).days
    assert days_back >= 280, f"warmup of {days_back} calendar days won't cover sma_200"
    assert days_back == _FEATURE_WARMUP_DAYS


def test_filter_frames_to_window_drops_warmup_frames():
    """Frames outside [start, end] must be dropped so candidates
    only fire in the actual eval window."""
    from rl_swing.domain import FeatureFrame
    from rl_swing.rl.training.trainer import _filter_frames_to_window

    in_window = FeatureFrame(
        symbol="AAPL", as_of=datetime(2022, 6, 15),
        feature_version="v1", feature_names=("sma_200",),
        source_snapshot_id="t", values={"sma_200": 150.0},
    )
    pre_window = FeatureFrame(
        symbol="AAPL", as_of=datetime(2021, 6, 15),    # before start
        feature_version="v1", feature_names=("sma_200",),
        source_snapshot_id="t", values={"sma_200": 145.0},
    )
    post_window = FeatureFrame(
        symbol="AAPL", as_of=datetime(2023, 6, 15),    # after end
        feature_version="v1", feature_names=("sma_200",),
        source_snapshot_id="t", values={"sma_200": 155.0},
    )

    out = _filter_frames_to_window(
        [pre_window, in_window, post_window],
        date(2022, 1, 1), date(2022, 12, 31),
    )
    assert len(out) == 1
    assert out[0] is in_window


def test_filter_frames_to_window_inclusive_at_boundaries():
    """Frames AT start and end must be kept (closed interval)."""
    from rl_swing.domain import FeatureFrame
    from rl_swing.rl.training.trainer import _filter_frames_to_window

    at_start = FeatureFrame(
        symbol="AAPL", as_of=datetime(2022, 1, 1),
        feature_version="v1", feature_names=(),
        source_snapshot_id="t", values={},
    )
    at_end = FeatureFrame(
        symbol="AAPL", as_of=datetime(2022, 12, 31),
        feature_version="v1", feature_names=(),
        source_snapshot_id="t", values={},
    )
    out = _filter_frames_to_window(
        [at_start, at_end],
        date(2022, 1, 1), date(2022, 12, 31),
    )
    assert len(out) == 2
