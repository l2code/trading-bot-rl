"""Leakage checks.

These run on a list of ``FeatureFrame`` objects against the bars they
were built from and assert no future information made it in.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from rl_swing.domain import FeatureFrame, MarketBar


class LeakageError(AssertionError):
    pass


def assert_no_future_bars(
    frames: Iterable[FeatureFrame],
    bars: Iterable[MarketBar],
) -> None:
    """A FeatureFrame's ``as_of`` may not be before any bar listed as
    its source bar. We can't directly inspect feature provenance, but
    we can assert: there exists at least one bar at or before each
    frame's as_of (i.e. the frame couldn't have been built without
    one), and no bar from after as_of contributed (we can't prove that
    here without provenance, but we *can* assert no frame appears for
    a date that has no bar yet — which is the single most common
    leakage source).
    """
    bars_by_symbol: dict[str, list[datetime]] = {}
    for b in bars:
        bars_by_symbol.setdefault(b.symbol, []).append(b.timestamp)
    for s in bars_by_symbol:
        bars_by_symbol[s].sort()

    for f in frames:
        ts = bars_by_symbol.get(f.symbol)
        if not ts:
            raise LeakageError(
                f"Feature frame for {f.symbol}@{f.as_of} has no source bars."
            )
        if f.as_of < ts[0]:
            raise LeakageError(
                f"Feature frame for {f.symbol}@{f.as_of} predates the earliest bar {ts[0]}."
            )


def assert_features_finite(frames: Iterable[FeatureFrame]) -> None:
    import math
    for f in frames:
        for name, v in f.values.items():
            if v is None or math.isnan(v) or math.isinf(v):
                raise LeakageError(
                    f"non-finite feature {name}={v} for {f.symbol}@{f.as_of}"
                )
