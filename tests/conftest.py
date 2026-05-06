"""Shared fixtures.

* ``synthetic_bars``: deterministic momentum-regime bars for a small
  universe (SPY + 3 names), three years.
* ``feature_frames``: the feature pipeline output for those bars.
* ``portfolio_state``: an empty PortfolioState anchored at the last day.
* ``candidates``: union of the four MVP strategies' candidates.
* ``tmp_db``: a fresh sqlite bundle on a temp path.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from rl_swing.adapters.data.synthetic_provider import SyntheticProvider
from rl_swing.domain import (
    CandidateTrade,
    FeatureFrame,
    MarketBar,
    PortfolioState,
)
from rl_swing.features.pipelines import CoreDailyPipeline
from rl_swing.strategies.aggregator import StrategyAggregator
from rl_swing.strategies.breakout import BreakoutStrategy
from rl_swing.strategies.mean_reversion import RsiMeanReversionStrategy
from rl_swing.strategies.momentum import MomentumStrategy
from rl_swing.strategies.trend_following import TrendFollowingStrategy

UNIVERSE_SMALL = ["SPY", "AAPL", "MSFT", "NVDA"]


@pytest.fixture(scope="session")
def synthetic_bars_session() -> list[MarketBar]:
    prov = SyntheticProvider(regime="momentum", seed=11)
    return list(prov.get_bars(UNIVERSE_SMALL, date(2018, 1, 1), date(2020, 12, 31)))


@pytest.fixture
def synthetic_bars(synthetic_bars_session) -> list[MarketBar]:
    # Re-list to make the fixture independent of session-state mutation.
    return list(synthetic_bars_session)


@pytest.fixture(scope="session")
def feature_frames_session(synthetic_bars_session) -> list[FeatureFrame]:
    pipe = CoreDailyPipeline()
    return list(pipe.build(synthetic_bars_session))


@pytest.fixture
def feature_frames(feature_frames_session) -> list[FeatureFrame]:
    return list(feature_frames_session)


@pytest.fixture
def portfolio_state() -> PortfolioState:
    return PortfolioState(
        as_of=datetime(2020, 12, 31), cash=100_000.0, equity=100_000.0,
    )


@pytest.fixture
def candidates(feature_frames, portfolio_state) -> list[CandidateTrade]:
    agg = StrategyAggregator([
        MomentumStrategy(),
        RsiMeanReversionStrategy(),
        BreakoutStrategy(),
        TrendFollowingStrategy(),
    ])
    return list(agg.generate(feature_frames, portfolio_state))


@pytest.fixture
def tmp_db(tmp_path: Path):
    from rl_swing.adapters.storage.sqlite_repositories import SqliteStorageBundle
    db_path = tmp_path / "test.sqlite"
    return SqliteStorageBundle(database_url=f"sqlite:///{db_path}")
