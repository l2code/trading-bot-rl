"""Contract tests: every adapter satisfies its port."""
from __future__ import annotations

from rl_swing.adapters.broker.alpaca_live_broker import AlpacaLiveBrokerAdapter
from rl_swing.adapters.broker.alpaca_paper_broker import AlpacaPaperBrokerAdapter
from rl_swing.adapters.broker.noop_shadow_broker import NoOpShadowBrokerAdapter
from rl_swing.adapters.broker.simulated_broker import SimulatedBrokerAdapter
from rl_swing.adapters.data.parquet_provider import ParquetProvider
from rl_swing.adapters.data.synthetic_provider import SyntheticProvider
from rl_swing.adapters.data.wrds_parquet_provider import WrdsParquetProvider
from rl_swing.adapters.data.yfinance_provider import YFinanceProvider
from rl_swing.features.pipelines import CoreDailyPipeline
from rl_swing.ports import (
    BrokerAdapter,
    CandidateStrategy,
    FeaturePipeline,
    MarketDataProvider,
    PolicyScorer,
)
from rl_swing.rl.agents.baseline_scorers import (
    AlwaysTakePolicyScorer,
    NeverTakePolicyScorer,
    RandomPolicyScorer,
)
from rl_swing.rl.agents.dqn_scorer import DqnPolicyScorer
from rl_swing.rl.agents.ppo_scorer import PpoPolicyScorer
from rl_swing.strategies.aggregator import StrategyAggregator
from rl_swing.strategies.breakout import BreakoutStrategy
from rl_swing.strategies.mean_reversion import RsiMeanReversionStrategy
from rl_swing.strategies.momentum import MomentumStrategy
from rl_swing.strategies.trend_following import TrendFollowingStrategy


def test_market_data_providers_satisfy_port():
    for cls in [SyntheticProvider, ParquetProvider, WrdsParquetProvider, YFinanceProvider]:
        inst = cls()
        assert isinstance(inst, MarketDataProvider)


def test_feature_pipeline_satisfies_port():
    assert isinstance(CoreDailyPipeline(), FeaturePipeline)


def test_strategies_satisfy_port():
    for s in [MomentumStrategy(), RsiMeanReversionStrategy(),
              BreakoutStrategy(), TrendFollowingStrategy()]:
        assert isinstance(s, CandidateStrategy)
    agg = StrategyAggregator([MomentumStrategy()])
    assert isinstance(agg, CandidateStrategy)


def test_policy_scorers_satisfy_port():
    assert isinstance(RandomPolicyScorer(), PolicyScorer)
    assert isinstance(AlwaysTakePolicyScorer(), PolicyScorer)
    assert isinstance(NeverTakePolicyScorer(), PolicyScorer)
    assert isinstance(PpoPolicyScorer(model_id="x", artifact_path="y"), PolicyScorer)
    assert isinstance(DqnPolicyScorer(model_id="x", artifact_path="y"), PolicyScorer)


def test_broker_adapters_satisfy_port():
    assert isinstance(NoOpShadowBrokerAdapter(), BrokerAdapter)
    assert isinstance(SimulatedBrokerAdapter(), BrokerAdapter)
    assert isinstance(AlpacaPaperBrokerAdapter(), BrokerAdapter)
    assert isinstance(AlpacaLiveBrokerAdapter(), BrokerAdapter)


def test_runtime_module_imports_event_bus_and_modes():
    from rl_swing.ports import EventBus  # noqa: F401
    from rl_swing.runtime.event_bus import InMemoryEventBus
    bus = InMemoryEventBus()
    assert isinstance(bus, EventBus)
