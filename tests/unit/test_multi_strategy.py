"""Unit tests for v2 multi-strategy components."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rl_swing.features.pipelines import ALL_FEATURE_NAMES
from rl_swing.rl.agents.selector_scorers import (
    AlwaysFirstFiredSelectorScorer,
    AlwaysSkipSelectorScorer,
    HighestSignalSelectorScorer,
    RandomSelectorScorer,
)
from rl_swing.rl.env.multi_strategy_env import MultiStrategySwingTradingEnv
from rl_swing.rl.env.multi_strategy_observation import (
    SLOT_DIM,
    MultiStrategyObservationBuilder,
)
from rl_swing.strategies.breakout import BreakoutStrategy
from rl_swing.strategies.mean_reversion import RsiMeanReversionStrategy
from rl_swing.strategies.momentum import MomentumStrategy
from rl_swing.strategies.multi_strategy_packer import (
    MultiStrategyPacker,
    StrategyPack,
)


# ---------------------------------------------------------------------
# Fixture aliases on top of the shared conftest fixtures (which give
# us bars+frames+portfolio for SPY+AAPL+MSFT+NVDA over 2018-2020).
@pytest.fixture
def synthetic_data(synthetic_bars, feature_frames, portfolio_state):
    return synthetic_bars, feature_frames, portfolio_state


# ---------------------------------------------------------------------
def test_packer_groups_by_symbol_date_without_dedupe(synthetic_data):
    """When two strategies fire on the same (symbol, date), the packer
    keeps both — that's the entire point of v2 vs v1."""
    bars, frames, portfolio = synthetic_data
    packer = MultiStrategyPacker([
        MomentumStrategy(min_relative_strength=-0.5, min_r20=-0.5,
                         require_sma200_above=False),
        BreakoutStrategy(min_relative_volume=0.0, max_distance_below_high=-0.5),
        RsiMeanReversionStrategy(rsi_threshold=80.0),
    ])
    packs = packer.pack(frames, portfolio)
    # At least some packs should exist for the synthetic momentum data.
    assert len(packs) > 0
    # Every pack must have at least one strategy fired.
    for p in packs:
        assert p.n_fired >= 1
        assert len(p.candidates) == packer.n_slots
    # Pack ordering: chronological then alphabetical by symbol.
    for a, b in zip(packs, packs[1:], strict=False):
        assert (a.as_of, a.symbol) < (b.as_of, b.symbol)


def test_packer_n_slots_matches_strategy_count(synthetic_data):
    _, frames, portfolio = synthetic_data
    packer = MultiStrategyPacker([
        MomentumStrategy(),
        BreakoutStrategy(),
    ])
    assert packer.n_slots == 2
    assert len(packer.strategy_ids) == 2
    packs = packer.pack(frames, portfolio)
    for p in packs:
        assert len(p.candidates) == 2


def test_packer_empty_strategies_yields_no_packs(synthetic_data):
    _, frames, portfolio = synthetic_data
    packer = MultiStrategyPacker([])
    assert packer.pack(frames, portfolio) == []


# ---------------------------------------------------------------------
def test_observation_builder_dim_is_stable():
    """Observation dim should depend only on (feature_names,
    n_strategies), not on which strategies actually fired."""
    builder = MultiStrategyObservationBuilder(
        feature_names=("a", "b", "c"), n_strategies=3,
    )
    expected_dim = 3 + 3 * SLOT_DIM + 4   # features + slots + portfolio
    assert builder.dim == expected_dim


def test_observation_builder_zero_pads_unfired_slots(synthetic_data):
    """A pack with 1 fired strategy out of 3 should have zeros in 2
    slot positions."""
    _, frames, portfolio = synthetic_data
    packer = MultiStrategyPacker([
        MomentumStrategy(min_relative_strength=-0.5, min_r20=-0.5,
                         require_sma200_above=False),
        BreakoutStrategy(min_relative_volume=0.0, max_distance_below_high=-0.5),
        RsiMeanReversionStrategy(rsi_threshold=80.0),
    ])
    packs = packer.pack(frames, portfolio)
    builder = MultiStrategyObservationBuilder(
        feature_names=ALL_FEATURE_NAMES, n_strategies=3,
    )
    # Find a pack where at least one strategy didn't fire.
    partial = next((p for p in packs if p.n_fired < 3), None)
    if partial is None:
        pytest.skip("synthetic data has all 3 strategies firing on every pack")
    frame = next(f for f in frames
                 if f.symbol == partial.symbol and f.as_of == partial.as_of)
    obs = builder.build(partial, frame, portfolio)
    assert obs.shape == (builder.dim,)
    # Slot region starts after the feature-frame block.
    slot_start = len(ALL_FEATURE_NAMES)
    for i, c in enumerate(partial.candidates):
        slot_base = slot_start + i * SLOT_DIM
        if c is None:
            # All slot fields zero for an unfired strategy.
            assert np.all(obs[slot_base:slot_base + SLOT_DIM] == 0.0)
        else:
            # ``fired`` flag is the first slot field.
            assert obs[slot_base] == 1.0


def test_observation_builder_fired_mask(synthetic_data):
    pack = StrategyPack(
        symbol="AAA", as_of=datetime(2020, 1, 5),
        candidates=(None, None, None),
    )
    builder = MultiStrategyObservationBuilder(
        feature_names=("x",), n_strategies=3,
    )
    mask = builder.fired_mask(pack)
    assert mask.tolist() == [0, 0, 0]


# ---------------------------------------------------------------------
def test_env_action_space_is_discrete_n_plus_one(synthetic_data):
    bars, frames, portfolio = synthetic_data
    packer = MultiStrategyPacker([
        MomentumStrategy(),
        BreakoutStrategy(),
    ])
    packs = packer.pack(frames, portfolio)
    env = MultiStrategySwingTradingEnv(
        bars=bars, packs=packs, feature_frames=frames,
        feature_names=ALL_FEATURE_NAMES, n_strategies=2,
        sampler_kind="chronological",
    )
    assert env.action_space.n == 3  # skip + 2 strategies


def test_env_skip_action_uses_counterfactual_reward(synthetic_data):
    bars, frames, portfolio = synthetic_data
    packer = MultiStrategyPacker([
        MomentumStrategy(min_relative_strength=-0.5, min_r20=-0.5,
                         require_sma200_above=False),
    ])
    packs = packer.pack(frames, portfolio)
    if not packs:
        pytest.skip("no packs in synthetic window")
    env = MultiStrategySwingTradingEnv(
        bars=bars, packs=packs, feature_frames=frames,
        feature_names=ALL_FEATURE_NAMES, n_strategies=1,
        sampler_kind="chronological",
    )
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    # Skip the first pack — info should report 'skip' and have a CF.
    obs2, reward, terminated, truncated, info2 = env.step(0)
    assert info2["action"] == "skip"
    assert "best_cf_return" in info2


def test_env_illegal_action_returns_penalty(synthetic_data):
    bars, frames, portfolio = synthetic_data
    packer = MultiStrategyPacker([
        MomentumStrategy(min_relative_strength=-0.5, min_r20=-0.5,
                         require_sma200_above=False),
        # A strategy that never fires on this synthetic data.
        RsiMeanReversionStrategy(rsi_threshold=-1.0),
    ])
    packs = packer.pack(frames, portfolio)
    if not packs:
        pytest.skip("no packs in synthetic window")
    env = MultiStrategySwingTradingEnv(
        bars=bars, packs=packs, feature_frames=frames,
        feature_names=ALL_FEATURE_NAMES, n_strategies=2,
        sampler_kind="chronological",
        illegal_action_penalty=0.5,
    )
    env.reset(seed=0)
    # Action 2 = take strategy 1 (the never-firing rsi strategy).
    _, reward, _, _, info = env.step(2)
    assert info["action"] == "illegal_strategy_not_fired"
    assert reward == -0.5


# ---------------------------------------------------------------------
def test_env_skip_counterfactual_mode_validation():
    """FIX-26: invalid mode must raise ValueError at construction
    time so a typo in experiment YAML can't silently fall through."""
    with pytest.raises(ValueError, match="skip_counterfactual_mode"):
        MultiStrategySwingTradingEnv(
            bars=[], packs=[], feature_frames=[],
            feature_names=ALL_FEATURE_NAMES, n_strategies=1,
            skip_counterfactual_mode="invalid_mode_typo",
        )


def test_env_skip_counterfactual_mode_default_is_highest_signal(synthetic_data):
    """FIX-26: default mode is highest_signal (no hindsight peek)."""
    bars, frames, portfolio = synthetic_data
    packer = MultiStrategyPacker([
        MomentumStrategy(min_relative_strength=-0.5, min_r20=-0.5,
                         require_sma200_above=False),
    ])
    packs = packer.pack(frames, portfolio)
    if not packs:
        pytest.skip("no packs in synthetic window")
    env = MultiStrategySwingTradingEnv(
        bars=bars, packs=packs, feature_frames=frames,
        feature_names=ALL_FEATURE_NAMES, n_strategies=1,
        sampler_kind="chronological",
    )
    assert env.skip_counterfactual_mode == "highest_signal"


def test_env_skip_counterfactual_mode_max_is_legacy(synthetic_data):
    """FIX-26: max mode is reachable for reproducing pre-FIX-26 runs."""
    bars, frames, portfolio = synthetic_data
    packer = MultiStrategyPacker([
        MomentumStrategy(min_relative_strength=-0.5, min_r20=-0.5,
                         require_sma200_above=False),
    ])
    packs = packer.pack(frames, portfolio)
    if not packs:
        pytest.skip("no packs in synthetic window")
    env = MultiStrategySwingTradingEnv(
        bars=bars, packs=packs, feature_frames=frames,
        feature_names=ALL_FEATURE_NAMES, n_strategies=1,
        sampler_kind="chronological",
        skip_counterfactual_mode="max",
    )
    env.reset(seed=0)
    _, reward, _, _, info = env.step(0)
    assert info["action"] == "skip"
    # max mode CAN access counterfactual; reward is mirrored take.
    assert reward != 0.0 or info.get("best_cf_return") in (0.0, None)


def test_baseline_scorers_respect_fired_mask():
    """The selector baselines must never return an action for a
    strategy that didn't fire in the pack."""
    pack = StrategyPack(
        symbol="X", as_of=datetime(2024, 1, 1),
        candidates=(None, None, None),  # nothing fired
    )
    fake_frame = type("FakeFrame", (), {"feature_version": "v"})()
    fake_portfolio = type("FakePortfolio", (), {})()

    for scorer in [
        RandomSelectorScorer(seed=42),
        AlwaysSkipSelectorScorer(),
        AlwaysFirstFiredSelectorScorer(),
        HighestSignalSelectorScorer(),
    ]:
        a = scorer.select(pack, fake_frame, fake_portfolio)
        assert a == 0, f"{scorer.model_id} returned non-skip on empty pack"


# ---------------------------------------------------------------------
def test_variant_registry_loads_both_variants():
    """The plug-in dispatcher resolves filter_v001 and selector_v002
    by name — adding a new variant is one registry entry."""
    from rl_swing.rl.variants.base import load_variant

    v1 = load_variant("filter_v001")
    v2 = load_variant("selector_v002")
    assert v1.name == "filter_v001"
    assert v2.name == "selector_v002"
    # Both must conform to the TrainingVariant Protocol.
    assert hasattr(v1, "build_env")
    assert hasattr(v1, "evaluate")
    assert hasattr(v2, "build_env")
    assert hasattr(v2, "evaluate")
