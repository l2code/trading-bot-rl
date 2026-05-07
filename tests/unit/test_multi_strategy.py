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


# ---------------------------------------------------------------------
# FEAT-29: action mask unit tests for the MaskablePPO selector lane.
#
# The whole point of #29 is that the env tells MaskablePPO which slots
# are legal so the policy literally cannot pick a non-fired strategy.
# These tests pin the contract:
#   - skip (index 0) is ALWAYS True.
#   - strategy slot k is True iff candidates[k] is not None.
#   - operator-requested edge case: a pack with exactly one fired
#     strategy yields a mask permitting only skip + that one slot,
#     and nothing else.
def _make_candidate(strategy_id: str, symbol: str = "AAA",
                    signal_strength: float = 0.5):
    """Minimal CandidateTrade for mask tests — values don't matter
    since the mask only cares about None vs not-None."""
    from rl_swing.domain.candidates import CandidateTrade
    return CandidateTrade(
        candidate_id=f"{strategy_id}-{symbol}-test",
        as_of=datetime(2024, 1, 1),
        symbol=symbol,
        strategy_id=strategy_id,
        direction="long",
        entry_timing="next_open",
        base_size_pct=0.10,
        max_holding_days=10,
        stop_rule_id=None,
        exit_rule_id="time_or_atr",
        signal_strength=signal_strength,
        metadata={},
    )


def _env_with_pack(pack: StrategyPack, n_strategies: int) -> MultiStrategySwingTradingEnv:
    """Build a minimal env with a single hand-built pack so mask
    tests don't depend on the synthetic data fixture or feature
    frames being present at the pack's date.

    The env's ``reset`` requires a feature frame for the first pack;
    we register a stub frame so reset succeeds before we read the
    mask. Bars / cost / reward defaults are fine — these tests don't
    step the env, they only inspect ``action_masks()``.
    """
    from rl_swing.domain import FeatureFrame

    stub_values = {name: 0.0 for name in ALL_FEATURE_NAMES}
    frame = FeatureFrame(
        symbol=pack.symbol, as_of=pack.as_of,
        feature_version="features_v001_core_daily",
        values=stub_values,
        feature_names=ALL_FEATURE_NAMES,
        source_snapshot_id="test-fixture",
    )
    env = MultiStrategySwingTradingEnv(
        bars=[], packs=[pack], feature_frames=[frame],
        feature_names=ALL_FEATURE_NAMES, n_strategies=n_strategies,
        sampler_kind="chronological",
    )
    env.reset(seed=0)
    return env


def test_action_mask_skip_is_always_true_when_nothing_fired():
    """Edge case: 0-fired pack should never reach the env in
    practice (the packer drops them), but if it does, the mask must
    still permit skip — the policy needs *some* legal action."""
    pack = StrategyPack(
        symbol="AAA", as_of=datetime(2024, 1, 5),
        candidates=(None, None, None),
    )
    env = _env_with_pack(pack, n_strategies=3)
    mask = env.action_masks()
    assert mask.dtype == bool
    assert mask.shape == (4,)  # skip + 3 strategy slots
    assert mask[0] is np.True_ or bool(mask[0]) is True
    assert not mask[1] and not mask[2] and not mask[3]


def test_action_mask_single_fired_strategy_permits_only_skip_plus_that_slot():
    """OPERATOR-REQUESTED (FEAT-29 scope, 2026-05-06): pack with
    exactly one fired strategy must produce a mask permitting only
    skip plus that one slot, and nothing else.

    This is the load-bearing invariant of the entire feature: the
    mask is what stops MaskablePPO from picking non-fired slots.
    Regressing this test means the masking is structurally broken
    and MaskablePPO would fall back to penalty-shaping, defeating
    the whole point of #29.
    """
    fired_only_in_slot_1 = (
        None,
        _make_candidate("rsi_mean_reversion"),
        None,
    )
    pack = StrategyPack(
        symbol="AAA", as_of=datetime(2024, 1, 5),
        candidates=fired_only_in_slot_1,
    )
    env = _env_with_pack(pack, n_strategies=3)
    mask = env.action_masks()

    assert mask.shape == (4,)
    # Skip (0) and the fired slot (1+1=2) must be True; everything
    # else must be False.
    assert bool(mask[0]) is True, "skip must always be legal"
    assert bool(mask[1]) is False, "slot 0 didn't fire — must be masked"
    assert bool(mask[2]) is True, "slot 1 fired — must be unmasked"
    assert bool(mask[3]) is False, "slot 2 didn't fire — must be masked"
    # Total legal actions = exactly 2.
    assert int(mask.sum()) == 2


def test_action_mask_all_fired_permits_every_action():
    pack = StrategyPack(
        symbol="AAA", as_of=datetime(2024, 1, 5),
        candidates=(
            _make_candidate("momentum"),
            _make_candidate("rsi_mean_reversion"),
            _make_candidate("breakout"),
        ),
    )
    env = _env_with_pack(pack, n_strategies=3)
    mask = env.action_masks()
    assert mask.shape == (4,)
    assert mask.tolist() == [True, True, True, True]


def test_maskable_scorer_static_mask_helper_matches_env_mask():
    """The inference scorer rebuilds the mask from a pack at predict
    time (it never has direct access to the training env). Pin that
    its helper is bit-identical to the env's ``action_masks()`` so
    train-time and eval-time use the SAME mask logic."""
    from rl_swing.rl.agents.selector_scorers import MaskablePpoSelectorScorer

    pack = StrategyPack(
        symbol="AAA", as_of=datetime(2024, 1, 5),
        candidates=(None, _make_candidate("rsi_mean_reversion"), None),
    )
    env = _env_with_pack(pack, n_strategies=3)
    env_mask = env.action_masks()
    scorer_mask = MaskablePpoSelectorScorer._action_mask_for_pack(pack, n_strategies=3)
    assert env_mask.tolist() == scorer_mask.tolist()


def test_variant_registry_loads_masked_variant():
    """selector_v002_masked is registered alongside the unmasked
    variants and conforms to the TrainingVariant Protocol."""
    from rl_swing.rl.variants.base import load_variant

    v2m = load_variant("selector_v002_masked")
    assert v2m.name == "selector_v002_masked"
    assert hasattr(v2m, "build_env")
    assert hasattr(v2m, "evaluate")


# ---------------------------------------------------------------------
# FEAT-30: supervised ranker baseline (sklearn HistGB).
def test_supervised_ranker_returns_skip_for_empty_pack(tmp_path):
    """Empty pack (nothing fired) -> the ranker must return 0 (skip)
    without trying to predict anything. Exercises the early-return
    before model load, so it doesn't even need a real artifact."""
    from rl_swing.rl.agents.supervised_ranker_scorer import (
        SupervisedRankerSelectorScorer,
    )

    pack = StrategyPack(
        symbol="X", as_of=datetime(2024, 1, 1),
        candidates=(None, None, None),
    )
    fake_frame = type("FakeFrame", (), {"feature_version": "features_v001_core_daily"})()
    fake_portfolio = type("FakePortfolio", (), {})()
    scorer = SupervisedRankerSelectorScorer(
        artifact_path=str(tmp_path / "missing.joblib"),
        n_strategies=3,
    )
    assert scorer.select(pack, fake_frame, fake_portfolio) == 0


def test_supervised_ranker_picks_argmax_above_threshold(tmp_path, synthetic_data):
    """Train a tiny ranker on a few hand-built rows where slot 1 is
    obviously the best, then verify the scorer picks slot 1 (action=2)
    on a pack where slot 1 is fired and predicted positive."""
    import joblib
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor

    from rl_swing.rl.agents.supervised_ranker_scorer import (
        PER_SLOT_FEATURE_NAMES,
        SupervisedRankerSelectorScorer,
    )

    # Tiny synthetic dataset: slot_idx is the dominant feature; slot 1 maps to high return.
    # Per-slot feature values match what _make_candidate emits at inference
    # (signal_strength=0.5, base_size_pct=0.10, max_holding_days=10) so the
    # model isn't asked to extrapolate at predict time.
    n_features = len(PER_SLOT_FEATURE_NAMES)
    X = np.zeros((90, n_features), dtype=np.float64)
    y = np.zeros(90, dtype=np.float64)
    cols = {n: i for i, n in enumerate(PER_SLOT_FEATURE_NAMES)}
    for i in range(90):
        slot = i % 3
        X[i, cols["slot_idx"]] = float(slot)
        X[i, cols["slot_signal_strength"]] = 0.5
        X[i, cols["slot_base_size_pct"]] = 0.10
        X[i, cols["slot_max_holding_days_norm"]] = 10.0 / 30.0
        y[i] = {0: -0.5, 1: 1.5, 2: 0.0}[slot]
    model = HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=11)
    model.fit(X, y)

    art = tmp_path / "ranker.joblib"
    joblib.dump({
        "model": model,
        "feature_names": PER_SLOT_FEATURE_NAMES,
        "n_strategies": 3,
        "target_risk_pct": 0.02,
    }, str(art))

    # Build a pack where ALL three strategies fired so the ranker
    # has a real choice to make.
    pack = StrategyPack(
        symbol="AAA", as_of=datetime(2024, 1, 5),
        candidates=(
            _make_candidate("momentum"),
            _make_candidate("rsi_mean_reversion"),
            _make_candidate("breakout"),
        ),
    )
    # Need a real FeatureFrame for build_slot_features.
    from rl_swing.domain import FeatureFrame
    stub = FeatureFrame(
        symbol=pack.symbol, as_of=pack.as_of,
        feature_version="features_v001_core_daily",
        values={n: 0.0 for n in ALL_FEATURE_NAMES},
        feature_names=ALL_FEATURE_NAMES,
        source_snapshot_id="test",
    )
    fake_portfolio = type("FakePortfolio", (), {})()

    scorer = SupervisedRankerSelectorScorer(
        artifact_path=str(art), n_strategies=3,
    )
    action = scorer.select(pack, stub, fake_portfolio)
    # action == 0 means skip; action == k means take slot k-1.
    # Slot 1 has predicted return ~+1.5 (above 0 threshold) and is the max.
    assert action == 2, f"expected action=2 (slot 1, the dominant slot); got {action}"


def test_supervised_ranker_skips_when_max_below_threshold(tmp_path):
    """If every fired slot has predicted return < 0 (skip threshold),
    the ranker must return 0. Trains a model where every training row
    has a negative target, so any prediction will also be negative."""
    import joblib
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor

    from rl_swing.rl.agents.supervised_ranker_scorer import (
        PER_SLOT_FEATURE_NAMES,
        SupervisedRankerSelectorScorer,
    )

    n_features = len(PER_SLOT_FEATURE_NAMES)
    X = np.zeros((30, n_features), dtype=np.float64)
    y = np.full(30, -1.0, dtype=np.float64)  # every example is negative-EV
    model = HistGradientBoostingRegressor(max_iter=20, max_depth=3, random_state=11)
    model.fit(X, y)

    art = tmp_path / "ranker.joblib"
    joblib.dump({
        "model": model,
        "feature_names": PER_SLOT_FEATURE_NAMES,
        "n_strategies": 3,
        "target_risk_pct": 0.02,
    }, str(art))

    pack = StrategyPack(
        symbol="AAA", as_of=datetime(2024, 1, 5),
        candidates=(
            _make_candidate("momentum"),
            _make_candidate("rsi_mean_reversion"),
            None,
        ),
    )
    from rl_swing.domain import FeatureFrame
    stub = FeatureFrame(
        symbol=pack.symbol, as_of=pack.as_of,
        feature_version="features_v001_core_daily",
        values={n: 0.0 for n in ALL_FEATURE_NAMES},
        feature_names=ALL_FEATURE_NAMES,
        source_snapshot_id="test",
    )
    fake_portfolio = type("FakePortfolio", (), {})()

    scorer = SupervisedRankerSelectorScorer(
        artifact_path=str(art), n_strategies=3,
    )
    assert scorer.select(pack, stub, fake_portfolio) == 0  # skip
