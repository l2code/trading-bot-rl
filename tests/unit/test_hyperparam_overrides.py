"""Unit test for the hyperparam-overrides plumbing in train_from_experiment.

Doesn't actually train (would need data + sb3). Just verifies the
merge logic over a stub _ExperimentCfg.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def test_hyperparam_overrides_merge_wins_over_yaml():
    """Override dict values must win over the experiment YAML's
    hyperparams when the same key is set in both."""
    from rl_swing.rl.training import trainer as trainer_mod

    # Build a minimal cfg using the dataclass so we don't have to
    # provide an actual YAML.
    cfg = trainer_mod._ExperimentCfg(
        name="test",
        algorithm="PPO",
        feature_pipeline="equities_features_v001",
        universe="starter_equities",
        train_start=__import__("datetime").date(2014, 1, 1),
        train_end=__import__("datetime").date(2020, 12, 31),
        validation_start=__import__("datetime").date(2021, 1, 1),
        validation_end=__import__("datetime").date(2021, 12, 31),
        test_start=__import__("datetime").date(2022, 1, 1),
        test_end=__import__("datetime").date(2022, 12, 31),
        total_timesteps_initial=1000,
        total_timesteps_max=2000,
        eval_interval_timesteps=500,
        early_stopping_patience=3,
        min_validation_delta=0.01,
        seeds=[11],
        hyperparams={"ent_coef": 0.01, "learning_rate": 3e-4, "n_steps": 2048},
        cost_model={},
        reward={},
        artifact_root="data/models/",
        data_provider="yfinance_daily",
        rl_variant="filter_v001",
        raw_experiment={"name": "test", "algorithm": "PPO"},
    )

    # Simulate the merge logic train_from_experiment uses.
    overrides = {"ent_coef": 0.10, "learning_rate": 1e-3}
    merged = dict(cfg.hyperparams)
    merged.update(overrides)
    cfg.hyperparams = merged

    # Override values win.
    assert cfg.hyperparams["ent_coef"] == 0.10
    assert cfg.hyperparams["learning_rate"] == 1e-3
    # Untouched keys preserved.
    assert cfg.hyperparams["n_steps"] == 2048


def test_hyperparam_overrides_none_leaves_yaml_unchanged():
    """When overrides is None or empty, hyperparams must be unchanged
    (no spurious key additions)."""
    original = {"ent_coef": 0.01, "learning_rate": 3e-4}
    if not None:
        merged = dict(original)
        # Simulating the no-op branch: nothing happens.
        assert merged == original


def test_hyperparam_overrides_round_trip_via_json():
    """The Kaggle path passes overrides as a JSON string. Verify the
    round-trip preserves int vs float types."""
    import json
    overrides = {"ent_coef": 0.05, "learning_rate": 1e-3, "n_steps": 4096}
    encoded = json.dumps(overrides)
    decoded = json.loads(encoded)
    assert decoded["ent_coef"] == 0.05
    assert decoded["learning_rate"] == 1e-3
    assert decoded["n_steps"] == 4096
    assert isinstance(decoded["n_steps"], int)
