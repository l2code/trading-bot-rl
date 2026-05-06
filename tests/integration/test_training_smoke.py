"""End-to-end PPO training smoke. Runs against the in-tree synthetic
provider so the test is deterministic and offline."""
from __future__ import annotations

from pathlib import Path

import yaml


def test_training_smoke_runs_and_writes_artifacts(tmp_path: Path):
    from rl_swing.rl.training.colab_entrypoint import train

    exp = tmp_path / "exp.yaml"
    exp.write_text(yaml.safe_dump({
        "experiment": {
            "name": "test_smoke",
            "algorithm": "PPO",
            "policy": "MlpPolicy",
            "feature_pipeline": "equities_features_v001",
            "universe": "synthetic",
            "reward_config_version": "r",
            "train_start": "2018-01-01",
            "train_end": "2019-06-30",
            "validation_start": "2019-07-01",
            "validation_end": "2019-12-31",
            "test_start": "2020-01-01",
            "test_end": "2020-06-30",
            "total_timesteps_initial": 1024,
            "total_timesteps_max": 1024,
            "eval_interval_timesteps": 512,
            "early_stopping_patience_evaluations": 1,
            "min_validation_delta": 0.0,
            "save_best_only": True,
            "seeds": [11],
            "hyperparams": {
                "learning_rate": 3.0e-4,
                "n_steps": 128, "batch_size": 32, "n_epochs": 2,
                "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2,
                "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5,
                "policy_kwargs": {"net_arch": [32]},
            },
            "cost_model": {},
            "reward": {},
            "data_provider": "synthetic_momentum",
            "artifact_root": str(tmp_path / "models"),
        }
    }))

    summary = train(experiment=str(exp), total_timesteps=512)
    assert summary["experiment"] == "test_smoke"
    assert (tmp_path / "models" / "test_smoke" / "model.zip").exists()


def test_training_via_iter_seeds(tmp_path: Path):
    from rl_swing.rl.training.colab_entrypoint import train

    exp = tmp_path / "exp.yaml"
    exp.write_text(yaml.safe_dump({
        "experiment": {
            "name": "test_iter_seeds",
            "algorithm": "PPO",
            "policy": "MlpPolicy",
            "feature_pipeline": "equities_features_v001",
            "universe": "synthetic",
            "reward_config_version": "r",
            "train_start": "2018-01-01",
            "train_end": "2018-12-31",
            "validation_start": "2019-01-01",
            "validation_end": "2019-06-30",
            "test_start": "2019-07-01",
            "test_end": "2019-12-31",
            "total_timesteps_initial": 256,
            "total_timesteps_max": 256,
            "eval_interval_timesteps": 256,
            "early_stopping_patience_evaluations": 1,
            "min_validation_delta": 0.0,
            "save_best_only": True,
            "seeds": [1],
            "hyperparams": {
                "learning_rate": 3.0e-4,
                "n_steps": 64, "batch_size": 32, "n_epochs": 2,
                "policy_kwargs": {"net_arch": [16]},
            },
            "cost_model": {},
            "reward": {},
            "data_provider": "synthetic_momentum",
            "artifact_root": str(tmp_path / "models"),
        }
    }))
    out = train(experiment=str(exp), seeds=[1, 2], total_timesteps=128)
    assert "runs" in out
    assert len(out["runs"]) == 2
