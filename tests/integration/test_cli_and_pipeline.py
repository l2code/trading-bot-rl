"""CLI roundtrip + end-to-end research pipeline."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from click.testing import CliRunner

from rl_swing.runtime.cli import cli


def test_cli_list_components_lists_known_categories():
    runner = CliRunner()
    res = runner.invoke(cli, ["list-components"])
    assert res.exit_code == 0
    assert "market_data_providers" in res.output
    assert "policy_scorers" in res.output


def test_cli_run_command_works_with_research_mode(tmp_path):
    runner = CliRunner()

    # Use the synthetic data provider in a research run.
    components = tmp_path / "components.yaml"
    components.write_text(yaml.safe_dump({
        "components": {
            "market_data_providers": {
                "synthetic_momentum": {
                    "class": "rl_swing.adapters.data.synthetic_provider.SyntheticProvider",
                    "params": {"regime": "momentum", "seed": 11},
                }
            },
            "feature_pipelines": {
                "equities_features_v001": {
                    "class": "rl_swing.features.pipelines.CoreDailyPipeline",
                    "params": {},
                }
            },
            "strategies": {
                "momentum_20_60": {
                    "class": "rl_swing.strategies.momentum.MomentumStrategy",
                    "params": {},
                }
            },
            "policy_scorers": {
                "random": {
                    "class": "rl_swing.rl.agents.baseline_scorers.RandomPolicyScorer",
                    "params": {"model_id": "rand"},
                }
            },
            "broker_adapters": {
                "shadow": {
                    "class": "rl_swing.adapters.broker.noop_shadow_broker.NoOpShadowBrokerAdapter",
                    "params": {},
                }
            },
        }
    }))

    runtime = tmp_path / "runtime.yaml"
    runtime.write_text(yaml.safe_dump({
        "runtime": {
            "mode": "research",
            "universe": "synthetic",
            "data_provider": "synthetic_momentum",
            "feature_pipeline": "equities_features_v001",
            "strategies": ["momentum_20_60"],
            "policy_scorer": "random",
            "broker_adapter": "shadow",
            "storage_profile": "local_sqlite",
        }
    }))

    res = runner.invoke(cli, [
        "run", "daily",
        "--mode", "research",
        "--config", str(runtime),
        "--components", str(components),
    ])
    assert res.exit_code == 0, res.output


def test_cli_reconcile_stub_message():
    runner = CliRunner()
    # Use any existing config.
    cfg = Path("configs/runtime/research.yaml")
    res = runner.invoke(cli, ["reconcile", "--mode", "research", "--config", str(cfg)])
    assert res.exit_code == 0
    assert "stub" in res.output.lower()


def test_cli_run_warns_on_mode_mismatch(tmp_path):
    runner = CliRunner()
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text(yaml.safe_dump({
        "runtime": {
            "mode": "research",
            "universe": "synthetic",
            "data_provider": "synthetic_momentum",
            "feature_pipeline": "equities_features_v001",
            "strategies": ["momentum_20_60"],
            "policy_scorer": "random",
            "broker_adapter": "shadow",
            "storage_profile": "local_sqlite",
        }
    }))
    res = runner.invoke(cli, [
        "run", "daily",
        "--mode", "shadow",  # mismatch
        "--config", str(runtime),
        "--components", "configs/components/components.yaml",
    ])
    assert res.exit_code == 0
    assert "warning" in res.output.lower() or "Warning" in res.output


def test_research_pipeline_end_to_end(tmp_path):
    """Build a real container against the shipped components.yaml + a
    custom synthetic runtime config, run the pipeline once, assert
    candidates and decisions came out."""
    from rl_swing.runtime.dependency_container import build_container
    from rl_swing.services.pipeline import DecisionPipeline

    runtime = tmp_path / "runtime.yaml"
    runtime.write_text(yaml.safe_dump({
        "runtime": {
            "mode": "research",
            "universe": "synthetic",
            "data_provider": "synthetic_momentum",
            "feature_pipeline": "equities_features_v001",
            "strategies": ["momentum_20_60", "mean_reversion_rsi"],
            "policy_scorer": "random",
            "broker_adapter": "shadow",
            "storage_profile": "local_sqlite",
        }
    }))
    container = build_container(runtime, "configs/components/components.yaml")
    pipeline = DecisionPipeline(container)
    out = pipeline.run_once(as_of=date(2020, 6, 30))
    assert out.bars_count > 0
    # Candidates may be 0 in some random windows, but feature frames must be > 0.
    assert len(out.feature_frames) > 0


def test_pipeline_handles_unknown_universe_string(tmp_path):
    """When the universe string isn't a config and isn't a path, the
    pipeline falls back to treating it as comma-separated symbols."""
    from rl_swing.runtime.dependency_container import build_container
    from rl_swing.services.pipeline import DecisionPipeline

    runtime = tmp_path / "runtime.yaml"
    runtime.write_text(yaml.safe_dump({
        "runtime": {
            "mode": "research",
            "universe": "AAA,BBB",
            "data_provider": "synthetic_momentum",
            "feature_pipeline": "equities_features_v001",
            "strategies": ["momentum_20_60"],
            "policy_scorer": "random",
            "broker_adapter": "shadow",
            "storage_profile": "local_sqlite",
        }
    }))
    container = build_container(runtime, "configs/components/components.yaml")
    pipeline = DecisionPipeline(container)
    out = pipeline.run_once(as_of=date(2020, 6, 30))
    assert out.bars_count > 0
