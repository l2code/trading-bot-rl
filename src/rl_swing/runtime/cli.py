"""``rl-swing`` CLI entry point.

Subcommands:
    rl-swing run daily --mode shadow --config configs/runtime/shadow.yaml
    rl-swing train     --experiment configs/experiments/ppo_filter_smoke.yaml
    rl-swing validate  --model-id ppo_filter_v001 --experiment <cfg>
    rl-swing reconcile --mode paper
    rl-swing list-components

The CLI is deliberately thin — it parses args and calls services.
All real logic lives in ``rl_swing.services.*``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import click

DEFAULT_COMPONENTS = "configs/components/components.yaml"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, default=False)
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """RL swing-trading bot CLI."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


@cli.command()
@click.argument("frequency", type=click.Choice(["daily"]))
@click.option("--mode", type=click.Choice(["research", "shadow", "paper", "live_guarded"]),
              required=True)
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False),
              required=True)
@click.option("--components", "components_path",
              type=click.Path(exists=True, dir_okay=False),
              default=DEFAULT_COMPONENTS, show_default=True)
def run(frequency: str, mode: str, config_path: str, components_path: str) -> None:
    """Run the daily decision pipeline in the given mode."""
    from rl_swing.runtime.dependency_container import build_container
    from rl_swing.services.pipeline import DecisionPipeline

    container = build_container(config_path, components_path)
    if container.config.mode != mode:
        click.echo(
            f"Warning: --mode={mode!r} does not match config mode "
            f"{container.config.mode!r}. Using config mode.",
            err=True,
        )
    pipeline = DecisionPipeline(container)
    pipeline.run_once()


@cli.command()
@click.option("--experiment", "experiment_path",
              type=click.Path(exists=True, dir_okay=False),
              required=True)
@click.option("--total-timesteps", type=int, default=None,
              help="Override the experiment's total_timesteps_initial.")
@click.option("--seed", type=int, default=None,
              help="Override the experiment's seed list (single seed).")
@click.option("--data-provider", type=str, default=None,
              help="Override the data provider (e.g. synthetic_momentum).")
@click.option("--n-envs", type=int, default=1, show_default=True,
              help="Parallel envs (SubprocVecEnv when >1). 4 is a good default on a 4-core CPU.")
def train(
    experiment_path: str,
    total_timesteps: int | None,
    seed: int | None,
    data_provider: str | None,
    n_envs: int,
) -> None:
    """Train a PolicyScorer per the given experiment config."""
    from rl_swing.rl.training.trainer import train_from_experiment

    train_from_experiment(
        experiment_path=experiment_path,
        total_timesteps_override=total_timesteps,
        seed_override=seed,
        data_provider_override=data_provider,
        n_envs=n_envs,
    )


@cli.command()
@click.option("--experiment", "experiment_path",
              type=click.Path(exists=True, dir_okay=False),
              required=True)
@click.option("--model-id", type=str, default=None,
              help="Specific trained model id (defaults to experiment.name).")
@click.option("--report-dir", type=click.Path(),
              default="data/reports", show_default=True)
@click.option("--test-start", "test_start", type=str, default=None,
              help="Override the experiment YAML's test_start (YYYY-MM-DD). "
                   "Useful for multi-cycle walk-forward without per-year YAMLs.")
@click.option("--test-end", "test_end", type=str, default=None,
              help="Override the experiment YAML's test_end (YYYY-MM-DD).")
@click.option("--data-provider", "data_provider", type=str, default=None,
              help="Override the experiment YAML's data_provider "
                   "(e.g. yfinance_daily). YAML default for v002 / v002_masked "
                   "is silently 'synthetic_momentum'; pass yfinance_daily for "
                   "real data evaluations.")
def validate(
    experiment_path: str,
    model_id: str | None,
    report_dir: str,
    test_start: str | None,
    test_end: str | None,
    data_provider: str | None,
) -> None:
    """Run walk-forward validation + baseline comparison."""
    from datetime import date

    from rl_swing.rl.validation.walk_forward import validate_from_experiment

    validate_from_experiment(
        experiment_path=experiment_path,
        model_id=model_id,
        report_dir=Path(report_dir),
        test_start_override=date.fromisoformat(test_start) if test_start else None,
        test_end_override=date.fromisoformat(test_end) if test_end else None,
        data_provider_override=data_provider,
    )


@cli.command()
@click.option("--mode", type=click.Choice(["research", "shadow", "paper", "live_guarded"]),
              required=True)
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False),
              required=True)
def reconcile(mode: str, config_path: str) -> None:
    """Run the reconciliation cycle for the given mode (stub for paper/live)."""
    click.echo(f"reconcile in mode={mode} from config={config_path}")
    click.echo("(Reconciliation service is a stub in this build — Phase 8.)")


@cli.command("list-components")
@click.option("--components", "components_path",
              type=click.Path(exists=True, dir_okay=False),
              default=DEFAULT_COMPONENTS, show_default=True)
def list_components(components_path: str) -> None:
    """Print the registered component names by category."""
    from rl_swing.runtime.registry import ComponentRegistry
    reg = ComponentRegistry.from_yaml(components_path)
    for category in reg.categories():
        click.echo(f"\n[{category}]")
        for name in reg.names(category):
            spec = reg.get_spec(category, name)
            click.echo(f"  {name:30s} -> {spec.cls_path}")


def main() -> None:
    cli(prog_name="rl-swing")


if __name__ == "__main__":
    main()
