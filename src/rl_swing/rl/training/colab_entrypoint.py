"""Colab entry point.

The Colab notebook calls ``train()`` so the same training code path
runs locally and remotely. The function takes simple kwargs (no
argparse) so it's friendly inside a notebook.

Typical Colab use:

    !git clone https://github.com/<USER>/trading-bot-rl.git
    %cd trading-bot-rl
    !pip install -e .

    from google.colab import drive
    drive.mount('/content/drive')

    from rl_swing.rl.training.colab_entrypoint import train
    train(
        experiment="configs/experiments/ppo_filter_v001.yaml",
        total_timesteps=500_000,
        seed=11,
        artifact_root="/content/drive/MyDrive/rl-swing/models",
    )
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def train(
    experiment: str | Path,
    *,
    total_timesteps: int | None = None,
    seed: int | None = None,
    seeds: Iterable[int] | None = None,
    data_provider: str | None = None,
    artifact_root: str | None = None,
    n_envs: int = 1,
) -> dict:
    """Run training from a notebook. Returns the same summary dict the
    CLI ``rl-swing train`` writes to disk.

    ``n_envs > 1`` uses ``SubprocVecEnv`` to run that many parallel
    environment workers. Roughly linear speedup up to your CPU core
    count for our daily-bar workload.
    """
    from rl_swing.rl.training.trainer import train_from_experiment

    if seeds is not None and seed is None:
        # Loop over the requested seeds; aggregate.
        summaries = []
        for s in seeds:
            summaries.append(
                train_from_experiment(
                    experiment_path=str(experiment),
                    total_timesteps_override=total_timesteps,
                    seed_override=int(s),
                    data_provider_override=data_provider,
                    artifact_root_override=artifact_root,
                    n_envs=n_envs,
                )
            )
        return {"runs": summaries}
    return train_from_experiment(
        experiment_path=str(experiment),
        total_timesteps_override=total_timesteps,
        seed_override=seed,
        data_provider_override=data_provider,
        artifact_root_override=artifact_root,
        n_envs=n_envs,
    )
