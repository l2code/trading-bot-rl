# Training on Google Colab

The in-sandbox smoke test (10–50k steps, single seed, synthetic data)
runs in a couple of minutes locally. The spec's recommended runs —
500k–2M steps × 3–5 seeds across 5 walk-forward cycles — are too much
for a developer laptop or a sandbox session. Colab is a good fit:
free / Pro GPU, longer-running sessions, good integration with Drive
for artifact persistence.

## Pre-flight (one-time)

1. **Push this repo to GitHub** (public is easiest; private requires a
   PAT in the notebook). The Colab notebook clones from GitHub.
2. **Create a Drive folder** for artifacts, e.g.
   `MyDrive/rl-swing/models/`. The notebook will mount Drive and
   write checkpoints + reports there.

## Run a training experiment

Open the notebook in Colab via:

```
https://colab.research.google.com/github/<USER>/trading-bot-rl/blob/main/notebooks/04_colab_training.ipynb
```

The notebook does the following:

1. Clones the repo into `/content/trading-bot-rl/`.
2. `pip install -e .` so `rl-swing` and the `rl_swing.rl.training`
   modules are importable.
3. Optionally `drive.mount('/content/drive')` to persist artifacts.
4. Calls `rl_swing.rl.training.colab_entrypoint.train(...)` with a
   chosen experiment config and total-timesteps override.

Example invocation cell:

```python
from rl_swing.rl.training.colab_entrypoint import train

summary = train(
    experiment="configs/experiments/ppo_filter_v001.yaml",
    total_timesteps=500_000,
    seeds=[11, 22, 33],
    artifact_root="/content/drive/MyDrive/rl-swing/models",
)
print(summary["runs"][0]["best_validation_score"])
```

After it finishes, the artifacts (`best.zip`, `last.zip`, per-seed
`metadata.json`, the alias `model.zip`, and a per-experiment
`training_summary.json`) live under
`/content/drive/MyDrive/rl-swing/models/<experiment>/`.

## Walk-forward cycles

For multi-cycle walk-forward, run the same `train()` call across
several experiment YAMLs that differ only in their date ranges
(`train_*`, `validation_*`, `test_*`). Each cycle's output goes into
a separate experiment folder.

After training, run validation locally (cheaper than Colab):

```bash
rl-swing validate \
  --experiment configs/experiments/ppo_filter_cycle1.yaml \
  --report-dir data/reports
```

The validator compares the trained model against the configured
baselines (random, always_take_25/50/100, never_take, buy-and-hold)
under both normal and 2× cost stress.

## Notes

- **GPU vs CPU.** Daily-bar swing trading is usually
  *environment-speed-limited*, not GPU-limited (per spec §12.5.12). Use
  a CPU runtime for the first credible run before paying for GPU.
- **Session timeouts.** Colab sessions disconnect after extended idle.
  Save artifacts to Drive after every run. If a 2M-step run can't fit
  in one session, stop at ~500k–1M and resume by training a new model
  (or implement explicit `model.load(); model.learn()` resumption — not
  in scope for this build).
- **Reproducibility.** Every run records seed, code commit (TODO: bake
  into metadata), feature version, and reward config version in
  `metadata.json`. Validation reads these and refuses to score with a
  feature-version mismatch.
- **WRDS data on Colab.** The `WrdsParquetProvider` reads from a
  parquet cache. To use it on Colab, upload your cache to Drive (e.g.
  `MyDrive/wrds-cache/`) and pass `cache_dir="/content/drive/.../wrds-cache"`
  via the registry/component config. Live `wrds` pulls from Colab need
  the `wrds` extra and `~/.pgpass` set up — usually easier to keep
  ingestion local.
