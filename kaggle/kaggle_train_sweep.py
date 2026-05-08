"""rl-swing hyperparam-sweep training on Kaggle Notebooks (script kernel).

Sister script to ``kaggle_train.py``. Runs a grid sweep over a list of
``hyperparam_overrides`` cells (e.g. (ent_coef, learning_rate) pairs)
in a single kernel session. Per cell:

  1. Train ``train()`` with the cell's hyperparam_overrides over the
     full seed list, into a per-cell artifact_root.
  2. Validate that cell's best artifact via
     ``validate_from_experiment``.
  3. Append the cell's train + validation summaries to a single
     ``sweep_summary.json`` so all cells land in one structured output.

Used by FEAT-32 M3.b (#92) for the v003 masked-PPO Optuna grid sweep,
but generic enough that v002 #27 RESEARCH-8 can reuse it.

Configuration via env vars (mirrors ``kaggle_train.py`` plus a sweep-
specific knob):

    RL_SWING_EXPERIMENT          — experiment YAML, e.g.
                                  ``configs/experiments/ppo_portfolio_v003.yaml``
    RL_SWING_TOTAL_TIMESTEPS     — per-seed total timesteps (override).
    RL_SWING_SEEDS               — comma-separated seeds, e.g. "11,22,33".
    RL_SWING_DATA_PROVIDER       — explicit data provider (FIX-#78 guardrail
                                   wants this set on selector / portfolio variants).
    RL_SWING_N_ENVS              — vec-env workers per seed.
    RL_SWING_REPO_URL            — git URL.
    RL_SWING_REPO_BRANCH         — git branch.
    RL_SWING_SWEEP_CELLS         — JSON list of dicts, each a hyperparam
                                   override cell. Example:
                                   ``[{"ent_coef":0.01,"learning_rate":3e-4},
                                      {"ent_coef":0.05,"learning_rate":3e-4}]``

Outputs (in ``/kaggle/working``):
  - ``sweep_summary.json``  — aggregate of all cells' summaries.
    Written incrementally after every cell (atomic tmp+rename) so a
    Kaggle worker restart mid-sweep can't lose more than the
    in-flight cell. Carries ``complete: false`` until the final write
    flips it to ``true`` after the last cell.
  - ``progress.json``       — heartbeat. ``{cells_done, cells_total,
    current, elapsed_s, status}``. Updated before each cell starts +
    after each cell completes. The host can pull this with
    ``kaggle kernels output --force`` once Kaggle has flushed it.
  - ``artifacts/cell_<NN>/...``  — per-cell training artifacts.
  - ``cell_<NN>/validation_summary.json`` — per-cell validate output.

The structure is deliberately a thin extension of ``kaggle_train.py``:
clone the repo, install missing deps, then run the per-cell loop.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------
# 0) Config from env, with sensible defaults.
# ----------------------------------------------------------------------
EXPERIMENT    = os.environ.get(
    "RL_SWING_EXPERIMENT",
    "configs/experiments/ppo_portfolio_v003.yaml",
)
TOTAL_STEPS   = os.environ.get("RL_SWING_TOTAL_TIMESTEPS")
SEEDS_RAW     = os.environ.get("RL_SWING_SEEDS")
DATA_PROVIDER = os.environ.get("RL_SWING_DATA_PROVIDER")
N_ENVS        = int(os.environ.get("RL_SWING_N_ENVS", "1") or "1")
SWEEP_CELLS_RAW = os.environ.get("RL_SWING_SWEEP_CELLS")
REPO_URL      = os.environ.get(
    "RL_SWING_REPO_URL", "https://github.com/l2code/trading-bot-rl.git"
)
REPO_BRANCH   = os.environ.get("RL_SWING_REPO_BRANCH", "main")
WORKING       = Path("/kaggle/working")
REPO_DIR      = WORKING / "trading-bot-rl"
ARTIFACTS     = WORKING / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

if not SWEEP_CELLS_RAW:
    raise SystemExit("[kaggle_train_sweep] RL_SWING_SWEEP_CELLS is required (JSON list).")
SWEEP_CELLS = json.loads(SWEEP_CELLS_RAW)
if not isinstance(SWEEP_CELLS, list) or not SWEEP_CELLS:
    raise SystemExit("[kaggle_train_sweep] RL_SWING_SWEEP_CELLS must be a non-empty JSON list.")

print(f"[kaggle_train_sweep] experiment={EXPERIMENT!r}")
print(f"[kaggle_train_sweep] total_timesteps={TOTAL_STEPS!r}  seeds={SEEDS_RAW!r}  n_envs={N_ENVS}")
print(f"[kaggle_train_sweep] repo={REPO_URL}@{REPO_BRANCH}")
print(f"[kaggle_train_sweep] sweep cells: {len(SWEEP_CELLS)}")
for i, cell in enumerate(SWEEP_CELLS):
    print(f"  cell[{i:02d}] = {cell}")


# ----------------------------------------------------------------------
# 1) Clone the repo and install deps. Same pattern as kaggle_train.py.
# ----------------------------------------------------------------------
def _run(cmd, **kw):
    print("$", " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=True, **kw)


if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)
_run(["git", "clone", "--depth=1", "-b", REPO_BRANCH, REPO_URL, str(REPO_DIR)])
os.chdir(REPO_DIR)

SRC_DIR = str(REPO_DIR / "src")
for k in [m for m in list(sys.modules) if m == "rl_swing" or m.startswith("rl_swing.")]:
    del sys.modules[k]
sys.path[:] = [SRC_DIR] + [p for p in sys.path if p != SRC_DIR]


def _ensure(import_name: str, pip_name: str | None = None) -> None:
    try:
        __import__(import_name)
    except ImportError:
        pkg = pip_name or import_name
        print(f"[kaggle_train_sweep] installing missing {pkg}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pkg], check=False,
        )


_ensure("stable_baselines3")
_ensure("gymnasium")
_ensure("sb3_contrib", "sb3-contrib")

import rl_swing  # noqa: E402

print(f"[kaggle_train_sweep] rl_swing.__file__ = {rl_swing.__file__}")


# ----------------------------------------------------------------------
# 2) Per-cell loop: train → validate → record.
# ----------------------------------------------------------------------
from rl_swing.rl.training.colab_entrypoint import train  # noqa: E402
from rl_swing.rl.validation.walk_forward import (  # noqa: E402
    validate_from_experiment,
)

seeds = None
if SEEDS_RAW:
    seeds = [int(s) for s in SEEDS_RAW.split(",") if s.strip()]
total_timesteps = int(TOTAL_STEPS) if TOTAL_STEPS else None

cell_results: list[dict] = []
sweep_t0 = time.time()
SUMMARY_PATH = WORKING / "sweep_summary.json"
PROGRESS_PATH = WORKING / "progress.json"


def _write_progress(*, cells_done: int, current: dict | None,
                    elapsed_s: float, status: str) -> None:
    """Per-cell heartbeat. Crash-safe: replaces atomically via tmp+rename
    so a Kaggle worker restart can't catch a half-written progress file.
    Cheap; written N+1 times per sweep (start + after each of N cells).

    A fresh process started after a worker restart can read this to
    decide whether the sweep already produced a usable partial summary;
    today the kernel just retries from cell 0, but downstream tooling
    (the host orchestrator) can branch on it.
    """
    payload = {
        "cells_done": int(cells_done),
        "cells_total": int(len(SWEEP_CELLS)),
        "current": current,
        "elapsed_s": float(elapsed_s),
        "status": status,
    }
    tmp = PROGRESS_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    tmp.replace(PROGRESS_PATH)


def _write_summary_partial(elapsed_s: float) -> None:
    """Per-cell incremental write of sweep_summary.json.

    The all-at-end aggregation in the previous version was lost on the
    M3.b run when Kaggle restarted the worker after cell 09 of 12, so
    every cell's metric data sat in process memory and never reached
    disk. This atomic-replace pattern guarantees that a worker restart
    can't lose more than the current in-flight cell.

    Cost: ~1-2 KB per cell × 12 cells = ~25 KB writes amortized over
    a 4 hr run. Negligible.
    """
    payload = {
        "experiment": EXPERIMENT,
        "total_timesteps_per_seed": total_timesteps,
        "seeds": seeds,
        "data_provider": DATA_PROVIDER,
        "n_envs": N_ENVS,
        "sweep_secs": float(elapsed_s),
        "n_cells": len(SWEEP_CELLS),
        "cells_completed": len(cell_results),
        "complete": False,  # set True only by the final write at end
        "cells": cell_results,
    }
    tmp = SUMMARY_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    tmp.replace(SUMMARY_PATH)


# Initial heartbeat: before any cell starts, so a very-fast restart
# still leaves a progress.json on disk.
_write_progress(cells_done=0, current=None, elapsed_s=0.0, status="starting")

for i, cell in enumerate(SWEEP_CELLS):
    _write_progress(
        cells_done=len(cell_results),
        current={"cell_idx": i, "hyperparam_overrides": cell},
        elapsed_s=time.time() - sweep_t0,
        status="cell_running",
    )
    print(f"\n========== cell {i:02d}/{len(SWEEP_CELLS) - 1}  hyperparams={cell} ==========")
    cell_artifact_root = ARTIFACTS / f"cell_{i:02d}"
    cell_artifact_root.mkdir(parents=True, exist_ok=True)
    cell_t0 = time.time()
    train_summary: dict | None = None
    train_error: str | None = None
    try:
        train_summary = train(
            experiment=EXPERIMENT,
            total_timesteps=total_timesteps,
            seeds=seeds,
            data_provider=DATA_PROVIDER,
            artifact_root=str(cell_artifact_root),
            n_envs=N_ENVS,
            hyperparam_overrides=dict(cell),
        )
    except Exception as e:  # pragma: no cover - Kaggle runtime
        train_error = f"{type(e).__name__}: {e}"
        print(f"[kaggle_train_sweep] cell {i:02d} training raised: {train_error}")
    train_secs = time.time() - cell_t0

    val_summary: dict | None = None
    val_error: str | None = None
    if train_summary is not None:
        try:
            val_dir = WORKING / f"cell_{i:02d}"
            val_dir.mkdir(parents=True, exist_ok=True)
            val_summary = validate_from_experiment(
                EXPERIMENT,
                report_dir=val_dir,
                artifact_root_override=str(cell_artifact_root),
                include_cost_stress=False,  # cost-stress isn't part of the M3.b gate
            )
            with open(val_dir / "validation_summary.json", "w", encoding="utf-8") as f:
                json.dump(val_summary, f, indent=2, default=str)
        except Exception as e:  # pragma: no cover - Kaggle runtime
            val_error = f"{type(e).__name__}: {e}"
            print(f"[kaggle_train_sweep] cell {i:02d} validate raised: {val_error}")

    cell_results.append({
        "cell_idx": i,
        "hyperparam_overrides": cell,
        "train_seconds": train_secs,
        "train_error": train_error,
        "train_summary": train_summary,
        "validation_summary": val_summary,
        "val_error": val_error,
    })
    # Per-cell incremental write so Kaggle worker restart can't lose
    # more than the in-flight cell. M3.b's first attempt lost 10 cells
    # to exactly this — see research/diary/2026-05-07_feat32_m3_kaggle_NO_GO.md.
    elapsed_now = time.time() - sweep_t0
    _write_summary_partial(elapsed_s=elapsed_now)
    _write_progress(
        cells_done=len(cell_results), current=None,
        elapsed_s=elapsed_now, status="cell_done",
    )
    print(
        f"[kaggle_train_sweep] cell {i:02d} done in {train_secs:.0f}s "
        f"(train_error={bool(train_error)} val_error={bool(val_error)}); "
        f"sweep_summary.json updated  ({len(cell_results)}/{len(SWEEP_CELLS)} cells)"
    )

sweep_secs = time.time() - sweep_t0


# ----------------------------------------------------------------------
# 3) Final aggregate sweep_summary.json with complete=True flag.
# Mirrors the per-cell partial writes above; the only difference is
# the ``complete`` flag and the absence of a current/in-flight cell.
# ----------------------------------------------------------------------
final_payload = {
    "experiment": EXPERIMENT,
    "total_timesteps_per_seed": total_timesteps,
    "seeds": seeds,
    "data_provider": DATA_PROVIDER,
    "n_envs": N_ENVS,
    "sweep_secs": sweep_secs,
    "n_cells": len(SWEEP_CELLS),
    "cells_completed": len(cell_results),
    "complete": True,
    "cells": cell_results,
}
with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump(final_payload, f, indent=2, default=str)
_write_progress(
    cells_done=len(cell_results), current=None,
    elapsed_s=sweep_secs, status="complete",
)
print(f"[kaggle_train_sweep] wrote {SUMMARY_PATH}  (total {sweep_secs:.0f}s)")
