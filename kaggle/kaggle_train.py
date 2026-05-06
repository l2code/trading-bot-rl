"""rl-swing training on Kaggle Notebooks (script kernel).

This file is the actual code that runs on Kaggle. It is uploaded by
``scripts/kaggle_run.py`` along with the kernel-metadata.json next to
it. The wrapper script substitutes runtime values (experiment name,
total timesteps, seeds, GitHub URL/branch) before pushing.

Outputs are written to ``/kaggle/working/`` which Kaggle auto-collects
and the orchestrator downloads back via ``kaggle kernels output``.

Configuration is via env vars so the same script can be re-pushed for
different experiments without editing the script body:

    RL_SWING_EXPERIMENT     — path inside the cloned repo, e.g.
                              "configs/experiments/ppo_filter_smoke.yaml"
                              (default).
    RL_SWING_TOTAL_TIMESTEPS — override total timesteps. Default: use
                              experiment file's value.
    RL_SWING_SEEDS          — comma-separated seeds, e.g. "11,22,33".
                              Default: experiment file's seed list.
    RL_SWING_REPO_URL       — git URL to clone. Default:
                              "https://github.com/l2code/trading-bot-rl.git".
    RL_SWING_REPO_BRANCH    — branch to checkout. Default: "main".
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
EXPERIMENT  = os.environ.get(
    "RL_SWING_EXPERIMENT", "configs/experiments/ppo_filter_smoke.yaml"
)
TOTAL_STEPS = os.environ.get("RL_SWING_TOTAL_TIMESTEPS")  # may be None
SEEDS_RAW   = os.environ.get("RL_SWING_SEEDS")            # may be None
REPO_URL    = os.environ.get(
    "RL_SWING_REPO_URL", "https://github.com/l2code/trading-bot-rl.git"
)
REPO_BRANCH = os.environ.get("RL_SWING_REPO_BRANCH", "main")
WORKING     = Path("/kaggle/working")
REPO_DIR    = WORKING / "trading-bot-rl"
ARTIFACTS   = WORKING / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

print(f"[kaggle_train] experiment={EXPERIMENT!r}")
print(f"[kaggle_train] total_timesteps={TOTAL_STEPS!r}  seeds={SEEDS_RAW!r}")
print(f"[kaggle_train] repo={REPO_URL}@{REPO_BRANCH}")


# ----------------------------------------------------------------------
# 1) Clone the repo and pip-install (non-editable; Colab/Kaggle prefer it).
# ----------------------------------------------------------------------
def _run(cmd, **kw):
    print("$", " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=True, **kw)


if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)
_run(["git", "clone", "--depth=1", "-b", REPO_BRANCH, REPO_URL, str(REPO_DIR)])
os.chdir(REPO_DIR)

# Don't ``pip install .`` — Kaggle's pip flow has been observed to leave
# subpackages off the install path (we hit ``ModuleNotFoundError: No
# module named 'rl_swing.rl.env'`` despite the built wheel containing
# it). Easier and more reliable: add the src directory to sys.path
# directly. Kaggle's base image already has every runtime dep we need
# (numpy, pandas, torch, sklearn, gymnasium, click, pyyaml, pyarrow,
# yfinance), and we install stable-baselines3 below if it's missing.
sys.path.insert(0, str(REPO_DIR / "src"))


def _ensure(import_name: str, pip_name: str | None = None) -> None:
    try:
        __import__(import_name)
    except ImportError:
        pkg = pip_name or import_name
        print(f"[kaggle_train] installing missing {pkg}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            check=False,
        )


_ensure("stable_baselines3")
_ensure("gymnasium")


# ----------------------------------------------------------------------
# 2) Train.
# ----------------------------------------------------------------------
from rl_swing.rl.training.colab_entrypoint import train  # noqa: E402

seeds = None
if SEEDS_RAW:
    seeds = [int(s) for s in SEEDS_RAW.split(",") if s.strip()]

t0 = time.time()
summary = train(
    experiment=EXPERIMENT,
    total_timesteps=int(TOTAL_STEPS) if TOTAL_STEPS else None,
    seeds=seeds,
    artifact_root=str(ARTIFACTS),
)
elapsed = time.time() - t0
print(f"[kaggle_train] training finished in {elapsed:.1f}s")


# ----------------------------------------------------------------------
# 3) Write a top-level summary so the orchestrator can pick it up
#    without sifting the artifact tree.
# ----------------------------------------------------------------------
summary_path = WORKING / "summary.json"
with open(summary_path, "wt", encoding="utf-8") as f:
    json.dump({
        "experiment": EXPERIMENT,
        "elapsed_seconds": elapsed,
        "summary": summary,
    }, f, indent=2, default=str)
print(f"[kaggle_train] wrote summary to {summary_path}")


# ----------------------------------------------------------------------
# 4) Optional: try walk-forward validation. Failures here don't fail the run.
# ----------------------------------------------------------------------
try:
    from rl_swing.rl.validation.walk_forward import validate_from_experiment

    report_dir = WORKING / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    val_summary = validate_from_experiment(
        EXPERIMENT,
        report_dir=report_dir,
        include_cost_stress=True,
    )
    with open(WORKING / "validation_summary.json", "wt", encoding="utf-8") as f:
        json.dump(val_summary, f, indent=2, default=str)
    print(f"[kaggle_train] wrote validation_summary.json")
except Exception as e:
    print(f"[kaggle_train] WARN: validation step failed: {e}")
