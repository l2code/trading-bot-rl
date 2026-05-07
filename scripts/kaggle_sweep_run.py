#!/usr/bin/env python3
"""Push a hyperparam-sweep run to Kaggle Notebooks (FEAT-32 M3.b).

Sister script to ``scripts/kaggle_run.py`` that uses
``kaggle/kaggle_train_sweep.py`` as the kernel body and injects the
sweep grid as a JSON env var. Per cell the kernel runs ``train()``
over all configured seeds (using the cell's ``hyperparam_overrides``)
and then ``validate_from_experiment`` against the cell's artifacts.

Output (downloaded back via ``kaggle kernels output``) lands in
``data/kaggle/<slug>/output/sweep_summary.json``.

Usage::

    scripts/kaggle_sweep_run.py \\
        --experiment configs/experiments/ppo_portfolio_v003.yaml \\
        --total-timesteps 100000 \\
        --seeds 11,22,33 \\
        --data-provider yfinance_daily \\
        --n-envs 4 \\
        --slug rl-swing-v003-m3b-sweep \\
        --title "rl-swing v003 m3b sweep" \\
        --grid sweep_grid_v003_m3b.json \\
        --no-wait

The grid file is a JSON list of dicts, e.g.::

    [
      {"ent_coef": 0.01, "learning_rate": 0.0001},
      {"ent_coef": 0.05, "learning_rate": 0.0003},
      ...
    ]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT     = Path(__file__).resolve().parents[1]
KAGGLE_DIR    = REPO_ROOT / "kaggle"
RUN_ROOT      = REPO_ROOT / "data" / "kaggle"
TEMPLATE_META = KAGGLE_DIR / "kernel-metadata.template.json"
SCRIPT_FILE   = KAGGLE_DIR / "kaggle_train_sweep.py"


def _kaggle_username() -> str:
    cfg_path = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if not cfg_path.exists():
        sys.exit(f"Kaggle credentials not found at {cfg_path}.")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if "username" not in cfg:
        sys.exit(f"Bad kaggle.json: no 'username' key in {cfg_path}")
    return cfg["username"]


def _slugify(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9-]+", "-", s).strip("-").lower()
    return s[:60] or "run"


def _materialize_kernel(
    out_dir: Path,
    *,
    username: str,
    slug: str,
    title: str,
    experiment: str,
    total_timesteps: int | None,
    seeds: list[int] | None,
    data_provider: str | None,
    n_envs: int,
    repo_url: str,
    repo_branch: str,
    sweep_cells: list[dict],
    is_private: bool,
) -> None:
    """Copy ``kaggle_train_sweep.py`` to ``out_dir`` with the runtime
    knobs injected as env-var defaults at the top, and write the
    kernel-metadata.json next to it. Mirrors ``kaggle_run._materialize_kernel``
    but adds ``RL_SWING_SWEEP_CELLS`` and uses the sweep kernel script."""
    out_dir.mkdir(parents=True, exist_ok=True)

    src = SCRIPT_FILE.read_text(encoding="utf-8")
    overrides_block = (
        "# --- injected by scripts/kaggle_sweep_run.py ---\n"
        "import os as _os\n"
        f"_os.environ.setdefault('RL_SWING_EXPERIMENT', {experiment!r})\n"
        f"_os.environ.setdefault('RL_SWING_REPO_URL', {repo_url!r})\n"
        f"_os.environ.setdefault('RL_SWING_REPO_BRANCH', {repo_branch!r})\n"
    )
    if total_timesteps is not None:
        overrides_block += f"_os.environ.setdefault('RL_SWING_TOTAL_TIMESTEPS', {str(total_timesteps)!r})\n"
    if seeds:
        overrides_block += f"_os.environ.setdefault('RL_SWING_SEEDS', {','.join(map(str, seeds))!r})\n"
    if data_provider:
        overrides_block += f"_os.environ.setdefault('RL_SWING_DATA_PROVIDER', {data_provider!r})\n"
    if n_envs and n_envs > 1:
        overrides_block += f"_os.environ.setdefault('RL_SWING_N_ENVS', {str(n_envs)!r})\n"
    overrides_block += (
        f"_os.environ.setdefault('RL_SWING_SWEEP_CELLS', "
        f"{json.dumps(sweep_cells)!r})\n"
    )
    overrides_block += "# --- end injection ---\n"

    lines = src.splitlines(keepends=True)
    in_docstring = False
    docstring_quote = ""
    last_future = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if in_docstring:
            if stripped.endswith(docstring_quote):
                in_docstring = False
            continue
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote = stripped[:3]
            if stripped.count(quote) >= 2 and len(stripped) > 3:
                continue
            in_docstring = True
            docstring_quote = quote
            continue
        if stripped.startswith("from __future__"):
            last_future = i
            continue
        break
    insert_at = last_future + 1 if last_future >= 0 else 0

    new_src = "".join(lines[:insert_at]) + "\n" + overrides_block + "\n" + "".join(lines[insert_at:])
    out_script = out_dir / "kaggle_train_sweep.py"
    out_script.write_text(new_src, encoding="utf-8")

    meta = json.loads(TEMPLATE_META.read_text(encoding="utf-8"))
    meta["id"] = f"{username}/{slug}"
    meta["title"] = title
    meta["code_file"] = "kaggle_train_sweep.py"
    meta["is_private"] = "true" if is_private else "false"
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8",
    )


def _kaggle(*args: str, capture: bool = False) -> str:
    cmd = ["kaggle", *args]
    if capture:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return (res.stdout + res.stderr).strip()
    subprocess.run(cmd, check=True)
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--total-timesteps", type=int, default=None)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--data-provider", type=str, default=None)
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--repo-url", default="https://github.com/l2code/trading-bot-rl.git")
    ap.add_argument("--repo-branch", default="main")
    ap.add_argument("--slug", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--grid", required=True,
                    help="Path to a JSON file (list of dicts) OR an inline JSON list.")
    ap.add_argument("--private", action="store_true", default=True,
                    help="Push as private kernel (default: True).")
    ap.add_argument("--public", dest="private", action="store_false",
                    help="Push as public kernel (overrides --private).")
    ap.add_argument("--no-wait", action="store_true",
                    help="Push and exit without polling.")
    args = ap.parse_args()

    # Parse grid.
    grid_arg = args.grid
    sweep_cells: list[dict]
    if Path(grid_arg).exists():
        sweep_cells = json.loads(Path(grid_arg).read_text(encoding="utf-8"))
    else:
        sweep_cells = json.loads(grid_arg)
    if not isinstance(sweep_cells, list) or not sweep_cells:
        sys.exit("--grid must be a non-empty JSON list (file or inline)")
    for cell in sweep_cells:
        if not isinstance(cell, dict):
            sys.exit("each grid cell must be a JSON object")

    username = _kaggle_username()
    exp_stem = Path(args.experiment).stem
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    slug = _slugify(args.slug or f"rl-swing-{exp_stem}-sweep-{ts}")
    title = args.title or f"rl-swing {exp_stem} sweep {ts}"
    kernel_ref = f"{username}/{slug}"

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None

    run_dir = RUN_ROOT / slug
    if run_dir.exists():
        shutil.rmtree(run_dir)

    _materialize_kernel(
        run_dir,
        username=username, slug=slug, title=title,
        experiment=args.experiment,
        total_timesteps=args.total_timesteps,
        seeds=seeds,
        data_provider=args.data_provider,
        n_envs=args.n_envs,
        repo_url=args.repo_url,
        repo_branch=args.repo_branch,
        sweep_cells=sweep_cells,
        is_private=args.private,
    )
    print(f"[kaggle_sweep_run] materialized kernel at {run_dir}")
    print(f"[kaggle_sweep_run] {len(sweep_cells)} cells; private={args.private}")
    print(f"[kaggle_sweep_run] pushing as {kernel_ref}")
    _kaggle("kernels", "push", "-p", str(run_dir))

    if args.no_wait:
        print(f"[kaggle_sweep_run] pushed. Track at https://www.kaggle.com/code/{kernel_ref}")
        return 0

    print("[kaggle_sweep_run] kernel pushed; polling not implemented in this launcher")
    return 0


if __name__ == "__main__":
    sys.exit(main())
