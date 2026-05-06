#!/usr/bin/env python3
"""Run an rl-swing training experiment on Kaggle Notebooks, headless.

Workflow:

    1. Resolve your Kaggle username from ``~/.kaggle/kaggle.json``.
    2. Materialize a per-run kernel directory with ``kaggle_train.py``
       and a ``kernel-metadata.json`` (substituted from the template).
       The script's runtime knobs (experiment path, total timesteps,
       seeds, repo URL/branch) are baked into the kernel as
       env-var-style assignments at the top of the script.
    3. ``kaggle kernels push`` to upload + start.
    4. Poll ``kaggle kernels status`` until terminal.
    5. ``kaggle kernels output`` to download the artifacts into
       ``data/kaggle/<run_id>/``.

Usage:

    scripts/kaggle_run.py --experiment configs/experiments/ppo_filter_smoke.yaml \\
        --total-timesteps 16000

Optional flags:

    --seeds 11,22,33
    --repo-url https://github.com/<user>/trading-bot-rl.git
    --repo-branch main
    --slug <kernel-slug>     # defaults to '<experiment-stem>-<timestamp>'
    --no-wait                # push and exit; don't poll
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT     = Path(__file__).resolve().parents[1]
KAGGLE_DIR    = REPO_ROOT / "kaggle"
RUN_ROOT      = REPO_ROOT / "data" / "kaggle"
TEMPLATE_META = KAGGLE_DIR / "kernel-metadata.template.json"
SCRIPT_FILE   = KAGGLE_DIR / "kaggle_train.py"


# ----------------------------------------------------------------------
def _kaggle_username() -> str:
    cfg_path = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if not cfg_path.exists():
        sys.exit(f"Kaggle credentials not found at {cfg_path}. "
                 f"Get them from https://www.kaggle.com/settings/account.")
    with open(cfg_path, "rt", encoding="utf-8") as f:
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
    repo_url: str,
    repo_branch: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy script and inject the env-var defaults so this specific run
    # is fully self-contained on Kaggle (no env vars need to be set in
    # the kernel's settings).
    #
    # IMPORTANT: ``from __future__ import ...`` MUST be the first
    # statement in a Python file, so we have to inject AFTER any future
    # imports rather than at the very top.
    src = SCRIPT_FILE.read_text(encoding="utf-8")
    overrides_block = (
        "# --- injected by scripts/kaggle_run.py ---\n"
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
    overrides_block += "# --- end injection ---\n"

    # Find the spot to insert: after the last consecutive ``from __future__``
    # statement (and any preceding module docstring / comments / blanks).
    lines = src.splitlines(keepends=True)
    insert_at = 0
    in_docstring = False
    docstring_quote = ""
    last_future = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if in_docstring:
            if stripped.endswith(docstring_quote):
                in_docstring = False
            continue
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if (stripped.startswith('"""') or stripped.startswith("'''")):
            quote = stripped[:3]
            # One-line or multi-line module docstring
            if stripped.count(quote) >= 2 and len(stripped) > 3:
                continue
            in_docstring = True
            docstring_quote = quote
            continue
        if stripped.startswith("from __future__"):
            last_future = i
            continue
        # First non-future, non-comment, non-docstring line: stop scanning.
        break
    insert_at = last_future + 1 if last_future >= 0 else 0

    new_src = "".join(lines[:insert_at]) + "\n" + overrides_block + "\n" + "".join(lines[insert_at:])
    out_script = out_dir / "kaggle_train.py"
    out_script.write_text(new_src, encoding="utf-8")

    # Build kernel-metadata.json from the template.
    meta = TEMPLATE_META.read_text(encoding="utf-8")
    meta = meta.replace("{USERNAME}", username)
    meta = meta.replace("{SLUG}", slug)
    meta = meta.replace("{TITLE}", title)
    (out_dir / "kernel-metadata.json").write_text(meta, encoding="utf-8")


def _kaggle(*args: str, capture: bool = False) -> str:
    cmd = ["kaggle", *args]
    if capture:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return (res.stdout + res.stderr).strip()
    subprocess.run(cmd, check=True)
    return ""


def _wait_for_completion(kernel_ref: str, *, poll_seconds: float = 20.0,
                         timeout_seconds: float = 12 * 3600) -> str:
    """Poll until the kernel reaches a terminal state. Returns the final status."""
    started = time.time()
    last_status = ""
    while True:
        out = _kaggle("kernels", "status", kernel_ref, capture=True)
        # The CLI prints a couple of lines; pick the status line.
        m = re.search(r'has status "(\w+)"', out)
        status = m.group(1) if m else out.strip()
        if status != last_status:
            print(f"[kaggle_run] {datetime.utcnow().isoformat(timespec='seconds')} status={status}")
            last_status = status
        if status in {"complete", "error", "cancelAcknowledged", "cancelRequested"}:
            return status
        if time.time() - started > timeout_seconds:
            print(f"[kaggle_run] TIMEOUT after {timeout_seconds}s")
            return "timeout"
        time.sleep(poll_seconds)


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", required=True,
                    help="Path inside the repo, e.g. configs/experiments/ppo_filter_smoke.yaml")
    ap.add_argument("--total-timesteps", type=int, default=None)
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated, e.g. 11,22,33")
    ap.add_argument("--data-provider", type=str, default=None,
                    help="Override the experiment's data provider, "
                         "e.g. yfinance_daily, synthetic_momentum.")
    ap.add_argument("--repo-url", default="https://github.com/l2code/trading-bot-rl.git")
    ap.add_argument("--repo-branch", default="main")
    ap.add_argument("--slug", default=None,
                    help="Kaggle kernel slug. Defaults to <experiment-stem>-<timestamp>.")
    ap.add_argument("--title", default=None,
                    help="Kernel title for the Kaggle UI.")
    ap.add_argument("--no-wait", action="store_true",
                    help="Push and exit; don't poll for completion.")
    args = ap.parse_args()

    username = _kaggle_username()

    exp_stem = Path(args.experiment).stem
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    slug = _slugify(args.slug or f"rl-swing-{exp_stem}-{ts}")
    title = args.title or f"rl-swing {exp_stem} {ts}"
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
        repo_url=args.repo_url,
        repo_branch=args.repo_branch,
    )
    print(f"[kaggle_run] materialized kernel at {run_dir}")
    print(f"[kaggle_run] pushing as {kernel_ref}")
    _kaggle("kernels", "push", "-p", str(run_dir))

    if args.no_wait:
        print(f"[kaggle_run] pushed. Track at https://www.kaggle.com/code/{kernel_ref}")
        return 0

    print(f"[kaggle_run] polling {kernel_ref}; this may take minutes-hours...")
    status = _wait_for_completion(kernel_ref)
    print(f"[kaggle_run] terminal status: {status}")

    out_dir = run_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[kaggle_run] downloading outputs to {out_dir}")
    _kaggle("kernels", "output", kernel_ref, "-p", str(out_dir))

    summary = out_dir / "summary.json"
    if summary.exists():
        print(f"[kaggle_run] summary.json:")
        print(summary.read_text())
    else:
        print(f"[kaggle_run] WARN: no summary.json in output. "
              f"Check the Kaggle UI at https://www.kaggle.com/code/{kernel_ref}")

    return 0 if status == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
