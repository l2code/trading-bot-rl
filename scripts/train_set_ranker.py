"""Offline trainer for the SlateEncoder-based set ranker (FEAT-34 PR-1).

Walks the training window, simulates every fired (pack × strategy)
trade with the SAME ExecutionSimulator + cost model the v002 env
uses, then trains a SlateEncoder via per-slate regression: for each
pack, predict every fired slot's realized risk-adjusted return AND
the skip-counterfactual return. Loss = MSE summed over the slate.

This is the cheap supervised diagnostic for the slate-encoder
inductive bias. PR-2 (gated on this scorer materially improving
over the HistGB FEAT-7 ranker) wires the same encoder as a sb3
features extractor for MaskablePPO.

Usage:
    python scripts/train_set_ranker.py \\
        --experiment configs/experiments/ppo_selector_v002_masked.yaml \\
        --data-provider yfinance_daily \\
        --epochs 30 --batch-size 256 --lr 1e-3
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from rl_swing.adapters.data.synthetic_provider import SyntheticProvider  # noqa: E402
from rl_swing.adapters.data.yfinance_provider import YFinanceProvider  # noqa: E402
from rl_swing.domain import PortfolioState  # noqa: E402
from rl_swing.features.pipelines import CoreDailyPipeline  # noqa: E402
from rl_swing.rl.agents.set_ranker_scorer import (  # noqa: E402
    CTX_FEATURE_NAMES,
    SLOT_FEATURE_NAMES,
    build_ctx_features,
    build_slot_feature_row,
    build_slot_mask,
)
from rl_swing.rl.agents.slate_encoder import (  # noqa: E402
    SlateEncoder,
    SlateEncoderConfig,
)
from rl_swing.rl.env.cost_model import EquityExecutionModel  # noqa: E402
from rl_swing.rl.env.execution_simulator import ExecutionSimulator  # noqa: E402
from rl_swing.strategies.breakout import BreakoutStrategy  # noqa: E402
from rl_swing.strategies.mean_reversion import RsiMeanReversionStrategy  # noqa: E402
from rl_swing.strategies.momentum import MomentumStrategy  # noqa: E402
from rl_swing.strategies.multi_strategy_packer import MultiStrategyPacker  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger("train_set_ranker")


_FEATURE_WARMUP_DAYS = int(252 * 1.5)


def _build_provider(name: str):
    if name == "yfinance_daily":
        return YFinanceProvider(auto_adjust=True)
    if name.startswith("synthetic_"):
        regime = name.replace("synthetic_", "")
        return SyntheticProvider(regime=regime, seed=11)
    raise ValueError(f"unknown data provider: {name}")


def _load_universe(name: str) -> list[str]:
    path = _REPO_ROOT / "configs" / "universes" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"universe not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    nested = (cfg.get("universe") or {}).get("symbols") or []
    flat = cfg.get("symbols") or []
    return list(nested or flat)


def _build_default_strategies():
    return [
        MomentumStrategy(
            min_relative_strength=-0.05, min_r20=-0.02,
            require_sma200_above=False,
        ),
        RsiMeanReversionStrategy(rsi_threshold=35.0),
        BreakoutStrategy(
            min_relative_volume=0.7, max_distance_below_high=-0.02,
        ),
    ]


def _simulate_outcome(candidate, frame, by_symbol, cost_model, sim):
    atr_pct = float(frame.values.get("atr_pct_14", 0.02))
    rv20 = float(frame.values.get("realized_vol_20", 0.20))
    vol_percentile = min(1.0, max(0.0, rv20 / 0.6))
    adv = float(frame.values.get("dollar_volume", 0.0))
    notional = 100_000.0 * candidate.base_size_pct
    cost_bps = cost_model.cost_bps(
        atr_pct=atr_pct, volatility_percentile=vol_percentile,
        in_event_window=False, notional=notional, avg_dollar_volume=adv,
    )
    bars = by_symbol.get(candidate.symbol) or []
    idx = -1
    for i, b in enumerate(bars):
        if b.timestamp == candidate.as_of:
            idx = i
            break
    if idx < 0:
        return None
    return sim.simulate(
        bars=bars, entry_index=idx,
        size_pct=candidate.base_size_pct,
        max_holding_days=candidate.max_holding_days,
        cost_bps=cost_bps, atr_pct=atr_pct,
        starting_equity=100_000.0,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True)
    p.add_argument("--data-provider", default=None)
    p.add_argument("--output", default="data/models/selector_baseline_set_ranker/model.pt")
    p.add_argument("--target-risk-pct", type=float, default=0.02)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    # FEAT-34 PR-1b stabilization: lower default LR + warmup + grad clip.
    # PR-1's lr=1e-3 caused end-stage divergence around epoch 23.
    p.add_argument("--lr", type=float, default=5e-4,
                   help="Base LR after warmup (PR-1b lowered from 1e-3 -> 5e-4).")
    p.add_argument("--warmup-epochs", type=int, default=3,
                   help="Linear LR warmup from lr/10 to lr over the first "
                        "N epochs. 0 disables warmup.")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="torch.nn.utils.clip_grad_norm_ max_norm. 0 disables.")
    p.add_argument("--embed-dim", type=int, default=32)
    # PR-1b: support a comma-separated seed list (e.g. "11,22,33"). The
    # trainer fits each seed independently and persists the best-by-val.
    p.add_argument("--seeds", type=str, default="11",
                   help="Comma-separated torch / numpy seeds (e.g. '11,22,33'). "
                        "Each seed trains an independent encoder; the best-by-"
                        "val-loss seed's weights are saved as the artifact.")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="Fraction of packs held out for early-stopping eval.")
    args = p.parse_args()

    with open(args.experiment, encoding="utf-8") as f:
        exp = (yaml.safe_load(f) or {}).get("experiment") or {}

    universe = exp.get("universe", "starter_equities")
    provider_name = args.data_provider or exp.get("data_provider", "synthetic_momentum")
    train_start = date.fromisoformat(exp["train_start"])
    train_end = date.fromisoformat(exp["train_end"])

    cost_cfg = exp.get("cost_model") or {}
    cost_model = EquityExecutionModel(**cost_cfg)
    sim = ExecutionSimulator()

    # 1) bars + frames (warmup mirrors the trainer)
    from datetime import timedelta
    warmup_start = train_start - timedelta(days=_FEATURE_WARMUP_DAYS)
    provider = _build_provider(provider_name)
    symbols = _load_universe(universe)
    _log.info("loading bars provider=%s universe=%s symbols=%d window=%s..%s (warmup=%s)",
              provider_name, universe, len(symbols), train_start, train_end, warmup_start)
    t0 = time.time()
    bars = list(provider.get_bars(symbols, warmup_start, train_end, "1d", True))
    _log.info("loaded %d bars in %.1fs", len(bars), time.time() - t0)

    pipeline = CoreDailyPipeline()
    all_frames = list(pipeline.build(bars))
    frames = [f for f in all_frames if train_start <= f.as_of.date() <= train_end]
    frames_by_key = {(f.symbol, f.as_of): f for f in frames}
    _log.info("built %d feature frames in window", len(frames))

    portfolio = PortfolioState(
        as_of=datetime(train_end.year, train_end.month, train_end.day),
        cash=100_000.0, equity=100_000.0,
    )
    packer = MultiStrategyPacker(_build_default_strategies())
    packs = packer.pack(frames, portfolio)
    n_slots = packer.n_slots
    _log.info("packed %d packs (n_slots=%d)", len(packs), n_slots)

    by_symbol: dict[str, list] = {}
    for b in bars:
        by_symbol.setdefault(b.symbol, []).append(b)
    for sym in by_symbol:
        by_symbol[sym].sort(key=lambda b: b.timestamp)

    # 2) Build per-pack tensors. Each pack contributes one row of:
    #   slot_features  : (n_slots, slot_feat_dim)  — zero on unfired
    #   slot_mask      : (n_slots,)
    #   ctx            : (ctx_dim,)
    #   slot_targets   : (n_slots,)  — risk_adj return per fired slot, 0 on unfired
    #   slot_target_mask : (n_slots,)  — 1 where the regression target is valid
    #   skip_target    : scalar      — risk_adj return of the highest-signal CF
    #
    # All slot targets share the same simulator results that the env uses.
    slot_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    ctx_rows: list[np.ndarray] = []
    slot_targets: list[np.ndarray] = []
    slot_target_masks: list[np.ndarray] = []
    skip_targets: list[float] = []
    n_dropped = 0

    for pack in packs:
        frame = frames_by_key.get((pack.symbol, pack.as_of))
        if frame is None:
            continue
        slot_feats = np.stack(
            [build_slot_feature_row(pack, k) for k in range(n_slots)], axis=0,
        )
        mask = build_slot_mask(pack, n_slots)
        ctx = build_ctx_features(pack, frame, n_slots)

        # Per-slot regression targets.
        targets = np.zeros(n_slots, dtype=np.float32)
        target_mask = np.zeros(n_slots, dtype=np.float32)
        outcomes_for_skip = []
        for k, c in enumerate(pack.candidates):
            if c is None:
                continue
            outcome = _simulate_outcome(c, frame, by_symbol, cost_model, sim)
            if outcome is None:
                continue
            r_adj = float(outcome.return_pct) / float(args.target_risk_pct)
            targets[k] = r_adj
            target_mask[k] = 1.0
            outcomes_for_skip.append((float(c.signal_strength), r_adj))
        if not outcomes_for_skip:
            n_dropped += 1
            continue
        # Skip target: the *negation* of the highest-signal counterfactual's
        # risk_adj return. Skipping is "good" iff the best-by-signal trade
        # would have been negative — so the skip head learns to dominate
        # when no slot has high predicted positive return. This mirrors
        # the env's reward_for_skip semantics under skip_counterfactual_mode
        # = "highest_signal" (FIX-#26).
        outcomes_for_skip.sort(key=lambda kv: -kv[0])  # sort by signal desc
        best_signal_r = outcomes_for_skip[0][1]
        skip_target = -best_signal_r

        slot_rows.append(slot_feats)
        mask_rows.append(mask)
        ctx_rows.append(ctx)
        slot_targets.append(targets)
        slot_target_masks.append(target_mask)
        skip_targets.append(skip_target)

    if not slot_rows:
        _log.error("no training rows — abort")
        return 1

    X_slot = np.stack(slot_rows, axis=0)
    X_mask = np.stack(mask_rows, axis=0)
    X_ctx = np.stack(ctx_rows, axis=0)
    Y_slot = np.stack(slot_targets, axis=0)
    Y_slot_mask = np.stack(slot_target_masks, axis=0)
    Y_skip = np.asarray(skip_targets, dtype=np.float32).reshape(-1, 1)
    n = X_slot.shape[0]
    _log.info("training set: n_packs=%d slot_feat_dim=%d ctx_dim=%d dropped=%d",
              n, X_slot.shape[-1], X_ctx.shape[-1], n_dropped)

    # FEAT-34 PR-1b: feature standardization. The ctx features include
    # raw frame fields (e.g. dollar_volume, prices) that span many
    # orders of magnitude; without standardization the linear layers
    # in phi/rho produce massive predictions and the MSE loss
    # explodes (PR-1b diagnosis). Compute train-only mean/std (single
    # split for all seeds — this is a feature pipeline transformation,
    # not a per-seed data augmentation), persist with the bundle so
    # inference applies the same transformation.
    ctx_mean = X_ctx.mean(axis=0).astype(np.float32)
    ctx_std = X_ctx.std(axis=0).astype(np.float32)
    ctx_std = np.where(ctx_std < 1e-6, 1.0, ctx_std)  # avoid div-by-zero on constant columns
    X_ctx = ((X_ctx - ctx_mean) / ctx_std).astype(np.float32)
    # Per-slot features are mostly already in well-behaved ranges
    # (signal_strength in [0,1], base_size_pct < 0.2, etc.), but
    # standardize anyway for symmetry and so a future feature
    # addition doesn't silently regress training.
    slot_flat = X_slot.reshape(-1, X_slot.shape[-1])
    slot_mean = slot_flat.mean(axis=0).astype(np.float32)
    slot_std = slot_flat.std(axis=0).astype(np.float32)
    slot_std = np.where(slot_std < 1e-6, 1.0, slot_std)
    X_slot = ((X_slot - slot_mean) / slot_std).astype(np.float32)
    _log.info("standardized features: ctx mean/std and per-slot mean/std computed on full train set")

    # 3) Train. FEAT-34 PR-1b: multi-seed loop + LR warmup + gradient
    # clipping + rank/top-1 diagnostics on the val set.
    import torch
    from torch import optim
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        _log.error("no seeds parsed from --seeds=%r", args.seeds)
        return 1

    cfg = SlateEncoderConfig(
        slot_feature_dim=X_slot.shape[-1],
        ctx_feature_dim=X_ctx.shape[-1],
        n_slots=n_slots,
        embed_dim=args.embed_dim,
    )

    def loss_fn(out, y_slot, y_smask, y_skip):
        # Per-slot MSE only at fired slots (where the target is valid).
        per_slot_se = (out["slot_logits"] - y_slot) ** 2
        masked_se = per_slot_se * y_smask
        n_valid = y_smask.sum().clamp(min=1.0)
        slot_loss = masked_se.sum() / n_valid
        skip_loss = ((out["skip_logit"] - y_skip) ** 2).mean()
        return slot_loss + skip_loss

    def diag_top1(model_, slot_t_, mask_t_, ctx_t_, y_slot_, y_smask_):
        """PR-1b: rank/top-1 diagnostic. For each pack with >=1 fired
        slot whose target is valid, compute predicted argmax slot vs
        ground-truth argmax slot. Returns (top1_acc, n_packs_used)."""
        model_.eval()
        with torch.no_grad():
            out = model_(slot_t_, mask_t_, ctx_t_)
        # Mask predictions and targets to fired slots only.
        very_neg = torch.finfo(out["slot_logits"].dtype).min
        masked_pred = out["slot_logits"].masked_fill(~y_smask_.bool(), very_neg)
        masked_tgt = y_slot_.masked_fill(~y_smask_.bool(), very_neg)
        # Skip packs with zero valid slots.
        any_valid = y_smask_.sum(dim=1) > 0
        if not any_valid.any():
            return 0.0, 0
        pred_top = masked_pred.argmax(dim=1)
        tgt_top = masked_tgt.argmax(dim=1)
        match = (pred_top == tgt_top) & any_valid
        n_used = int(any_valid.sum())
        return float(match.sum().item()) / max(1, n_used), n_used

    def run_single_seed(seed: int) -> dict:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SlateEncoder(cfg)
        opt = optim.Adam(model.parameters(), lr=args.lr)

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_val = max(1, int(n * args.val_frac))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        def to_t(arr, idx, dtype=torch.float32):
            return torch.from_numpy(arr[idx]).to(dtype)

        val_slot = to_t(X_slot, val_idx)
        val_mask = to_t(X_mask, val_idx)
        val_ctx = to_t(X_ctx, val_idx)
        val_y_slot = to_t(Y_slot, val_idx)
        val_y_smask = to_t(Y_slot_mask, val_idx)
        val_y_skip = to_t(Y_skip, val_idx)

        best_val = float("inf")
        best_state = None
        best_top1 = 0.0
        history = []
        t_seed = time.time()

        for epoch in range(args.epochs):
            # PR-1b: linear LR warmup from lr/10 -> lr over the first
            # warmup_epochs. Stabilizes early training; combined with
            # grad-clip removes the late-epoch divergence we saw in PR-1.
            if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
                warmup_factor = (epoch + 1) / args.warmup_epochs
                cur_lr = args.lr * (0.1 + 0.9 * warmup_factor)
            else:
                cur_lr = args.lr
            for pg in opt.param_groups:
                pg["lr"] = cur_lr

            model.train()
            ep_perm = rng.permutation(tr_idx)
            ep_loss = 0.0
            ep_n = 0
            for i in range(0, len(ep_perm), args.batch_size):
                batch_idx = ep_perm[i:i + args.batch_size]
                slot_t = to_t(X_slot, batch_idx)
                mask_t = to_t(X_mask, batch_idx)
                ctx_t = to_t(X_ctx, batch_idx)
                y_slot_t = to_t(Y_slot, batch_idx)
                y_smask_t = to_t(Y_slot_mask, batch_idx)
                y_skip_t = to_t(Y_skip, batch_idx)
                opt.zero_grad()
                out = model(slot_t, mask_t, ctx_t)
                loss = loss_fn(out, y_slot_t, y_smask_t, y_skip_t)
                loss.backward()
                # PR-1b: gradient clipping.
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=args.grad_clip,
                    )
                opt.step()
                ep_loss += float(loss.detach()) * len(batch_idx)
                ep_n += len(batch_idx)
            train_loss = ep_loss / max(1, ep_n)

            model.eval()
            with torch.no_grad():
                val_out = model(val_slot, val_mask, val_ctx)
                val_loss = float(loss_fn(val_out, val_y_slot, val_y_smask, val_y_skip))
            top1_acc, n_top1_packs = diag_top1(
                model, val_slot, val_mask, val_ctx, val_y_slot, val_y_smask,
            )
            history.append({
                "epoch": epoch, "lr": cur_lr,
                "train_loss": train_loss, "val_loss": val_loss,
                "val_top1_acc": top1_acc, "val_top1_n_packs": n_top1_packs,
            })
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                best_top1 = top1_acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            _log.info(
                "seed=%d epoch=%2d lr=%.5f train_loss=%.5f val_loss=%.5f "
                "val_top1=%.4f (n=%d) %s",
                seed, epoch, cur_lr, train_loss, val_loss,
                top1_acc, n_top1_packs, "(NEW BEST)" if improved else "",
            )
        _log.info(
            "seed=%d fit took %.1fs; best val_loss=%.5f best_top1=%.4f",
            seed, time.time() - t_seed, best_val, best_top1,
        )
        return {
            "seed": seed,
            "best_val_loss": float(best_val),
            "best_top1_acc": float(best_top1),
            "best_state": best_state,
            "history": history,
        }

    # Train each seed; keep the best-by-val artifact.
    seed_runs = [run_single_seed(s) for s in seeds]
    best_seed_run = min(seed_runs, key=lambda r: r["best_val_loss"])
    best_state = best_seed_run["best_state"]
    best_val = best_seed_run["best_val_loss"]
    best_top1 = best_seed_run["best_top1_acc"]
    _log.info(
        "best across %d seeds: seed=%d best_val_loss=%.5f best_top1=%.4f",
        len(seed_runs), best_seed_run["seed"], best_val, best_top1,
    )
    # Per-seed summary line (for the diary):
    _log.info("per-seed summary:")
    for r in seed_runs:
        _log.info(
            "  seed=%d best_val=%.5f best_top1=%.4f",
            r["seed"], r["best_val_loss"], r["best_top1_acc"],
        )

    # Restore best.
    model = SlateEncoder(cfg)
    model.load_state_dict(best_state)
    model.eval()

    # 4) Persist.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "state_dict": model.state_dict(),
        "config": cfg,
        "slot_feature_names": SLOT_FEATURE_NAMES,
        "ctx_feature_names": CTX_FEATURE_NAMES,
        "n_strategies": n_slots,
        "target_risk_pct": float(args.target_risk_pct),
        "trained_at": datetime.utcnow().isoformat(),
        "n_train_examples": int(n),
        "best_val_loss": float(best_val),
        "best_val_top1_acc": float(best_top1),
        "best_seed": int(best_seed_run["seed"]),
        "seeds": seeds,
        "per_seed_best_val_loss": [
            {"seed": r["seed"], "best_val_loss": r["best_val_loss"],
             "best_top1_acc": r["best_top1_acc"]}
            for r in seed_runs
        ],
        "data_provider": provider_name,
        "universe": universe,
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_epochs": args.warmup_epochs,
        "grad_clip": args.grad_clip,
        "embed_dim": args.embed_dim,
        # FEAT-34 PR-1b: feature normalization stats. Inference must
        # apply (x - mean) / std before feeding into the encoder.
        "ctx_mean": ctx_mean,
        "ctx_std": ctx_std,
        "slot_mean": slot_mean,
        "slot_std": slot_std,
    }
    torch.save(bundle, str(out_path))

    import json
    meta = {k: (v if not hasattr(v, "shape") else "<tensor>") for k, v in bundle.items()
            if k != "state_dict"}
    meta["config"] = {
        "slot_feature_dim": cfg.slot_feature_dim,
        "ctx_feature_dim": cfg.ctx_feature_dim,
        "n_slots": cfg.n_slots,
        "embed_dim": cfg.embed_dim,
    }
    meta_path = out_path.with_suffix(".metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    _log.info("wrote %s and %s", out_path, meta_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
