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
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=11)
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

    # 3) Train.
    import torch
    from torch import optim
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = SlateEncoderConfig(
        slot_feature_dim=X_slot.shape[-1],
        ctx_feature_dim=X_ctx.shape[-1],
        n_slots=n_slots,
        embed_dim=args.embed_dim,
    )
    model = SlateEncoder(cfg)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # train/val split
    rng = np.random.default_rng(args.seed)
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

    def loss_fn(out, y_slot, y_smask, y_skip):
        # Per-slot MSE only at fired slots (where the target is valid).
        per_slot_se = (out["slot_logits"] - y_slot) ** 2
        masked_se = per_slot_se * y_smask
        n_valid = y_smask.sum().clamp(min=1.0)
        slot_loss = masked_se.sum() / n_valid
        skip_loss = ((out["skip_logit"] - y_skip) ** 2).mean()
        return slot_loss + skip_loss

    best_val = float("inf")
    best_state = None
    t0 = time.time()
    for epoch in range(args.epochs):
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
            opt.step()
            ep_loss += float(loss.detach()) * len(batch_idx)
            ep_n += len(batch_idx)
        train_loss = ep_loss / max(1, ep_n)

        model.eval()
        with torch.no_grad():
            val_out = model(val_slot, val_mask, val_ctx)
            val_loss = float(loss_fn(val_out, val_y_slot, val_y_smask, val_y_skip))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        _log.info("epoch=%2d train_loss=%.5f val_loss=%.5f (best=%.5f)",
                  epoch, train_loss, val_loss, best_val)
    _log.info("fit took %.1fs; best val_loss=%.5f", time.time() - t0, best_val)

    # Restore best.
    if best_state is not None:
        model.load_state_dict(best_state)

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
        "data_provider": provider_name,
        "universe": universe,
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "embed_dim": args.embed_dim,
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
