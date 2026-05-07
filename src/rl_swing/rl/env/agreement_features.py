"""Cross-strategy agreement features for v002 selector / ranker (FEAT-7).

The post-PR-#71 finding: masked-PPO is bit-identical to
``selector_baseline_first_fired`` and the v0 supervised ranker fails
the Phase-24 gate. Both signal that the current slate doesn't carry
information beyond strategy priority order. This module adds
per-pack and per-slot agreement features so that the policy / ranker
can condition on:

  - how many strategies fired on this (symbol, date)
  - the signal-strength shape across fired strategies (max, mean,
    std, gap between the top two)
  - whether THIS slot is the top-signal slot, and its rank

Both the v002 environment's ``MultiStrategyObservationBuilder`` and
the supervised ranker's per-slot feature builder consume from this
module so train-time and inference-time observations are bit-
identical.

Pure functions; no env / sklearn / torch deps.
"""
from __future__ import annotations

from rl_swing.strategies.multi_strategy_packer import StrategyPack

# Pack-level agreement field names. Stable; the obs builder emits
# them in this exact order, and the supervised ranker reads them in
# the same order, so any feature added here must be appended at the
# end so existing trained artifacts remain loadable.
PACK_AGREEMENT_FIELDS: tuple[str, ...] = (
    "pack_n_fired",
    "pack_all_fired",
    "pack_signal_max",
    "pack_signal_mean",
    "pack_signal_std",
    "pack_signal_gap_top2",
    "pack_same_symbol_strategy_agreement",
)

# Per-slot agreement field names. Same stability rule.
PER_SLOT_AGREEMENT_FIELDS: tuple[str, ...] = (
    "slot_is_top_signal",
    "slot_rank_by_signal",
)


def compute_pack_agreement(pack: StrategyPack, n_strategies: int) -> dict[str, float]:
    """Pack-level agreement features.

    All fields are floats so the obs vector stays homogeneous. Fields
    that are conceptually booleans (``pack_all_fired``) emit 0.0 / 1.0.

    For empty packs (zero fired) the moments are 0.0; this is benign
    because the obs builder zero-pads the per-slot block too.
    """
    fired = [c for c in pack.candidates if c is not None]
    n_fired = len(fired)
    n_total = max(int(n_strategies), 1)

    if n_fired == 0:
        return {
            "pack_n_fired": 0.0,
            "pack_all_fired": 0.0,
            "pack_signal_max": 0.0,
            "pack_signal_mean": 0.0,
            "pack_signal_std": 0.0,
            "pack_signal_gap_top2": 0.0,
            "pack_same_symbol_strategy_agreement": 0.0,
        }

    signals = sorted((float(c.signal_strength) for c in fired), reverse=True)
    s_max = signals[0]
    s_mean = sum(signals) / n_fired
    if n_fired >= 2:
        # population std (N denominator). Cheap and stable for n_fired
        # in {1..N}; we don't need sample std here.
        var = sum((s - s_mean) ** 2 for s in signals) / n_fired
        s_std = var ** 0.5
        s_gap = signals[0] - signals[1]
    else:
        s_std = 0.0
        s_gap = 0.0

    return {
        "pack_n_fired": float(n_fired),
        "pack_all_fired": 1.0 if n_fired == n_total else 0.0,
        "pack_signal_max": s_max,
        "pack_signal_mean": s_mean,
        "pack_signal_std": s_std,
        "pack_signal_gap_top2": s_gap,
        # Per pack, every fired slot is on the same symbol+date, so
        # n_fired IS the same-symbol agreement count. Kept as a
        # separate name for forward-compat with multi-day rollups.
        "pack_same_symbol_strategy_agreement": float(n_fired),
    }


def compute_slot_agreement(pack: StrategyPack, slot_idx: int) -> dict[str, float]:
    """Per-slot agreement features.

    Tie-break for ``is_top_signal`` and ``rank_by_signal``: when two
    fired slots share the same signal_strength, the lower slot_idx
    wins. This keeps the features deterministic and aligned with
    ``selector_baseline_first_fired`` semantics.

    Returns 0.0 for both fields when the slot is unfired — the
    builder zero-pads unfired slots, so this matches the existing
    convention.
    """
    if slot_idx < 0 or slot_idx >= len(pack.candidates):
        return {"slot_is_top_signal": 0.0, "slot_rank_by_signal": 0.0}
    me = pack.candidates[slot_idx]
    if me is None:
        return {"slot_is_top_signal": 0.0, "slot_rank_by_signal": 0.0}

    # Sort fired slots by (-signal_strength, slot_idx). Lower-idx wins
    # ties so the top of the rank is well-defined and matches
    # first_fired tie-break.
    fired = [
        (k, float(c.signal_strength))
        for k, c in enumerate(pack.candidates)
        if c is not None
    ]
    fired.sort(key=lambda kv: (-kv[1], kv[0]))
    rank = next(i for i, (k, _) in enumerate(fired) if k == slot_idx)
    return {
        "slot_is_top_signal": 1.0 if rank == 0 else 0.0,
        "slot_rank_by_signal": float(rank),
    }


def pack_agreement_vector(pack: StrategyPack, n_strategies: int) -> tuple[float, ...]:
    """Return the pack-level agreement features in
    ``PACK_AGREEMENT_FIELDS`` order. Cached-friendly for the obs
    builder hot path (no dict allocation per step)."""
    d = compute_pack_agreement(pack, n_strategies)
    return tuple(d[name] for name in PACK_AGREEMENT_FIELDS)


def slot_agreement_vector(pack: StrategyPack, slot_idx: int) -> tuple[float, ...]:
    """Return the per-slot agreement features in
    ``PER_SLOT_AGREEMENT_FIELDS`` order."""
    d = compute_slot_agreement(pack, slot_idx)
    return tuple(d[name] for name in PER_SLOT_AGREEMENT_FIELDS)
