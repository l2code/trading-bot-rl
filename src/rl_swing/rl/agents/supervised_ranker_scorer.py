"""SupervisedRankerSelectorScorer — supervised contextual-bandit
baseline for v002 / v002_masked (FEAT-30).

Where the v002 PPO learns to pick a strategy slot from full-batch
rewards, this scorer skips the RL machinery: at training time it
simulates every fired (pack × strategy) on the train window, records
the realized risk-adjusted return as a regression target, and fits a
gradient-boosted tree. At inference time it predicts the per-slot
risk_adj_return and picks the argmax — or skip if the max prediction
is below zero (no fired strategy has predicted positive EV).

The point of this baseline is the diagnostic from the operator
roadmap (CLAUDE.md §4 / RFC #30): if a simple ranker beats masked-PPO
on the Phase-24 gate, RL machinery isn't earning its complexity yet.

Lazy-loaded: importing this module doesn't require sklearn/joblib
in environments that only run unit tests against the baselines.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rl_swing.domain import FeatureFrame, PortfolioState
from rl_swing.features.pipelines import ALL_FEATURE_NAMES
from rl_swing.rl.env.agreement_features import (
    PACK_AGREEMENT_FIELDS,
    PER_SLOT_AGREEMENT_FIELDS,
    pack_agreement_vector,
    slot_agreement_vector,
)
from rl_swing.strategies.multi_strategy_packer import StrategyPack

_log = logging.getLogger(__name__)


# Per-slot feature columns. Stable; the trainer must emit features in
# this exact order so train and inference agree to the bit. FEAT-7
# adds the pack-level + per-slot agreement fields between the
# original frame features and the original per-slot fields.
PER_SLOT_FEATURE_NAMES: tuple[str, ...] = (
    *ALL_FEATURE_NAMES,
    *PACK_AGREEMENT_FIELDS,           # FEAT-7: pack-level agreement
    "slot_idx",                       # 0..N-1
    "slot_signal_strength",           # candidate.signal_strength
    "slot_base_size_pct",
    "slot_max_holding_days_norm",
    *PER_SLOT_AGREEMENT_FIELDS,       # FEAT-7: per-slot is_top_signal, rank
)


def build_slot_features(
    frame: FeatureFrame,
    slot_idx: int,
    candidate,  # CandidateTrade
    *,
    pack: StrategyPack | None = None,
    n_strategies: int | None = None,
) -> np.ndarray:
    """Build a single per-(pack, slot) feature vector. Must match the
    column order in PER_SLOT_FEATURE_NAMES.

    FEAT-7: ``pack`` and ``n_strategies`` are optional for backward-
    compat (callers that don't have a pack get zero-filled agreement
    features). Real callers (the trainer + the inference scorer)
    always pass them.
    """
    base = [frame.values.get(name, 0.0) for name in ALL_FEATURE_NAMES]
    if pack is not None and n_strategies is not None:
        base.extend(pack_agreement_vector(pack, n_strategies))
    else:
        base.extend([0.0] * len(PACK_AGREEMENT_FIELDS))
    base.append(float(slot_idx))
    base.append(float(candidate.signal_strength))
    base.append(float(candidate.base_size_pct))
    base.append(float(candidate.max_holding_days) / 30.0)
    if pack is not None:
        base.extend(slot_agreement_vector(pack, slot_idx))
    else:
        base.extend([0.0] * len(PER_SLOT_AGREEMENT_FIELDS))
    return np.asarray(base, dtype=np.float64)


@dataclass
class SupervisedRankerSelectorScorer:
    """Inference-side wrapper around a fitted regressor saved by
    ``scripts/train_supervised_ranker.py``.

    Artifact layout (joblib):
      {
        "model": <sklearn regressor>,
        "feature_names": tuple[str, ...],
        "n_strategies": int,
        "target_risk_pct": float,
        "trained_at": isoformat str,
        "n_train_examples": int,
      }
    """
    artifact_path: str
    n_strategies: int
    model_id: str = "selector_baseline_supervised"
    feature_version: str = "features_v001_core_daily"
    skip_threshold: float = 0.0  # predicted risk_adj < this -> skip

    def __post_init__(self) -> None:
        self._model = None
        self._meta: dict = {}
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return self._model
        path = Path(self.artifact_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Supervised ranker artifact not found: {self.artifact_path}."
            )
        import joblib  # type: ignore[import-untyped]
        bundle = joblib.load(str(path))
        self._model = bundle["model"]
        self._meta = {k: v for k, v in bundle.items() if k != "model"}
        return self._model

    def select(
        self,
        pack: StrategyPack,
        feature: FeatureFrame,
        portfolio_state: PortfolioState,
    ) -> int:
        # Short-circuit empty packs BEFORE loading the model — saves
        # an artifact-load + lets the scorer behave correctly even
        # when no model is on disk (e.g. in tests, or when running
        # validate before the offline ranker has been trained).
        rows = []
        slot_idxs = []
        for k, c in enumerate(pack.candidates):
            if c is None:
                continue
            rows.append((k, c))
            slot_idxs.append(k)
        if not rows:
            return 0  # nothing fired -> skip
        if feature.feature_version != self.feature_version:
            raise RuntimeError(
                f"Feature version mismatch: ranker trained on "
                f"{self.feature_version!r}, frame is "
                f"{feature.feature_version!r}."
            )
        with self._lock:
            model = self._load()
        # FEAT-7: pass pack + n_strategies so the per-slot rows include
        # the pack-level + per-slot agreement features the trainer
        # emitted. Without this, inference-time vectors would be
        # zero-padded in the agreement columns, silently shrinking the
        # effective feature set vs train.
        X = np.vstack([
            build_slot_features(
                feature, k, c, pack=pack, n_strategies=self.n_strategies,
            )
            for k, c in rows
        ])
        preds = np.asarray(model.predict(X), dtype=np.float64)
        best = int(np.argmax(preds))
        if preds[best] < self.skip_threshold:
            return 0
        # action = 1 + slot_idx of the chosen fired slot
        return 1 + slot_idxs[best]


def write_metadata(path: Path, meta: dict) -> None:
    """Side-by-side metadata.json. Useful for humans + smoke tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
