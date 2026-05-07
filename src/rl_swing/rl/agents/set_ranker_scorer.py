"""SetRankerSelectorScorer — supervised set/slate ranker (FEAT-34 PR-1).

Wraps a trained ``SlateEncoder`` for the SelectorScorer Protocol.
At inference time builds the full slate input (per-slot features +
slot mask + pack-level context) and asks the encoder for per-slot
logits + skip logit. Picks argmax over [skip, slot_0, ..., slot_{N-1}],
respecting the fired mask (an unfired slot's logit is replaced with
-inf before argmax).

This is the cheap supervised diagnostic for the slate-encoder
inductive bias. PR-2 (gated on this scorer materially improving
over the FEAT-7 HistGB ranker) will reuse the same encoder as a
sb3 features extractor inside MaskablePPO.
"""
from __future__ import annotations

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


# Stable feature column orders used by both the offline trainer and
# this scorer. The two MUST agree to the bit.
SLOT_FEATURE_NAMES: tuple[str, ...] = (
    "slot_signal_strength",
    "slot_base_size_pct",
    "slot_max_holding_days_norm",
    *PER_SLOT_AGREEMENT_FIELDS,         # is_top_signal, rank_by_signal
)
CTX_FEATURE_NAMES: tuple[str, ...] = (
    *ALL_FEATURE_NAMES,
    *PACK_AGREEMENT_FIELDS,
)


def build_slot_feature_row(pack: StrategyPack, slot_idx: int) -> np.ndarray:
    """Per-slot feature vector for the SlateEncoder (NO slot_idx column;
    the whole point is permutation invariance, so we don't tell the
    model 'this is slot 0'). Zeros for unfired slots; that combined
    with the explicit slot_mask avoids the encoder leaking unfired
    slot data into the aggregate."""
    c = pack.candidates[slot_idx] if 0 <= slot_idx < len(pack.candidates) else None
    if c is None:
        return np.zeros(len(SLOT_FEATURE_NAMES), dtype=np.float32)
    slot_agree = slot_agreement_vector(pack, slot_idx)
    return np.asarray([
        float(c.signal_strength),
        float(c.base_size_pct),
        float(c.max_holding_days) / 30.0,
        *slot_agree,
    ], dtype=np.float32)


def build_ctx_features(pack: StrategyPack, frame: FeatureFrame, n_strategies: int) -> np.ndarray:
    """Pack-level context: frame features + pack-level agreement."""
    base = [frame.values.get(name, 0.0) for name in ALL_FEATURE_NAMES]
    base.extend(pack_agreement_vector(pack, n_strategies))
    return np.asarray(base, dtype=np.float32)


def build_slot_mask(pack: StrategyPack, n_strategies: int) -> np.ndarray:
    return np.asarray(
        [
            1 if (i < len(pack.candidates) and pack.candidates[i] is not None) else 0
            for i in range(n_strategies)
        ],
        dtype=np.float32,
    )


@dataclass
class SetRankerSelectorScorer:
    """Inference-side wrapper around a trained SlateEncoder.

    Artifact layout (torch.save):
      {
        "state_dict": <encoder state dict>,
        "config": SlateEncoderConfig,
        "slot_feature_names": SLOT_FEATURE_NAMES,
        "ctx_feature_names": CTX_FEATURE_NAMES,
        "n_strategies": int,
        "target_risk_pct": float,
        "trained_at": str,
        "n_train_examples": int,
        ...
      }
    """
    artifact_path: str
    n_strategies: int
    model_id: str = "selector_baseline_set_ranker"
    feature_version: str = "features_v001_core_daily"
    skip_threshold: float = 0.0
    """If max(slot_logits at fired slots) < skip_logit + skip_threshold,
    return skip. Default 0 means skip iff skip_logit dominates."""

    def __post_init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return self._model
        path = Path(self.artifact_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Set ranker artifact not found: {self.artifact_path}."
            )
        import torch  # type: ignore[import-untyped]

        from rl_swing.rl.agents.slate_encoder import SlateEncoder
        bundle = torch.load(str(path), map_location="cpu", weights_only=False)
        encoder = SlateEncoder(bundle["config"])
        encoder.load_state_dict(bundle["state_dict"])
        encoder.eval()
        self._model = encoder
        # PR-1b: feature normalization stats. Old (PR-1) artifacts
        # don't have these — fall back to identity.
        self._ctx_mean = bundle.get("ctx_mean")
        self._ctx_std = bundle.get("ctx_std")
        self._slot_mean = bundle.get("slot_mean")
        self._slot_std = bundle.get("slot_std")
        return self._model

    def select(
        self,
        pack: StrategyPack,
        feature: FeatureFrame,
        portfolio_state: PortfolioState,
    ) -> int:
        # Empty-pack short-circuit before model load (mirror of
        # SupervisedRankerSelectorScorer).
        if not any(c is not None for c in pack.candidates):
            return 0
        if feature.feature_version != self.feature_version:
            raise RuntimeError(
                f"Feature version mismatch: set ranker trained on "
                f"{self.feature_version!r}, frame is "
                f"{feature.feature_version!r}."
            )
        import torch
        # Build inputs: (1, N, slot_feat_dim), (1, N), (1, ctx_dim).
        slot_rows = np.stack(
            [build_slot_feature_row(pack, k) for k in range(self.n_strategies)],
            axis=0,
        )
        slot_mask = build_slot_mask(pack, self.n_strategies)
        ctx = build_ctx_features(pack, feature, self.n_strategies)

        with self._lock:
            model = self._load()
        # PR-1b: apply train-time standardization. Skip if the bundle
        # predates standardization (PR-1 artifact), in which case the
        # stats are None and we feed raw features.
        if self._slot_mean is not None and self._slot_std is not None:
            slot_rows = (slot_rows - self._slot_mean) / self._slot_std
        if self._ctx_mean is not None and self._ctx_std is not None:
            ctx = (ctx - self._ctx_mean) / self._ctx_std

        slot_t = torch.from_numpy(slot_rows.astype(np.float32)).unsqueeze(0)  # (1, N, F)
        mask_t = torch.from_numpy(slot_mask).unsqueeze(0)         # (1, N)
        ctx_t = torch.from_numpy(ctx.astype(np.float32)).unsqueeze(0)         # (1, ctx_dim)

        with torch.no_grad():
            out = model(slot_t, mask_t, ctx_t)

        slot_logits = out["slot_logits"][0].numpy()  # (N,)
        skip_logit = float(out["skip_logit"][0, 0])

        # Mask unfired slots before argmax so the policy literally
        # cannot pick them at inference (mirror of MaskablePPO mask).
        very_neg = -1e30
        masked_slot = np.where(slot_mask > 0, slot_logits, very_neg)
        best_slot = int(np.argmax(masked_slot))
        best_slot_logit = float(masked_slot[best_slot])

        if best_slot_logit + self.skip_threshold < skip_logit:
            return 0
        return 1 + best_slot
