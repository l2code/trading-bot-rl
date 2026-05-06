"""Feature-frame domain types.

A ``FeatureFrame`` represents the model-ready feature view for one
``(symbol, as_of)`` pair at one ``feature_version``.

Leakage rule (enforced by ``rl_swing.features.leakage_checks``):
    A FeatureFrame may only contain information available **as of**
    ``as_of`` — no future bars, no future-known event labels, no
    forward-filled macro values that weren't published yet.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class FeatureFrame:
    as_of: datetime
    symbol: str
    feature_version: str
    values: Mapping[str, float]
    feature_names: tuple[str, ...]
    source_snapshot_id: str

    def __post_init__(self) -> None:  # pragma: no cover - cheap invariant
        # `values` is required to cover the declared `feature_names`.
        missing = set(self.feature_names) - set(self.values.keys())
        if missing:
            raise ValueError(
                f"FeatureFrame for {self.symbol}@{self.as_of.isoformat()} "
                f"missing values for declared features: {sorted(missing)}"
            )

    def vector(self) -> list[float]:
        """Return a list of feature values in declared name order.

        Always use this method (never iterate ``values`` directly) so
        observation order is stable across training and inference.
        """
        return [float(self.values[name]) for name in self.feature_names]


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    """Identifier for a built batch of feature frames.

    Stored alongside ``MarketSnapshot`` so a model artifact can declare:
    'I was trained on market_snapshot=X and feature_snapshot=Y'.
    """

    snapshot_id: str
    feature_version: str
    market_snapshot_id: str
    metadata: dict = field(default_factory=dict)
