"""FeaturePipeline port."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from rl_swing.domain import FeatureFrame, MarketBar


@runtime_checkable
class FeaturePipeline(Protocol):
    feature_version: str
    feature_names: tuple[str, ...]

    def build(
        self,
        bars: Iterable[MarketBar],
        context: dict[str, Any] | None = None,
    ) -> Iterable[FeatureFrame]: ...
