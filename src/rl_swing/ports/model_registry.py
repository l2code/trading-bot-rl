"""ModelRegistry port.

The registry is the single source of truth for which models exist and
which are approved at which stage. No model with status != ``LIVE_APPROVED``
may be used by a live broker adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

ApprovalStatus = Literal[
    "TRAINED",
    "VALIDATED",
    "SHADOW_APPROVED",
    "PAPER_APPROVED",
    "LIVE_CANDIDATE",
    "LIVE_APPROVED",
    "REJECTED",
    "RETIRED",
]


@dataclass(frozen=True, slots=True)
class ModelEntry:
    model_id: str
    algorithm: str
    artifact_path: str
    feature_version: str
    universe_version: str
    reward_config_version: str
    training_start: datetime
    training_end: datetime
    validation_start: datetime
    validation_end: datetime
    test_start: datetime | None
    test_end: datetime | None
    hyperparameters: dict
    metrics: dict
    approval_status: ApprovalStatus = "TRAINED"
    approved_by: str | None = None
    approved_at: datetime | None = None
    code_commit: str | None = None
    seed: int | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    notes: str = ""


@runtime_checkable
class ModelRegistry(Protocol):
    def register(self, entry: ModelEntry) -> None: ...
    def get(self, model_id: str) -> ModelEntry | None: ...
    def list_by_status(self, status: ApprovalStatus) -> list[ModelEntry]: ...
    def transition(
        self,
        model_id: str,
        new_status: ApprovalStatus,
        approved_by: str,
        notes: str = "",
    ) -> ModelEntry: ...
