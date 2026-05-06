"""Runtime modes.

A runtime mode is a profile that selects a broker adapter, risk
profile, and a couple of safety flags. The decision pipeline is the
**same** in every mode — only these knobs change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RuntimeMode = Literal["research", "shadow", "paper", "live_guarded"]


@dataclass(frozen=True, slots=True)
class ModeProfile:
    mode: RuntimeMode
    place_orders: bool
    allow_live_trading: bool
    require_reconciliation_before_new_orders: bool
    description: str


MODE_PROFILES: dict[RuntimeMode, ModeProfile] = {
    "research": ModeProfile(
        mode="research",
        place_orders=False,
        allow_live_trading=False,
        require_reconciliation_before_new_orders=False,
        description="Historical replay. Simulated broker. Used for training/validation/backtest.",
    ),
    "shadow": ModeProfile(
        mode="shadow",
        place_orders=False,
        allow_live_trading=False,
        require_reconciliation_before_new_orders=False,
        description="Daily signal generation against latest data. NoOp broker. No orders.",
    ),
    "paper": ModeProfile(
        mode="paper",
        place_orders=True,
        allow_live_trading=False,
        require_reconciliation_before_new_orders=True,
        description="Alpaca paper account. Fake money but real execution flow.",
    ),
    "live_guarded": ModeProfile(
        mode="live_guarded",
        place_orders=False,                    # default off; second flag must flip
        allow_live_trading=False,              # second gate
        require_reconciliation_before_new_orders=True,
        description=(
            "Tightly capped live experiment. Default OFF. Requires manual approval token "
            "AND ``allow_live_trading=true`` AND ``place_orders=true`` to actually trade."
        ),
    ),
}


def get_mode_profile(mode: RuntimeMode) -> ModeProfile:
    if mode not in MODE_PROFILES:
        raise ValueError(f"Unknown runtime mode: {mode!r}. "
                         f"Known: {list(MODE_PROFILES)}")
    return MODE_PROFILES[mode]
