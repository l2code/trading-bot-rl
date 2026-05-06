"""ActionMapper.

Maps the discrete action ``{0, 1, 2, 3}`` to the
``PolicyAction`` literal and a size multiplier. Single source of
truth referenced by both the env and policy scorers.
"""
from __future__ import annotations

from typing import cast

from rl_swing.domain import ACTION_TO_SIZE, PolicyAction

ACTION_INT_TO_LITERAL: dict[int, PolicyAction] = {
    0: cast(PolicyAction, "skip"),
    1: cast(PolicyAction, "take_25"),
    2: cast(PolicyAction, "take_50"),
    3: cast(PolicyAction, "take_100"),
}

LITERAL_TO_ACTION_INT: dict[PolicyAction, int] = {v: k for k, v in ACTION_INT_TO_LITERAL.items()}


def to_literal(action: int) -> PolicyAction:
    if action not in ACTION_INT_TO_LITERAL:
        raise ValueError(f"unknown action {action}")
    return ACTION_INT_TO_LITERAL[action]


def to_size_multiplier(action: int) -> float:
    return ACTION_TO_SIZE[to_literal(action)]
