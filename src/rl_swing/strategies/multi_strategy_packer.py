"""MultiStrategyPacker — v2 candidate aggregation without dedupe.

Where v1's ``StrategyAggregator`` collapses overlapping signals (same
symbol + date from multiple strategies) into a single best-strength
candidate, this packer keeps every strategy's candidate visible. The
v2 selector env steps once per ``(symbol, date)`` and presents the
agent with the *full slate* of strategy proposals so it can pick the
optimal one (or skip).

The pack is just an ordered tuple aligned with the canonical strategy
list — slots are zero-padded for strategies that didn't fire on this
(symbol, date), so the observation shape stays fixed.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from rl_swing.domain import CandidateTrade, FeatureFrame, PortfolioState
from rl_swing.ports import CandidateStrategy


@dataclass(frozen=True)
class StrategyPack:
    """All strategy proposals for one (symbol, as_of)."""
    symbol: str
    as_of: datetime
    # ``candidates[i]`` is the candidate from the ``i``-th strategy in
    # the packer's ``strategy_ids`` list, or ``None`` if that strategy
    # didn't fire on this (symbol, as_of).
    candidates: tuple[CandidateTrade | None, ...]

    @property
    def n_fired(self) -> int:
        return sum(1 for c in self.candidates if c is not None)


class MultiStrategyPacker:
    """Run a list of strategies and group their candidates into per-
    (symbol, date) ``StrategyPack`` objects, without deduplication.

    A pack is emitted only for (symbol, as_of) pairs where AT LEAST
    one strategy fired. This avoids polluting episodes with empty
    decisions.
    """

    def __init__(self, strategies: list[CandidateStrategy]) -> None:
        self.strategies = strategies
        self.strategy_ids: tuple[str, ...] = tuple(
            getattr(s, "strategy_id", f"strategy_{i}") for i, s in enumerate(strategies)
        )

    @property
    def n_slots(self) -> int:
        return len(self.strategies)

    def pack(
        self,
        features: Iterable[FeatureFrame],
        portfolio_state: PortfolioState,
    ) -> list[StrategyPack]:
        frames = list(features)
        # Build a slot table: (symbol, as_of) -> list[CandidateTrade | None]
        slot_table: dict[tuple[str, datetime], list[CandidateTrade | None]] = {}
        for i, strategy in enumerate(self.strategies):
            for cand in strategy.generate(frames, portfolio_state):
                key = (cand.symbol, cand.as_of)
                row = slot_table.setdefault(
                    key, [None] * len(self.strategies)
                )
                # Defensive: if a strategy fires twice on the same key
                # (shouldn't happen given our generators), keep the
                # higher signal_strength.
                existing = row[i]
                if existing is None or cand.signal_strength > existing.signal_strength:
                    row[i] = cand

        packs = [
            StrategyPack(
                symbol=key[0], as_of=key[1], candidates=tuple(row),
            )
            for key, row in slot_table.items()
        ]
        packs.sort(key=lambda p: (p.as_of, p.symbol))
        return packs
