"""Episode samplers.

An episode in the trade-filter env is a sequence of candidate trades.
Two samplers cover the spec's needs:

* ``RandomWindowSampler`` for training — randomized rolling windows
  picked with a controlled RNG.
* ``ChronologicalSampler`` for validation/test — strict time order
  through a fixed window.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Sequence

from rl_swing.domain import CandidateTrade


@dataclass
class Episode:
    candidates: Sequence[CandidateTrade]
    start: date
    end: date


class _Base:
    def __init__(self, candidates: Sequence[CandidateTrade]) -> None:
        self.candidates = sorted(candidates, key=lambda c: (c.as_of, c.symbol))


class RandomWindowSampler(_Base):
    def __init__(
        self,
        candidates: Sequence[CandidateTrade],
        window_days: int = 60,
        min_candidates: int = 5,
        seed: int = 0,
    ) -> None:
        super().__init__(candidates)
        self.window_days = int(window_days)
        self.min_candidates = int(min_candidates)
        self._rng = random.Random(seed)

    def sample(self) -> Episode:
        if not self.candidates:
            return Episode(candidates=[], start=date.today(), end=date.today())
        first = self.candidates[0].as_of.date()
        last = self.candidates[-1].as_of.date()
        if (last - first).days <= self.window_days:
            return Episode(candidates=self.candidates, start=first, end=last)
        for _ in range(20):
            offset_days = self._rng.randint(0, max(0, (last - first).days - self.window_days))
            start = first + timedelta(days=offset_days)
            end = start + timedelta(days=self.window_days)
            window = [
                c for c in self.candidates
                if start <= c.as_of.date() <= end
            ]
            if len(window) >= self.min_candidates:
                return Episode(candidates=window, start=start, end=end)
        # fallback: return everything
        return Episode(candidates=self.candidates, start=first, end=last)


class ChronologicalSampler(_Base):
    def __init__(self, candidates: Sequence[CandidateTrade]) -> None:
        super().__init__(candidates)
        self._exhausted = False

    def sample(self) -> Episode:
        if self._exhausted:
            return Episode(candidates=[], start=date.today(), end=date.today())
        self._exhausted = True
        if not self.candidates:
            return Episode(candidates=[], start=date.today(), end=date.today())
        return Episode(
            candidates=self.candidates,
            start=self.candidates[0].as_of.date(),
            end=self.candidates[-1].as_of.date(),
        )

    def reset(self) -> None:
        self._exhausted = False
