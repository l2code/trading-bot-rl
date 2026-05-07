"""Per-day portfolio baselines for v3 chronological env (FEAT-32 M1).

Each baseline maps observation → action index in the env's
``Discrete(1 + max_top_k)`` action lattice. They're *policy*-style
deciders, not selector scorers, since v3 steps per trading day rather
than per pack.

Four baselines per the M1 plan:

  - ``portfolio_baseline_no_op``       — always action 0 (don't open)
  - ``portfolio_baseline_top1``        — always action 1 (take top-1)
  - ``portfolio_baseline_top2``        — always action 2 (take top-2)
  - ``portfolio_baseline_random_action`` — uniform over action lattice

These are deliberately stateless and observation-agnostic. The
ChronologicalSwingEnv enforces budget rules (cash, gross exposure)
inside ``open_position``, so e.g. ``portfolio_baseline_top2`` never
exceeds the 100% gross-exposure cap — the tracker silently refuses
when budget runs out.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class NoOpPortfolioPolicy:
    model_id: str = "portfolio_baseline_no_op"

    def decide(self, obs) -> int:
        return 0


@dataclass
class TopKPortfolioPolicy:
    """Always picks the same fixed action k. Used for the
    ``portfolio_baseline_top1`` and ``portfolio_baseline_top2``
    baselines. Both are degenerate strategies designed to surface
    'how good is naive aggression' on the v3 framing."""
    k: int
    model_id: str = ""

    def __post_init__(self) -> None:
        if not self.model_id:
            self.model_id = f"portfolio_baseline_top{self.k}"

    def decide(self, obs) -> int:
        return int(self.k)


@dataclass
class RandomActionPortfolioPolicy:
    """Uniform random over the action lattice. The v3 counterpart of
    v002's ``selector_baseline_random``. Phase-24 gate verdict is
    computed against this baseline."""
    n_actions: int
    seed: int = 42
    model_id: str = "portfolio_baseline_random_action"

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def decide(self, obs) -> int:
        return self._rng.randint(0, self.n_actions - 1)
