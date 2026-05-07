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


# ---------------------------------------------------------------------
# FEAT-32 M2: behavioral-cloning target policy.
# A non-trivial state-dependent rule with action variance across
# states. Used as the imitation target for the BC env-learnability
# diagnostic (literal top1 is trivial — labels are constant).
@dataclass
class BCTargetPortfolioPolicy:
    """Hand-coded state-dependent target with three action regions:
      - action 2 when slate has ≥2 fired packs AND signal gap_top2
        is small (similar signals → spread the risk by taking both)
      - action 1 when slate has ≥1 fired pack AND cash > 0.5
      - action 0 otherwise

    Reads obs at the indices laid out in
    ``ChronologicalSwingEnv.OBS_DIM`` documentation:
      [n_fired, signal_max, signal_mean, signal_std, signal_gap_top2,
       all_fired_ind, cash_pct, gross_exp, n_open_norm, dd_pct,
       realized_pnl_pct, day_norm]
    """
    n_actions: int = 3
    gap_threshold: float = 0.10
    cash_threshold: float = 0.50
    model_id: str = "portfolio_target_bc"

    def decide(self, obs) -> int:
        if obs is None:
            return 0
        try:
            n_fired = float(obs[0])
            signal_gap_top2 = float(obs[4])
            cash_pct = float(obs[6])
        except (IndexError, TypeError):
            return 0
        if self.n_actions >= 3 and n_fired >= 2 and signal_gap_top2 < self.gap_threshold:
            return 2
        if n_fired >= 1 and cash_pct > self.cash_threshold:
            return 1
        return 0


# ---------------------------------------------------------------------
# FEAT-32 M2: BC inference policy. Wraps a trained sklearn classifier.
@dataclass
class BCPortfolioPolicy:
    """Inference-side wrapper around a trained behavioral-cloning
    classifier (sklearn HistGradientBoostingClassifier or any
    sklearn-compatible classifier with ``predict``).

    Lazy-loaded so the module is importable without sklearn installed
    (e.g. unit tests of unrelated baselines).
    """
    artifact_path: str
    n_actions: int
    model_id: str = "portfolio_baseline_bc"

    def __post_init__(self) -> None:
        self._model = None
        self._meta: dict = {}

    def _load(self):
        if self._model is not None:
            return self._model
        from pathlib import Path
        path = Path(self.artifact_path)
        if not path.exists():
            raise FileNotFoundError(
                f"BC artifact not found: {self.artifact_path}."
            )
        import joblib  # type: ignore[import-untyped]
        bundle = joblib.load(str(path))
        self._model = bundle["model"]
        self._meta = {k: v for k, v in bundle.items() if k != "model"}
        return self._model

    def decide(self, obs) -> int:
        if obs is None:
            return 0
        import numpy as np
        model = self._load()
        x = np.asarray(obs, dtype=np.float64).reshape(1, -1)
        try:
            pred = int(model.predict(x)[0])
        except Exception:
            return 0
        if pred < 0 or pred >= self.n_actions:
            return 0
        return pred
