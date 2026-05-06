"""Phase-24-equivalent acceptance gate: ≥2-of-5 metric improvement.

Distilled from the trading-bot2 SDLC lessons (§4.2). Single-metric
gates are gameable (PF goes up but DD doubles? technically passes a
PF gate). A five-axis vector with a 'majority improved' threshold is
robust to that.

Five canonical metrics:
    total_return        — higher is better
    annualized_sharpe   — higher is better
    profit_factor       — higher is better
    max_drawdown        — LOWER is better
    turnover_take_rate  — informational; closer to a target band is
                          better, default rule is "neither extreme"

Verdict ladder:
    GO          — ≥ threshold_improved metrics strictly improved AND
                  no metric materially regressed.
    SHADOW_ONLY — exactly one metric improved with no material
                  regression. Insufficient for production but worth
                  observation.
    NO_GO       — anything else.

The gate operates on a single-cycle metric dict. Multi-cycle
aggregation (mean / median / worst-case) is the caller's
responsibility — see issue #5 for the multi-cycle walk-forward.

Threshold defaults to 2 (per trading-bot2 calibration) but is a
kwarg so we can A/B against tighter (3-of-5) once we have evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Verdict = Literal["GO", "SHADOW_ONLY", "NO_GO"]


# ---------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MetricSpec:
    """How to interpret one metric: which direction is improvement,
    and what counts as a 'material' regression that vetos GO."""
    key: str
    higher_is_better: bool
    # If the candidate is worse than baseline by more than this
    # (absolute delta), it counts as a material regression even if
    # other metrics improve. Material regressions cap the verdict
    # at NO_GO regardless of the improved-count.
    material_regression_threshold: float = 0.0


# Canonical Phase-24 metric set. Add new metrics by appending here
# and updating the WeightConfig in metrics.py if the composite score
# needs to weight them.
DEFAULT_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("total_return", higher_is_better=True,
               material_regression_threshold=0.05),    # >5% total-return loss is material
    MetricSpec("annualized_sharpe", higher_is_better=True,
               material_regression_threshold=0.5),     # >0.5 sharpe loss is material
    MetricSpec("profit_factor", higher_is_better=True,
               material_regression_threshold=0.3),     # >0.3 PF loss is material
    MetricSpec("max_drawdown", higher_is_better=False,
               material_regression_threshold=0.05),    # >5pp drawdown increase is material
    # NOTE: turnover_take_rate's "right direction" is genuinely
    # debatable — for a *filter* variant lower take-rate (more
    # selectivity) is often the point; for a *selector* it's near-
    # neutral. We mark it higher_is_better=True to match the current
    # composite-score convention, with a deliberately loose material-
    # regression threshold so it can't gate by itself. Tracked: a
    # follow-up RFC will pick a robust 5th metric (win_rate or
    # cost_drag_bps) without this ambiguity.
    MetricSpec("turnover_take_rate", higher_is_better=True,
               material_regression_threshold=0.5),     # informational; near-impossible to materially regress
)


@dataclass(frozen=True, slots=True)
class GateResult:
    verdict: Verdict
    n_improved: int
    n_regressed_materially: int
    per_metric: dict[str, dict[str, float | bool]]
    threshold_improved: int
    explanation: str

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "n_improved": self.n_improved,
            "n_regressed_materially": self.n_regressed_materially,
            "threshold_improved": self.threshold_improved,
            "per_metric": dict(self.per_metric),
            "explanation": self.explanation,
        }


# ---------------------------------------------------------------------
def evaluate_gate(
    candidate: dict,
    baseline: dict,
    *,
    threshold_improved: int = 2,
    metrics: tuple[MetricSpec, ...] = DEFAULT_METRICS,
) -> GateResult:
    """Decide whether ``candidate`` beats ``baseline`` per the
    Phase-24-equivalent gate.

    Both ``candidate`` and ``baseline`` are dicts keyed by metric
    name (e.g., the per-policy result rows produced by walk-forward).

    Verdict rules:
        GO          — n_improved ≥ threshold_improved AND
                      n_regressed_materially == 0
        SHADOW_ONLY — n_improved == 1 AND n_regressed_materially == 0
        NO_GO       — anything else (including any material regression)
    """
    per_metric: dict[str, dict[str, float | bool]] = {}
    n_improved = 0
    n_regressed_materially = 0

    for spec in metrics:
        cv = candidate.get(spec.key)
        bv = baseline.get(spec.key)
        if cv is None or bv is None:
            per_metric[spec.key] = {
                "candidate": cv,
                "baseline": bv,
                "delta": None,
                "improved": False,
                "regressed_materially": False,
                "missing": True,
            }
            continue
        delta = float(cv) - float(bv)
        # Direction: for higher-is-better, improvement is delta > 0.
        # For lower-is-better, improvement is delta < 0 (which is
        # effectively "candidate's value is below baseline's").
        improved = (delta > 0) if spec.higher_is_better else (delta < 0)
        # Material regression: delta in the wrong direction past the
        # threshold.
        regressed_materially = (
            (-delta > spec.material_regression_threshold)
            if spec.higher_is_better
            else (delta > spec.material_regression_threshold)
        )
        per_metric[spec.key] = {
            "candidate": float(cv),
            "baseline": float(bv),
            "delta": float(delta),
            "improved": bool(improved),
            "regressed_materially": bool(regressed_materially),
            "missing": False,
        }
        if improved:
            n_improved += 1
        if regressed_materially:
            n_regressed_materially += 1

    if n_regressed_materially > 0:
        verdict: Verdict = "NO_GO"
        explanation = (
            f"Material regression on {n_regressed_materially} metric(s). "
            f"Improved={n_improved} but caps verdict at NO_GO."
        )
    elif n_improved >= threshold_improved:
        verdict = "GO"
        explanation = (
            f"{n_improved} of {len(metrics)} metrics improved "
            f"(≥{threshold_improved} required); no material regressions."
        )
    elif n_improved == 1:
        verdict = "SHADOW_ONLY"
        explanation = (
            f"1 of {len(metrics)} metrics improved (need "
            f"{threshold_improved}); no material regressions. "
            f"Worth shadow observation but not GO."
        )
    else:
        verdict = "NO_GO"
        explanation = (
            f"{n_improved} of {len(metrics)} metrics improved "
            f"(need {threshold_improved}); no material regressions, "
            f"but improvement count below threshold."
        )

    return GateResult(
        verdict=verdict,
        n_improved=n_improved,
        n_regressed_materially=n_regressed_materially,
        per_metric=per_metric,
        threshold_improved=threshold_improved,
        explanation=explanation,
    )
