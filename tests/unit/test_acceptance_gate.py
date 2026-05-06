"""Unit tests for the Phase-24-equivalent acceptance gate."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rl_swing.rl.validation.acceptance_gate import (
    DEFAULT_METRICS,
    evaluate_gate,
)

# A clearly worse baseline for clean improvement counting.
BASE = {
    "total_return": 1.0,
    "annualized_sharpe": 1.0,
    "profit_factor": 1.5,
    "max_drawdown": 0.30,
    "turnover_take_rate": 0.5,
}


def test_all_five_improved_yields_GO():
    cand = {
        "total_return": 1.5,
        "annualized_sharpe": 1.5,
        "profit_factor": 2.0,
        "max_drawdown": 0.20,            # lower is better
        "turnover_take_rate": 0.7,
    }
    res = evaluate_gate(cand, BASE)
    assert res.verdict == "GO"
    assert res.n_improved == 5
    assert res.n_regressed_materially == 0


def test_three_of_five_improved_yields_GO():
    cand = {
        "total_return": 1.5,             # +
        "annualized_sharpe": 1.5,        # +
        "profit_factor": 2.0,            # +
        "max_drawdown": 0.30,            # =
        "turnover_take_rate": 0.5,       # =
    }
    res = evaluate_gate(cand, BASE)
    assert res.verdict == "GO"
    assert res.n_improved == 3


def test_two_of_five_at_threshold_yields_GO():
    cand = {
        "total_return": 1.5,             # +
        "annualized_sharpe": 1.5,        # +
        "profit_factor": 1.5,            # =
        "max_drawdown": 0.30,            # =
        "turnover_take_rate": 0.5,       # =
    }
    res = evaluate_gate(cand, BASE)
    assert res.verdict == "GO"
    assert res.n_improved == 2


def test_one_of_five_improved_yields_SHADOW_ONLY():
    cand = {
        "total_return": 1.5,             # +
        "annualized_sharpe": 1.0,        # =
        "profit_factor": 1.5,            # =
        "max_drawdown": 0.30,            # =
        "turnover_take_rate": 0.5,       # =
    }
    res = evaluate_gate(cand, BASE)
    assert res.verdict == "SHADOW_ONLY"
    assert res.n_improved == 1


def test_zero_improved_yields_NO_GO():
    """The exact v1-loose-NO_GO situation: candidate is bit-identical
    to baseline, so n_improved=0."""
    cand = dict(BASE)
    res = evaluate_gate(cand, BASE)
    assert res.verdict == "NO_GO"
    assert res.n_improved == 0
    assert res.n_regressed_materially == 0


def test_material_regression_caps_at_NO_GO():
    """Even with 4 of 5 improved, a material regression on max_drawdown
    drops the verdict to NO_GO."""
    cand = {
        "total_return": 2.0,             # ++
        "annualized_sharpe": 1.5,        # +
        "profit_factor": 2.5,            # +
        "max_drawdown": 0.50,            # MATERIAL regression (+0.20pp > 0.05 threshold)
        "turnover_take_rate": 0.7,       # +
    }
    res = evaluate_gate(cand, BASE)
    assert res.verdict == "NO_GO"
    assert res.n_improved == 4
    assert res.n_regressed_materially == 1


def test_threshold_kwarg_overrides_default():
    """Using threshold=3 means the 2-of-5 case becomes SHADOW_ONLY/NO_GO."""
    cand = {
        "total_return": 1.5,
        "annualized_sharpe": 1.5,
        "profit_factor": 1.5,
        "max_drawdown": 0.30,
        "turnover_take_rate": 0.5,
    }
    res = evaluate_gate(cand, BASE, threshold_improved=3)
    # 2 improved, threshold 3, no material regressions. Falls through
    # the SHADOW_ONLY check (which is "improved == 1") to NO_GO.
    assert res.verdict == "NO_GO"
    assert res.n_improved == 2


def test_missing_metric_doesnt_crash():
    cand = {
        "total_return": 1.5,
        "annualized_sharpe": 1.5,
        # profit_factor missing
        "max_drawdown": 0.20,
        "turnover_take_rate": 0.7,
    }
    res = evaluate_gate(cand, BASE)
    # 4 of 5 improved (the missing one is treated as not-improved),
    # which is enough for GO.
    assert res.verdict == "GO"
    assert res.per_metric["profit_factor"]["missing"] is True


def test_lower_is_better_metric_direction():
    """max_drawdown is lower-is-better; ensure direction is honored."""
    spec = next(m for m in DEFAULT_METRICS if m.key == "max_drawdown")
    assert spec.higher_is_better is False
    # Candidate has LOWER drawdown -> improvement.
    cand = dict(BASE)
    cand["max_drawdown"] = 0.10
    res = evaluate_gate(cand, BASE)
    assert res.per_metric["max_drawdown"]["improved"] is True


def test_explanation_string_describes_the_decision():
    cand = dict(BASE)
    res = evaluate_gate(cand, BASE)
    assert "0 of 5" in res.explanation
    assert "improvement count below threshold" in res.explanation


def test_to_dict_round_trip():
    cand = dict(BASE)
    res = evaluate_gate(cand, BASE)
    d = res.to_dict()
    assert d["verdict"] == "NO_GO"
    assert "per_metric" in d and "total_return" in d["per_metric"]
