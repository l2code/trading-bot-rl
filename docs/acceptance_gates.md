# Acceptance gates

Distilled from `SDLC_LESSONS_FOR_NEW_PROJECT.md` §4.2 (Phase-24
gate). Implemented in
`src/rl_swing/rl/validation/acceptance_gate.py`.

## The Phase-24-equivalent gate: ≥2 of 5

A candidate strategy / model / variant must improve at least **2 of
5** canonical metrics over the strongest baseline, with **no
material regression on any metric**, to earn a GO verdict.

### The five metrics

| Metric | Direction | Material regression threshold |
|--------|-----------|-------------------------------|
| `total_return`         | higher better | 0.05 (5pp loss) |
| `annualized_sharpe`    | higher better | 0.5 sharpe loss |
| `profit_factor`        | higher better | 0.3 PF loss |
| `max_drawdown`         | **lower** better | 0.05 (5pp drawdown increase) |
| `turnover_take_rate`   | higher better | 0.2 |

### Verdict ladder

```
                   improvements
                  0    1    2    3    4    5
                ┌──────────────────────────────┐
no material reg │NO_GO|SHAD |GO  |GO  |GO  |GO │
                │     |OW   │    │    │    │   │
                │     |ONLY │    │    │    │   │
                ├──────────────────────────────┤
material reg ≥1 │NO_GO|NO_GO|NO_GO|NO_GO|NO_GO|NO_GO│
                └──────────────────────────────┘
```

Material regressions are absolute deltas, not relative — a 5pp
drawdown increase is material whether the baseline DD was 10% or
30%.

## Why 2 of 5

Per the trading-bot2 calibration:
- 1 of 5 is too loose; nearly anything passes.
- 3 of 5 is too tight; nearly nothing passes.
- 2 of 5 is empirically the sweet spot.

The threshold is a kwarg (`threshold_improved`) so we can A/B
against tighter rules once we have more data.

## Why 5, not 1

Single-metric gates can be gamed:
- "Profit factor doubled!" (and so did max drawdown — a wash on
  any risk-adjusted basis).
- "Sharpe up 30%!" (with 70% of the trades cut, so the test sample
  is too small to trust).

A five-axis vector with a "majority improved" rule is robust to
any single metric being misleading.

## Usage

```python
from rl_swing.rl.validation.acceptance_gate import evaluate_gate

result = evaluate_gate(
    candidate=trained_model_metrics,    # dict
    baseline=baseline_always_take_metrics,  # dict
)
print(result.verdict)            # GO / SHADOW_ONLY / NO_GO
print(result.n_improved)         # int
print(result.explanation)        # human-readable
print(result.per_metric)         # per-metric improvement breakdown
```

## When to apply

- **Single-cycle WF result:** apply the gate per cycle. If 2 of 5
  cycles are GO and 3 are NO_GO, the aggregate is NO_GO. Robust to
  single-year flukes.
- **Multi-cycle WF result (planned, #5):** apply the gate to the
  AGGREGATE metrics (mean across cycles), not the per-cycle ones.
  This is what gates promotion to canonical / shadow / live.
- **Exploratory tier:** the gate output is informational only. A
  GO from the gate on yfinance is at most SHADOW_ONLY in our
  three-tier ladder (see `data_tiers.md`).

## Anti-patterns

- **Re-running until you pass.** Picking the seed where 2 of 5
  improved out of 10 seeds is p-hacking. Aggregate across all seeds
  before applying the gate.
- **Re-defining the metrics post-hoc.** The 5 are fixed in the
  module's `DEFAULT_METRICS` constant. If you want to change them,
  it's an RFC.
- **Ignoring material regressions because "the upside dominates."**
  The whole point of the regression threshold is to prevent this.
  Bigger upside doesn't compensate for a 10pp drawdown increase
  in this domain.

## See also

- `SDLC_LESSONS_FOR_NEW_PROJECT.md` §4.2
- `src/rl_swing/rl/validation/acceptance_gate.py`
- `tests/unit/test_acceptance_gate.py`
- Issue #6 (FEAT — gate module)
