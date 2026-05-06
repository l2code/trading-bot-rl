# RESEARCH-15 — per-strategy training-window EV analysis

> **PROVISIONAL (numbers superseded; ranking stands)** as of
> 2026-05-06. The script reused the production simulator and cost
> model, both of which carried P1 bugs at the time of the run:
> [#22](https://github.com/l2code/trading-bot-rl/issues/22) (asset
> return not scaled by size_pct) and
> [#23](https://github.com/l2code/trading-bot-rl/issues/23) (round-
> trip cost charged once). Both fixes are now merged. The
> **qualitative ranking** (momentum > breakout > rsi by mean
> risk-adjusted EV) is unchanged in the post-Phase-0 v1/v2 re-runs
> on the same 2014–2020 training window. The absolute EV numbers
> in this entry are still on the old simulator and should not be
> quoted as decision-grade; if a re-run of this script is needed
> for Phase 1 work (e.g. supervised baseline #30), do it before
> citing absolute numbers. The H1-vs-H2 conclusion (PARTIAL H2 —
> rational Momentum preference, irrational Breakout exclusion)
> survived the rebuild and motivates the masked-PPO direction in
> Phase 1.

**Date:** 2026-05-06
**Verdict:** **PARTIAL H2** (provisional) (Momentum-preference is rationally
motivated on training data, but agent's specialization to
*only* Momentum is irrational — leaves 8,411 positive-EV Breakout
candidates on the table).
**Issue:** [#15](https://github.com/l2code/trading-bot-rl/issues/15)
**Script:** `scripts/per_strategy_training_ev.py`
**Tier:** **exploratory** (yfinance starter_equities — same data
as v002 training).

---

## Question

Was the trained `selector_v002` policy's collapse to
`per_strategy_take_counts = [323, 0, 0]` (Momentum-only)
rationally motivated on the training distribution (H2) or pure
entropy-collapse (H1)?

## Methodology

Pure data analysis — no RL run. For the v002 training window
(2014-2020 yfinance starter_equities), used the same loose
strategy configs the agent saw, ran every candidate through the
same `ExecutionSimulator` + `EquityExecutionModel` the env uses,
and aggregated per-strategy descriptive stats.

## Per-strategy training-window EV (yfinance, 2014-01-01..2020-12-31, starter_equities, loose config)

| Strategy | n | mean ret | median ret | win rate | mean risk-adj | sharpe-ish | mean cost (bps) | median hold (d) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **momentum** | 15,025 | +0.0082 | +0.0054 | 0.543 | +0.327 | +0.167 | 10.5 | 14 |
| **rsi_mean_reversion** | 1,534 | +0.0035 | +0.0041 | 0.546 | +0.151 | +0.091 | 10.3 | 7 |
| **breakout** | 8,411 | +0.0066 | +0.0045 | 0.538 | +0.279 | +0.152 | 10.3 | 14 |

## Interpretation

### H2 partially confirmed: Momentum is rationally the "first pick"

Momentum has the highest mean per-trade return (+0.82%) and
highest mean risk-adjusted return (+0.327 vs Breakout's +0.279
and RSI's +0.151). Win rates are nearly identical (~54%) across
all three — the difference is in *magnitude* of winners minus
losers, not hit rate.

So a "specialize in the single highest-EV strategy" heuristic
*does* point at Momentum. The agent's preference is not random.

### H1 still needed: specialization to ONLY Momentum is irrational

But picking *only* Momentum and never Breakout is leaving real
money on the table. Breakout's mean risk-adjusted return
(+0.279) is 85% of Momentum's (+0.327) — well above the +0.151
the agent would get from RSI. A portfolio policy that takes
Momentum when it fires AND Breakout when it fires would have
clearly higher expected return than Momentum-only.

Concretely: 8,411 Breakout candidates fired across training.
Each had +0.0066 mean return = +66 bp. Skipping all of them
saves the cost (~10 bps per trade) but misses ~56 bps of net
EV per trade × 8,411 trades ≈ **substantial residual value
the agent walked away from**.

The fact that the agent locked into Momentum-only despite Breakout
being almost as good IS entropy-collapse: PPO found a positive-EV
local optimum (Momentum) and didn't have enough exploration
budget to discover that adding Breakout-when-fired would dominate.

### The combined diagnosis

Both H1 and H2 are partially correct:

- **H2 explains the *direction* of the collapse.** Momentum is
  genuinely the best single strategy on training data, so an
  entropy-starved policy preferring Momentum specifically (rather
  than, say, RSI) is internally consistent.
- **H1 explains the *severity* of the collapse.** Picking
  Momentum *exclusively* — never Breakout, even when Momentum
  doesn't fire — is the entropy-collapse signature. A higher-
  entropy policy would discover the portfolio optimum.

## What this means for #8 (Optuna sweep)

The Optuna sweep is still the right next experiment, but the
target metric should be: does higher `ent_coef` produce a
trained policy with `per_strategy_take_counts` that *includes*
Breakout (and ideally RSI)? If yes, entropy-collapse is the
sole problem. If a high-entropy policy still ignores Breakout,
the architecture itself is missing something (e.g., the
observation lacks the cross-strategy context needed to learn
"pick Momentum if it fires, else fall back to Breakout").

In other words: the bar Optuna has to clear is not "trained model
beats baseline" but "trained model's `per_strategy_take_counts`
shows real diversification across fired strategies."

## What this means for the variant lineup

A new selector baseline is now obviously valuable: **"take every
fired strategy"** (a portfolio policy that takes the candidate
from every strategy that fired on the pack, scaled appropriately).
That baseline bounds the EV the trained agent is leaving on the
table. Filing as a follow-up.

Also worth noting: this analysis tested H1/H2 on training data.
The 2022 test window (where `selector_baseline_random` already
beat trained 0.7037 vs 0.6665) is a different distribution —
2022 was bearish for tech (QQQ -33%) and Momentum specifically
struggled while RSI mean-reversion did better. So the agent's
training-rational specialization to Momentum becomes test-time
suboptimal under regime shift. That's a **regime mismatch**
problem on top of the entropy-collapse problem, separate from
both H1 and H2 as originally framed.

## Cross-references

- RFC #15 (this analysis closes it)
- RESEARCH-3 #3 (the v2 NO_GO that motivated this)
- #8 (Optuna sweep — next experiment, with refined success
  criteria from this analysis)
- New follow-up issue (to be filed): "FEAT: add
  `selector_baseline_take_all_fired` as a portfolio-policy
  reference baseline"
