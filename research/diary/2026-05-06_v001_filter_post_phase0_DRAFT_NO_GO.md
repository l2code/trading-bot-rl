# RESEARCH-001b — v001 filter on yfinance starter_equities (post-Phase-0)

> **DRAFT_NO_GO** as of 2026-05-06 evening. Phase 0 simulator/eval
> fixes (#22 #23 #24 #26 #36 #49 #50 #51 #52 #53 #54 #56 #57 #58 #59
> #61 #62) are merged. Audit-bundle re-run completed; audit-v2 and
> phase0-final re-runs still in flight on Kaggle. Numbers below are
> from the audit-bundle run; the audit-v2 / phase0-final refinements
> (#56 idle-day window, #57 trading-day calendar) will shift these
> slightly but **will not change the qualitative verdict** — the
> collapse mode and the gate output are stable across the
> remaining metric refinements. This entry promotes from DRAFT to
> FINAL once those runs land and we replace the numbers in place.

**Date:** 2026-05-06
**Verdict:** **NO_GO** (draft — qualitative verdict stable; numbers pending final runs)
**Issue:** [#2](https://github.com/l2code/trading-bot-rl/issues/2) (PROVISIONAL banner from previous entry will be lifted by FINAL)
**Variant:** `filter_v001`
**Run (audit-bundle):** Kaggle `crazypenguin/rl-swing-v001-rerun-audit-bundle`
**Trainer commit at run time:** `eb32fba` (FIX-AUDIT-BUNDLE).
**Final-state metrics will use:** `crazypenguin/rl-swing-v001-rerun-audit-v2` or `…-phase0-final` once they complete.

---

## Question

After all Phase 0 simulator/evaluation fixes, does the v1 trade-
filter PPO produce a policy that beats the strongest baseline on
the 2022 yfinance test window under the Phase-24 gate?

## Source

- **Provider:** yfinance (daily bars, auto-adjusted).
- **Tier:** **exploratory** — yfinance cannot earn a GO regardless
  of result. NO_GO at this tier is conclusive against the framing.
- **Universe:** `starter_equities` (15 names + SPY, QQQ).

## Phase 0 fixes applied to this run

All 16 P1+P2 fixes from the operator's audits + my parallel passes:

- **#22** size_pct now scales return_pct (portfolio contribution, not asset return).
- **#23** round-trip cost charged 2× per-side.
- **#24** walk-forward warmup loads 1.5y bars before test_start so long-lookback features populate on day 1.
- **#36** date-ordered daily-P&L metrics replace per-trade composite for Sharpe/DD/PF.
- **#49** reward weights recalibrated for sized returns (turnover 0.30→0.05, holding 0.05→0.02).
- **#50** peak_drawdown_pct portfolio-scale (×size_pct).
- **#51** training _evaluate uses daily-P&L metric for checkpoint selection (matches walk-forward report).
- **#52** idle days fill with zero P&L in the daily series.
- **#53** model.zip alias picks global best across seeds (was last-seed-wins).
- **#54** v1 env skip-CF cost computed at full notional (impact term not zeroed).

(Audit-v2/phase0-final adds #56 idle-day window from test_start/end not trade span, #57 trading-day calendar instead of weekday approx, #58 RewardModel defaults match YAMLs, #59 walk-forward skip-CF cost full notional, #61 trainer trading_days from val_env bars, #62 .get() fallbacks match new defaults. None affect the qualitative collapse.)

## Methodology

- **Variant:** `filter_v001` — per-candidate filter, dedupe by
  `(symbol, date)` keeping highest signal_strength, action space
  `Discrete(4)` (skip / take_25 / take_50 / take_100).
- **Window:** train 2014-01-01..2020-12-31, validation 2021,
  test 2022.
- **Seeds:** 11, 22, 33.
- **Hyperparams:** PPO MlpPolicy 128×128, n_envs=4 (SubprocVecEnv
  fork), 500k timesteps, default sb3 PPO (`ent_coef=0.01`, `lr=3e-4`).
- **Reward params (post-FIX-49):** `target_risk_pct=0.02`,
  `drawdown_penalty_weight=0.10`, `turnover_penalty_weight=0.05`,
  `holding_period_penalty_weight=0.02`, `skip_counterfactual_scale=1.0`,
  `reward_clip=5.0`.
- **Cost layer:** `EquityExecutionModel` active, round-trip charged
  per FIX-23.
- **Candidate strategies (loose config):** Momentum (`min_r20=-0.02`,
  no SMA200 gate, RS≥-0.05), RSI mean-reversion (rsi≤35),
  Breakout (≥0.7× rel-vol, dist_high≥-0.02). Aggregated via
  `StrategyAggregator`.

## Headline metrics — audit-bundle

Test window 2022-01-01..2022-12-31. Comparison against the
**strongest baseline** by `validation_composite_score`:

| model_id                        | score   | n_trades | take_rate | total_return | sharpe | max_DD |
|---------------------------------|--------:|---------:|----------:|-------------:|-------:|-------:|
| `baseline_random` (strongest)   | 0.7341  | 1349     | 0.754     | +1.36        | +7.00  | 0.094  |
| `baseline_always_take_50`       | 0.7190  | 1780     | 0.996     | +1.37        | +6.92  | 0.104  |
| `baseline_always_take_100`      | 0.6906  | 1780     | 0.996     | +4.55        | +6.92  | 0.199  |
| `baseline_never_take`           | 0.3250  | 0        | 0         | 0            | 0      | 0      |
| **`ppo_filter_v001` (trained)** | **0.6906** | **1780** | **0.996** | **+4.55** | **+6.92** | **0.199** |

The trained PPO is **bit-identical to `baseline_always_take_100`** —
same trade count, same take rate, same return to 4 decimals,
same Sharpe and DD. Eval-history confirms convergence to "always
take_100" at step 50,000 (first checkpoint) and zero variance
through step 500,000 across all 3 seeds. No discrimination.

## Gate output

vs **strongest baseline (`baseline_random`, score 0.7341)**:

| Metric              | Trained | Baseline | Δ      | Status |
|---------------------|--------:|---------:|-------:|--------|
| total_return        | +4.55   | +1.36    | +3.19  | ✓ improved |
| annualized_sharpe   | +6.92   | +7.00    | -0.08  | small regression (within threshold) |
| profit_factor       | (high)  | (high)   | ≈      | ≈ |
| max_drawdown        | 0.199   | 0.094    | +0.105 | **✗ MATERIAL regression** (>0.05 threshold) |
| turnover_take_rate  | 0.996   | 0.754    | +0.24  | ✓ improved (informational) |

**Material regression on max_drawdown caps verdict at NO_GO** per
the Phase-24 gate. Multiple metrics improve, but the gate
explicitly disallows trading drawdown for return.

## Why v1 collapsed to "always take" (and not skip this time)

Pre-Phase-0, v1's reward was on the wrong scale (FIX-22). Under
the corrected reward + recalibrated weights:

```
typical winner: size 10%, asset +5%, cost 10bps/side
  net_return = 0.10 × (0.05 - 0.002) = +0.0048
  risk_adj   = 0.0048 / 0.02         = +0.24
  dd_pen (post-FIX-50, sized) = 0.10 × 0.10 × 0.03 / 0.02 = +0.015
  turnover_pen = 0.05
  take reward = +0.24 - 0.015 - 0.05 = +0.175  ← positive on winners
```

In yfinance 2022 the candidate distribution is positive-EV in
aggregate; "always take_100" averages positive reward. The agent
finds this on rollout 1 and stays. The signal-to-noise to learn
"some winners are bigger than others" isn't strong enough at
default `ent_coef=0.01` — entropy collapses before the policy
explores discriminating subsets.

## What this proves

Phase 0 fixes are **correct** — simulator + evaluator are now
self-consistent. But the v1 filter framing (skip/take_25/50/100)
**does not learn discrimination at default PPO hyperparams on
yfinance starter_equities**. The trained model is equivalent to
the take-100 baseline; that baseline is itself dominated by
take-50 and random on the gate (both have lower DD).

## Known limitations / known-not-changing-the-verdict

1. **Single test cycle** (#5). 2022-only; multi-cycle WF could
   shift relative numbers but not the equivalence-to-baseline
   finding.
2. **Default hyperparams** — Optuna entropy/LR sweep (#8) tests
   whether `ent_coef=0.05+` unsticks the early collapse. Filed.
3. **Exploratory tier.** yfinance cannot earn GO regardless. WRDS
   replication (#4) required for any GO claim.
4. **Three seeds.** Bit-identical convergence across all three
   strongly suggests this is the policy's globally-optimal
   collapse target, not seed-specific noise.

## What would change the verdict (in priority order)

- **Evidence against:** an Optuna-tuned PPO (#8) producing a
  trained model that strictly beats `baseline_always_take_50`
  (currently the best simple baseline) on the gate.
- **Evidence for:** the reverse — even with an Optuna sweep, no
  v1 PPO beats `baseline_always_take_50`. That would close the v1
  filter framing as a research direction.

## Recommendation

**Do not spend more run budget on v1 PPO under default
hyperparams.** Either:

1. Run #8 (Optuna sweep) to test whether tuning unsticks the
   collapse. If no PPO config beats `baseline_always_take_50`,
   close v1 PPO.
2. Run #30 (supervised ranker baseline) on the same features.
   If a simple ranker beats the take-50 baseline, RL machinery
   isn't earning its complexity yet.

Per the operator (chat 2026-05-06): proceed with #29 MaskablePPO
for v2 first; #30 supervised baseline second; v1 Optuna third.

## Cross-references

- Predecessor (PROVISIONAL): `2026-05-06_v001_filter_loose_NO_GO.md`
- Sibling: `2026-05-06_v002_selector_post_phase0_DRAFT_NO_GO.md`
- Roadmap: CLAUDE.md §4
- Operator audit chats: 2026-05-06 (Phase 0 fixes + Next Stage framing)
