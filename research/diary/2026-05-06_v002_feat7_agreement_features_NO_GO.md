# RESEARCH-007 — cross-strategy agreement features + cheap-ranker re-test

**Date:** 2026-05-06
**Verdict:** **NO_GO** (marginal improvement; gap to random halved but ranker still loses 3-of-5 material regressions)
**Issue:** [#7](https://github.com/l2code/trading-bot-rl/issues/7) (agreement-features half closed; pairwise-one-hot + multi-day variants stay open)
**Variants affected:** observation builder for `selector_v002` + `selector_v002_masked`; supervised ranker baseline (`selector_baseline_supervised`)
**Run:** local on Loki — sklearn HistGB re-fit on 28,219 rows × 36 features (was 27); fit in 1.1s; in-sample MSE 0.01673
**Trainer commit at run time:** `5c5696d` (post FEAT-30 supervised ranker NO_GO merged)

---

## Question

After landing the operator-specified cross-strategy agreement
features (FEAT-7) — pack-level moments + per-slot rank/top-signal
flags — does the cheap supervised ranker improve materially on the
Phase-24 gate? The agreed test was: **if the ranker improves
materially, the slate now has usable signal and PPO/HPO becomes
worth run budget. Otherwise, v002 selector framing is approaching
a ceiling on yfinance.**

## Source

- **Provider:** yfinance (daily bars, auto-adjusted).
- **Tier:** **exploratory** — same as the prior v002 / masked / FEAT-30 runs.
- **Universe:** `starter_equities`.

## Methodology

- **New module** `src/rl_swing/rl/env/agreement_features.py`:
  pure-function helpers `compute_pack_agreement(pack, n_strategies)`
  and `compute_slot_agreement(pack, slot_idx)`. Pack-level fields:
  `pack_n_fired`, `pack_all_fired`, `pack_signal_max`,
  `pack_signal_mean`, `pack_signal_std`, `pack_signal_gap_top2`,
  `pack_same_symbol_strategy_agreement`. Per-slot fields:
  `slot_is_top_signal`, `slot_rank_by_signal`. Tie-break for
  `is_top_signal` / `rank` is the lower slot_idx (matches
  `selector_baseline_first_fired` semantics).
- **Wired** into `MultiStrategyObservationBuilder` (so the next
  masked-PPO retrain sees them) AND into the supervised ranker's
  `PER_SLOT_FEATURE_NAMES` + `build_slot_features`. Train-time
  and inference-time vectors are bit-identical.
- **No experiment-YAML flag.** Operator's call (chat 2026-05-06):
  the default-OFF flag pattern (CLAUDE.md §3.3) is for behavior
  changes that affect deployed paths; this is a research feature
  on a research-only codebase. A/B is the new ranker run vs the
  prior FEAT-30 ranker.
- **Re-trained the supervised ranker** with the new feature set
  (28,219 rows × 36 features, fit in 1.1s on Loki). Same
  HistGradientBoostingRegressor defaults; no Optuna sweep on the
  ranker itself (out of scope for v0).
- **Re-ran `rl-swing validate`** on the v002_masked YAML to
  produce a walk-forward report including the new ranker scorer.
  Computed Phase-24 gate via `acceptance_gate.evaluate_gate()` —
  not eyeballed.

## Headline metrics — FEAT-7 ranker on test 2022

Daily-P&L basis (FIX-#36), trading-day spread = 260 (FIX-#57).

| model_id                             | score   | n_trades | take_rate | total_return | sharpe | max_DD | per_strat |
|--------------------------------------|--------:|---------:|----------:|-------------:|-------:|-------:|-----------|
| `selector_baseline_random` (strongest) | 0.7186 | 1241    | 0.6941    | +2.1697      | +7.108 | 0.1557 | [839, 69, 333] |
| `selector_baseline_first_fired`      | 0.6905  | 1780    | 0.9955    | +4.8754      | +7.455 | 0.1992 | [1423, 79, 278] |
| `selector_baseline_highest_signal`   | 0.6910  | 1780    | 0.9955    | +4.5134      | +7.351 | 0.1975 | [1259, 116, 405] |
| **`selector_baseline_supervised` (FEAT-7)** | **0.7145** | **1031** | **0.5766** | **+0.8208** | **+4.645** | **0.1889** | **[729, 84, 218]** |

Reference (from PR #71, prior FEAT-30 ranker without agreement features):
- composite 0.7107, return +0.8214, sharpe +4.376, max_DD 0.1963, n_trades 1088, per_strat [778, 77, 233].

## Phase-24 gate output — FEAT-7 ranker vs strongest baseline

vs `selector_baseline_random` (score 0.7186):

| Metric              | Trained | Baseline | Δ       | Status |
|---------------------|--------:|---------:|--------:|--------|
| total_return        | +0.8208 | +2.1697  | -1.3490 | **✗ MATERIAL regression** (>0.05 threshold) |
| annualized_sharpe   | +4.645  | +7.108   | -2.463  | **✗ MATERIAL regression** (>0.5 threshold) |
| profit_factor       | 2.141   | 3.188    | -1.047  | **✗ MATERIAL regression** (>0.3 threshold) |
| max_drawdown        | 0.1889  | 0.1557   | +0.0332 | (under 0.05 threshold) |
| turnover_take_rate  | 0.5766  | 0.6941   | -0.1174 | (informational) |

**Verdict: NO_GO** by 3 material regressions — same shape as the
FEAT-30 ranker, just with smaller deltas.

## The leverage test: did agreement features help?

The test the operator authorized this PR around:

| Quantity | FEAT-30 (no agreement) | FEAT-7 (with agreement) | Δ |
|---|---:|---:|---:|
| composite_score | 0.7107 | **0.7145** | **+0.0038** |
| total_return    | +0.8214 | +0.8208 | -0.0006 |
| sharpe          | +4.376 | **+4.645** | **+0.269** |
| profit_factor   | 2.020 | **2.141** | **+0.121** |
| max_drawdown    | 0.1963 | **0.1889** | **-0.0074** |
| n_trades        | 1088 | 1031 | -57 (more selective) |
| **gap to `selector_baseline_random`** (score) | **-0.0079** | **-0.0041** | **gap halved** |

The features **did help** — composite +0.0038, sharpe +0.27, DD
-0.0074 (~3.7% better), the ranker became slightly more selective
(57 fewer trades). The gap to `selector_baseline_random` halved
from 0.0079 to 0.0041. **But** none of the three material gate
regressions flipped. The new features moved the ranker partway
along the right axis without crossing the gate.

This is the *marginal* improvement, not the *material* improvement
the operator's framing required. Direct quote from the operator's
chat (2026-05-06):
> "If ranker improves materially, the slate now has usable signal.
> Then PPO/HPO becomes worth run budget."

A 0.5% composite-score gain that doesn't flip any gate metric is
hard to call material. It's evidence that the features carry SOME
signal, but not enough to put yfinance + the v002 slate framing
above random.

## What this proves

- **Agreement features carry real-but-modest signal.** Sharpe +0.27,
  DD -0.0074, ranker uses them to be slightly more selective. They
  are doing what they were designed to do — just at a smaller
  magnitude than would flip the verdict.
- **The slate framing on yfinance is approaching a ceiling.** The
  prior FEAT-30 ranker was NO_GO with 3 material regressions; this
  ranker is NO_GO with 3 material regressions of slightly smaller
  magnitude. The same shape of failure repeats. Random's lower DD
  (0.156 vs ~0.189-0.199 for every trained policy) is doing most
  of the work, and skip-30%-uniformly is a portfolio-level noise
  reducer that the slate-feature ranker can't beat.
- **Re-training masked-PPO on the new obs is unlikely to fix
  this.** The features helped the ranker by ~0.0038 composite; if
  PPO at default `ent_coef=0.01` collapsed to `first_fired` on the
  smaller feature set, the marginal new signal is probably not
  enough to dislodge that attractor. PPO retraining would cost a
  Kaggle run and is a low-EV bet.

## Known limitations / known-not-changing-the-verdict

1. **Ranker hyperparams not tuned.** A sweep over HistGB's
   `learning_rate`, `max_iter`, `max_depth` plus a target-encoding
   on `slot_idx` could squeeze more out of the same features. **Not
   blocking — the verdict is stable across reasonable hyperparams
   on a noisy 28k yfinance dataset.**
2. **Single train/eval cycle.** 2022-only.
3. **Exploratory tier.** WRDS would change the noise floor and
   could change the calculus on whether the slate framing is
   exhausted.
4. **No PPO retrain on the new obs.** Deferred — see "Recommendation."

## What would change the verdict

- **Promotion above NO_GO:** the same ranker on WRDS canonical data
  ([#4](https://github.com/l2code/trading-bot-rl/issues/4))
  clearing the gate vs random. Or a richer feature set
  (#7-followup: prior win-rate per strategy on a strictly-prior
  window; #7-followup: pairwise one-hot of fired-strategy
  combinations) re-doing the leverage test on yfinance.
- **Closing v002 selector framing entirely:** if the masked-PPO
  retrain on the new obs (cost: ~45 min Kaggle private, gated on
  operator approval) also fails to clear the gate, the
  slate-with-priority-and-agreement framing is structurally
  exhausted on yfinance. Move to Phase 3:
  [#34 set/attention encoder](https://github.com/l2code/trading-bot-rl/issues/34)
  or [#32 portfolio-aware chronological v3](https://github.com/l2code/trading-bot-rl/issues/32).

## Recommendation

The marginal-not-material framing puts the next decision on the
operator. Two coherent paths from here:

**Path A — defer #27 Optuna; pivot to Phase 3 architectural work
(#34 / #32).** Justification: the features moved the gap by half
of what they would need to flip the verdict. Optuna-tuning PPO at
the same feature ceiling is unlikely to bridge the rest. Move to
set/attention or chronological v3.

**Path B — burn one masked-PPO retrain on the new obs as a tie-
breaker before #27.** Justification: the FEAT-7 features gave
the ranker a real (small) lift; PPO with masking + agreement
features + tuned `ent_coef` MIGHT bridge the rest. Cost is one
Kaggle private run (~45 min) plus the rerun's verdict diary.
Operator's call.

The diary's recommendation is **Path A**. The leverage-test
framing was set up before the run with explicit threshold
("material"), and the result didn't clear it. Honoring that
framework matters more than chasing the residual.

## Cross-references

- Predecessor (supervised ranker NO_GO without agreement features): [`2026-05-06_v002_masked_supervised_ranker_NO_GO.md`](2026-05-06_v002_masked_supervised_ranker_NO_GO.md)
- Predecessor (masked-PPO bit-identical to first_fired): [`2026-05-06_v002_masked_SHADOW_ONLY.md`](2026-05-06_v002_masked_SHADOW_ONLY.md) (addendum)
- Sibling diaries: v1 + v2 unmasked FINAL_NO_GO entries; see [`docs/scorecard.md`](../../docs/scorecard.md) research-state table.
- Operator scope chat: 2026-05-06 ("Path B" feature engineering before HPO; bit-identity finding reframed Phase 1).
- Roadmap: CLAUDE.md §4 Phase 1.
