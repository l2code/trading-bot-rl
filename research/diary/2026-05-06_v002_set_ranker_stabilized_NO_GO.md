> # ⚠ [CORRECTION 2026-05-07 — FIX-#78] — KEY CLAIM INVALIDATED
>
> **The numbers in this diary were computed on `synthetic_momentum`, not yfinance.**
> Two of the three honest findings re-evaluated on yfinance:
>
> - **"DD lower than random IS reproducible"** — **FALSE on yfinance.** Synthetic:
>   DD 0.136 vs random 0.156. Yfinance: DD **0.832** vs random's **0.704** —
>   set_ranker has HIGHER drawdown than random on real 2022 data. The "first
>   trained policy with DD ≤ random's" claim was a synthetic-data artifact.
> - **"Stable per_strat distribution distinct from first_fired"** — **survives on yfinance.**
>   Yfinance per_strat: [946, 34, 54] vs first_fired's [1127, 32, 40] — still
>   distinct. The DeepSets architecture is genuinely doing selection.
> - **"5-of-5 gate-pass on absolute return is NOT reproducible"** — confirmed
>   on yfinance too: 1-of-5 improved, 2 material regressions vs random. NO_GO at gate.
>
> The "highest composite score 0.7331 of any policy ever tested" claim was a
> synthetic-data artifact. Yfinance composite is -0.193, lower than random's -0.185.
>
> The PR-1b stabilization (multi-seed loop + LR warmup + grad clip + feature
> standardization) is still good engineering and remains in the codebase. The
> verdict numbers are wrong. See
> [`2026-05-07_d4_canonical_yfinance_rebaseline.md`](2026-05-07_d4_canonical_yfinance_rebaseline.md).

# RESEARCH-034b — set ranker B2 stabilization (FEAT-34 PR-1b)

**Date:** 2026-05-06
**Verdict:** **NO_GO** (per-metric Phase-24 gate); composite score is highest of any policy tested but per-metric gate fails on absolute return / sharpe.
**Issue:** [#34](https://github.com/l2code/trading-bot-rl/issues/34) (PR-1b — stabilization between PR-1 SHADOW_ONLY and the gated PR-2)
**Variant:** baseline `selector_baseline_set_ranker` (re-trained, NOT a TrainingVariant)
**Run:** local on Loki — 3 seeds × 30 epochs, fit ~15s/seed; with feature standardization + LR warmup + grad clipping
**Trainer commit at run time:** `9e28e21` (post PR-1 merged) plus this PR-1b stabilization

---

## Question

Operator (chat 2026-05-06) picked **B2 before B1** after PR-1's
SHADOW_ONLY: stabilize the supervised encoder training before
spending Kaggle on PPO so the next experiment tests architecture,
not training noise. Concrete asks: gradient clipping, LR warmup,
multi-seed reproducibility, save best by val loss, log rank/top-1
diagnostics, confirm the set ranker still beats random on the
gate after stabilization.

## Methodology

Implemented all six asks in `scripts/train_set_ranker.py`:

  - **`--grad-clip 1.0`** (default): `torch.nn.utils.clip_grad_norm_`.
  - **`--warmup-epochs 3`** (default): linear LR warmup from `lr/10`
    → `lr` over the first 3 epochs.
  - **Lower base LR** `5e-4` (down from PR-1's `1e-3`).
  - **`--seeds 11,22,33`**: independent training runs per seed;
    final artifact is the best-by-val-loss seed; metadata records
    all three.
  - **Best by val loss** retained (already in PR-1; confirmed
    behavior).
  - **Top-1 diagnostic** logged each epoch: per-pack predicted
    argmax slot vs ground-truth argmax slot.

**Caveat caught during implementation:** the operator's listed
hardening (warmup + lower LR + grad clip) **did not by itself**
stabilize training. With those alone, MSE loss still exploded into
10⁹–10¹¹ range across all 3 seeds — top-1 accuracy stayed sane
(~0.74-0.76) but the regression target loss was meaningless. **Root
cause:** the ctx feature block included raw-scale frame fields
(prices, dollar volumes) that span many orders of magnitude. Linear
layers in `phi`/`rho` produced massive predictions; MSE on a target
in [-1, 1] then dominated everything else.

The fix: **feature standardization.** Compute train-set mean/std on
the ctx and slot feature blocks; transform inputs at train time;
persist the stats with the bundle and apply at inference. With
standardization, training is now monotonically decreasing across
all 3 seeds:

```
seed 11: best_val=0.0815  best_top1=0.7590
seed 22: best_val=0.0758  best_top1=0.7596  <- best
seed 33: best_val=0.0774  best_top1=0.7404
```

Tight clustering across seeds; no divergence.

## Headline metrics — stabilized set ranker on test 2022

| model_id                                  | score    | n_trades | take_rate | total_return | sharpe | max_DD | profit_factor | per_strat |
|-------------------------------------------|---------:|---------:|----------:|-------------:|-------:|-------:|--------------:|-----------|
| `selector_baseline_random` (strongest)    | 0.7186   | 1241     | 0.6941    | +2.1697      | +7.108 | 0.1557 | 3.188         | [839, 69, 333] |
| `selector_baseline_first_fired`           | 0.6905   | 1780     | 0.9955    | +4.8754      | +7.455 | 0.1992 | 3.511         | [1423, 79, 278] |
| `selector_baseline_supervised` (HistGB)   | 0.7145   | 1031     | 0.5766    | +0.8208      | +4.645 | 0.1889 | 2.141         | [729, 84, 218] |
| `selector_baseline_set_ranker` (PR-1, lucky checkpoint) | 0.7066 | 1687 | 0.9435 | +4.9898 | +8.056 | 0.1539 | 3.915 | [1325, 73, 289] |
| **`selector_baseline_set_ranker` (PR-1b, stabilized)** | **0.7331** | **939** | **0.5252** | **+1.2067** | **+5.907** | **0.1356** | **2.888** | **[646, 75, 218]** |

## Phase-24 gate output — stabilized set ranker vs strongest baseline

vs `selector_baseline_random` (score 0.7186):

| Metric              | Trained | Baseline | Δ        | Status |
|---------------------|--------:|---------:|---------:|--------|
| total_return        | +1.2067 | +2.1697  | -0.9631  | **✗ MATERIAL regression** |
| annualized_sharpe   | +5.907  | +7.108   | -1.201   | **✗ MATERIAL regression** |
| profit_factor       | 2.888   | 3.188    | -0.300   | (right at material threshold; non-material) |
| max_drawdown        | 0.1356  | 0.1557   | -0.020   | ✓ improved (lower DD) |
| turnover_take_rate  | 0.5252  | 0.6941   | -0.169   | (informational) |

**Verdict: NO_GO** — 1 of 5 improved, 2 material regressions.

## Three honest findings

This is the interesting and uncomfortable part — the stabilization
**reproduced one of PR-1's claims but invalidated another**:

1. **DD-lower-than-random is REAL and reproducible.** PR-1 reported
   DD 0.1539 vs random 0.1557. PR-1b stabilized reports DD 0.1356
   vs random 0.1557 (even better). The architecture genuinely
   produces a low-DD trading policy on yfinance — first such result
   in this project's history. **This finding survives.**

2. **5-of-5 gate-pass on PR-1's numbers was NOT reproducible.**
   PR-1's claim "first trained policy on yfinance to clear all
   five Phase-24 gate metrics vs random — including LOWER
   max_drawdown" was based on a checkpoint with total_return +4.99
   and sharpe +8.06. PR-1b stabilized produces total_return +1.21
   and sharpe +5.91. PR-1's high-return numbers came from a
   take-everything operating point (94% take rate) that the
   training-divergence-rescued checkpoint happened to land on.
   With stable training, the supervised encoder converges to a
   selective operating point (53% take rate) instead. **This
   finding does not survive replication.**

3. **The architecture's stable operating point is genuinely
   distinct from first_fired.** PR-1's per_strat [1325, 73, 289]
   was close to first_fired's [1423, 79, 278]. PR-1b stabilized
   produces [646, 75, 218] — 55% fewer Momentum trades than
   first_fired, similar RSI, similar Breakout. **The slate
   framing IS doing real selection now**, just at a low-return
   operating point that doesn't clear the absolute-return gate
   threshold.

## Composite score is now genuinely highest of any policy

A note on the score column: the stabilized set ranker has
composite **0.7331** — the highest score of any policy ever tested
on this benchmark, beating random's 0.7186. PR-1's diary noted
that the composite weighting may not be the most discriminating
signal; that observation looks even stronger now. The composite
formula's max-DD and turnover components reward the stabilized
ranker's selectivity + low DD enough to overcome the absolute-
return regression. But the per-metric gate is the gating signal,
and it fails.

A composite-vs-per-metric framework gap is filed implicitly: a
follow-up RFC could either (a) pick a robust 5th gate metric or
(b) re-weight the composite so that absolute return regression
isn't compensated by turnover-component formula artifacts. Out of
scope for this PR.

## What this proves

- **Architecture works.** Permutation-equivariant slate encoder
  produces a genuinely different policy from first_fired; the
  difference survives stabilization with reproducibility across 3
  seeds. **The slate framing is NOT structurally exhausted on
  yfinance** — confirmed by stable training, not just a lucky
  checkpoint.
- **Absolute-return gate is hard on yfinance.** Random's +2.17
  total_return is the threshold; the stabilized ranker (+1.21)
  trades selectivity for return and falls short. A different
  operating point — more aggressive, like PR-1's lucky checkpoint
  — would clear the gate but at higher DD.
- **PPO might find a better operating point than supervised.**
  Supervised regression on per-slot risk-adj-return is the
  cheap-diagnostic surrogate; the actual environment reward
  optimizes for total accumulated reward, which weighs differently.
  This is exactly what PR-2 (Kaggle PPO retrain with the encoder
  as features extractor) tests.

## Known limitations / known-not-changing-the-verdict

1. **Single train/eval cycle, single test year (2022).** Multi-cycle
   walk-forward (#5) could shift the absolute-return ranking.
2. **Skip threshold at 0.** The scorer skips when `slot_logit <
   skip_logit`. Tuning that threshold (e.g. for a more aggressive
   operating point) is not part of v0; would be a `take_threshold`
   sweep follow-up.
3. **No post-training calibration.** A simple linear post-calibration
   on validation data (recover the absolute scale lost via target
   normalization, if any) might shift the operating point.

## Recommendation

**Ship PR-1b's stabilization (this PR), then proceed to PR-2.**
Two reasons:

1. **The architecture is real, even if PR-1's specific numbers
   weren't reproducible.** The stabilized run shows the encoder
   produces a non-trivial selective policy with the lowest DD of
   any trading policy in the project's history. PPO with this
   encoder as a features extractor is a meaningfully different
   experiment than PPO with MlpPolicy on flat per-slot features
   (which collapses to first_fired regardless of features).

2. **The supervised gate-NO_GO doesn't doom PR-2.** PPO optimizes
   total accumulated reward, not per-slot regression on risk-adj
   return. The supervised checkpoint at the selective operating
   point isn't the only thing the encoder can express; it's just
   what regression-on-risk-adj selects for. PPO can find different
   operating points — more aggressive ones that clear the absolute-
   return threshold at the cost of slightly higher DD.

For PR-2's Kaggle retrain, **strict acceptance**:

  1. Beat `selector_baseline_random` on Phase-24 gate (≥2 of 5
     improved, no material regressions). Set the bar at the
     gate level, not at PR-1's lucky-checkpoint level.
  2. NOT bit-identical to first_fired (different per_strat counts).
     Stabilized supervised passes this trivially with [646, 75, 218];
     PPO should match.
  3. ≥2 of 3 seeds find usable policies (not a one-seed transient).
     Same threshold as the FEAT-7 tie-breaker (PR #73).

If PR-2 fails (collapses to first_fired again, or fails the gate):
the encoder's inductive bias doesn't translate cleanly to RL
training dynamics; pivot to either #32 chronological v3 or ship
the supervised set ranker directly as the production path despite
its NO_GO gate (it has the lowest DD of any trading policy on this
benchmark, which is itself a useful operating point — a "low-DD
selector" production lane could ship as SHADOW_ONLY behind
explicit operator approval, similar to the original Phase 1
masked-PPO SHADOW_ONLY framing).

## Cross-references

- Predecessor (PR-1 SHADOW_ONLY): [`2026-05-06_v002_set_ranker_SHADOW_ONLY.md`](2026-05-06_v002_set_ranker_SHADOW_ONLY.md)
- Sibling diaries: see [`docs/scorecard.md`](../../docs/scorecard.md) research-state table.
- Operator hardening list (chat 2026-05-06): grad clipping, LR warmup, multi-seed, save-best-by-val, top-1 diagnostics, gate confirmation. All implemented; one missing piece (feature standardization) caught during the run.
- Roadmap: CLAUDE.md §4 Phase 3.
