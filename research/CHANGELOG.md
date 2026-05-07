# Project changelog

Rolling, reverse-chronological log of substantive project events:
findings, decisions, run verdicts, RFC outcomes, infrastructure
changes that affect how we work.

This is **not** the operating brief (`CLAUDE.md`) and **not** a
per-experiment artifact (`research/diary/`). It's the
"what-happened-when" log a future reader (human or Claude
session) reads to catch up without having to re-read every PR.

## Entry format

```
## YYYY-MM-DD — <kind>: <one-line summary>

**Issue/PR:** #N (link)
**Diary:** research/diary/...md (if applicable)

Two to five sentences of context: why this happened, what we
learned, what changed in our model of the project.
```

Kinds: `RESEARCH`, `RFC`, `FEAT`, `FIX`, `OPS`, `STRUCTURAL`.

When a PR closes a research issue OR makes a substantive process /
infra change, the same PR appends an entry here. CONTRIBUTING.md §12
codifies the rule.

---

## 2026-05-07 (FEAT-32 M3) — RESEARCH: v003 masked-PPO 500k×3 Kaggle NO_GO; bit-identical to portfolio_baseline_no_op

**Issue:** [#89](https://github.com/l2code/trading-bot-rl/issues/89) FEAT-32 M3
**PR (masking infra):** [#90](https://github.com/l2code/trading-bot-rl/pull/90) merged at SHA `cb60d3b`
**PR (verdict):** TBD
**Diary:** [`2026-05-07_feat32_m3_kaggle_NO_GO.md`](diary/2026-05-07_feat32_m3_kaggle_NO_GO.md)
**Kernel:** `crazypenguin/rl-swing-v003-masked-m3` (private)

After M2 PASS (PR #88) ruled out env degeneracy, M3 ran the Kaggle
private masked-PPO 500k×3 retrain on v003. Wall-time was ~95 min
(operator estimated ~45 min — calibration update: v003 chronological
≈ 2× v002 per-step due to portfolio bookkeeping every step). All 3
seeds (11/22/33) converged to **identical** validation composite
0.32499999999999996 (16 decimals). Trained MaskablePPO is
**bit-identical to `portfolio_baseline_no_op`**: per_action_counts
[511, 0, 0], n_trades=0, return=0, DD=0. The Phase-24 metric gate
trivially "passes" (5/5 improved by zeroing all metrics) but the
"AND not bit-identical to any baseline" criterion fails — same
shape as v002 masked-PPO → first_fired bit-identity (PR #71). v003's
architectural shift (per-day chronological vs per-pack contextual-
bandit) does NOT solve the v002 default-hyperparam exploration
problem; PPO at default `ent_coef=0.01` collapses to the all-skip
attractor on a 3-element action space when the no-trade option is
the highest-EV action (random=-0.278, top1=-0.301, top2=-0.316).
M2 PASS means v003 collapse is hyperparam, not architecture.
Recommended next: M3.b Optuna sweep on `ent_coef` + `lr` per the
FEAT-32 plan. M4 (multi-cycle yfinance per D4-b) is moot under
this verdict.

---

## 2026-05-07 (FEAT-32 M2) — FEAT: behavioral-cloning env-learnability diagnostic PASSES on v3 chronological env

**Issue:** [#87](https://github.com/l2code/trading-bot-rl/issues/87) FEAT-32 M2
**Diary:** [`2026-05-07_feat32_m2_bc_PASS.md`](diary/2026-05-07_feat32_m2_bc_PASS.md)

After M1 shipped the v3 chronological env scaffold (PR #86), M2 asks:
is the env even learnable? A supervised classifier
(`HistGradientBoostingClassifier`) was trained to imitate a hand-coded
3-region state-dependent target policy (`BCTargetPortfolioPolicy`)
over 80 random training windows on yfinance 2014-2020. Held-out val
accuracy on 20 fresh windows: **1.0000** (well above the M2 PASS
threshold of 0.70). The 12-dim obs carries enough signal to express
a non-trivial state-dependent policy; the env-degeneracy hypothesis
for any future PPO collapse is ruled out. M2 PASS unlocks M3 (Kaggle
masked-PPO 500k×3); if M3 collapses, the failure is hyperparam /
exploration, not architectural. Test count 315 → 326 (+11 BC tests).
The variant auto-includes `portfolio_baseline_bc` in `evaluate()`
when the artifact exists at
`data/models/portfolio_baseline_bc/model.joblib`, mirroring the
FEAT-30 supervised-ranker pattern from v2.

---

## 2026-05-07 (D4-b multi-cycle yfinance) — RESEARCH: 4-year regime sweep confirms NO_GO; D2 invalidated; "low-DD selector" was synthetic-only

**Issue:** [#5](https://github.com/l2code/trading-bot-rl/issues/5) D4-b
**PR (cache enabler):** [#84](https://github.com/l2code/trading-bot-rl/pull/84) (FIX-#83 range-coverage cache match)
**Diary:** [`2026-05-07_d4b_multi_cycle_yfinance_NO_GO.md`](diary/2026-05-07_d4b_multi_cycle_yfinance_NO_GO.md)

After FIX-#83 made the validate path economical (2021 went from 7+
min hung → 35 sec; full 4-year loop in <1 min), ran canonical-
yfinance Phase-24 gate on all existing artifacts for 2021/2022/2023/2024.

**16 of 16 cells NO_GO.** Every trained selector loses to random
in every year tested. Three multi-year reproducibility checks:

  - **set_ranker DD vs random's:** set_ranker has HIGHER DD in 4 of 4
    years. PR #75's "lowest DD of any policy" claim refuted with
    multi-year evidence — was 100% synthetic-data artifact, not even
    regime-fragile.
  - **set_ranker per_strat distinctness from first_fired:** survives
    in 4 of 4 years. The DeepSets architecture IS doing real selection
    beyond priority order across regimes — it's just consistently
    unprofitable on this data.
  - **masked-PPO bit-identity vs first_fired:** holds at composite-
    score level across all 4 years; tiny float drift on absolute
    return in 2023/2024 but per_strat distributions match. PR #71's
    finding is regime-stable.

**Implications:**

  - **D2 invalidated.** No production-grade low-DD selector in the
    current artifact set; D2 should not be ratified as written.
  - **Phase 1 NO_GO is regime-stable.** Default-hyperparam masked-PPO
    collapses to first_fired across 2021/2022/2023/2024, not just 2022.
  - **The slate framing on yfinance starter_equities is structurally
    exhausted at default-hyperparam selector-class compute.** Path
    forward (if any): #32 chronological v3 (different decision
    shape), #4 WRDS canonical (different data tier), or close v002
    selector research direction entirely.

State: 292 tests passing. Main is on canonical-yfinance multi-year
verdicts. No live deploys, no in-flight compute.

## 2026-05-07 (FIX-#78 contamination + 4-step recovery) — STRUCTURAL: every Phase 1 step 1 → Phase 3 step 1 PR-1c verdict was synthetic; 6 diaries corrected

**Issue:** [#78](https://github.com/l2code/trading-bot-rl/issues/78)
**PRs:** [#79](https://github.com/l2code/trading-bot-rl/pull/79) (step 1 guardrail), [#80](https://github.com/l2code/trading-bot-rl/pull/80) (step 2 YAML), [#81](https://github.com/l2code/trading-bot-rl/pull/81) (step 3 rebaseline diary), this PR (step 4 corrections)
**Diary:** [`2026-05-07_d4_canonical_yfinance_rebaseline.md`](diary/2026-05-07_d4_canonical_yfinance_rebaseline.md)

**Project-wide research-integrity finding.** `validate_from_experiment`
silently defaulted to `synthetic_momentum` whenever the experiment YAML
omitted `data_provider` (which every selector_v002* YAML did). This
contaminated 6 diary verdicts (PR #70/#71/#72/#73/#74/#75) plus the
PR #76 threshold-sweep analysis: trained on yfinance, post-training
gate-evaluated on synthetic. Smoking gun: `buy_and_hold_return(SPY,
2022)` reported +0.187 (synthetic) vs real -0.186 (SPY actually fell
~19% in 2022).

**4-step recovery:**

  1. **Step 1 (PR #79):** added `--test-start` / `--test-end` /
     `--data-provider` CLI flags PLUS a defensive guardrail —
     `validate_from_experiment` refuses to default to synthetic for
     selector-class variants unless `allow_synthetic_validation=True`.
     3 new unit tests cover the branches.
  2. **Step 2 (PR #80):** added `data_provider: yfinance_daily` to
     the two selector YAMLs (`ppo_selector_v002.yaml`,
     `ppo_selector_v002_masked.yaml`).
  3. **Step 3 (PR #81):** re-ran the canonical 2022 Phase-24 gate on
     real yfinance for every existing artifact. Wrote rebaseline
     diary documenting what survives and what doesn't.
  4. **Step 4 (this PR):** prepended `[CORRECTION]` banners to all 6
     contaminated diaries pointing at the rebaseline; updated
     CLAUDE.md §2 status table; updated docs/scorecard.md
     research-state table with strikethrough on superseded verdicts.

**What survives the synthetic→yfinance correction:**

  - Masked-PPO is bit-identical to `selector_baseline_first_fired` on
    yfinance just like on synthetic. PR #71's "RL machinery learned
    only first_fired" finding holds on real data.
  - Phase 1 closure verdict (PR #73 NO_GO at strict acceptance) stands.
  - Set_ranker's per_strat distribution distinct from first_fired
    survives ([946, 34, 54] vs [1127, 32, 40] on yfinance) — the
    DeepSets architecture is doing real selection beyond priority order.

**What does NOT survive:**

  - PR #74's "first trained policy on yfinance to clear all 5
    Phase-24 gate metrics vs random" — false on yfinance (1-of-5,
    2 material regressions).
  - PR #75's "lowest max_DD of any trading policy" — false on
    yfinance (set_ranker DD 0.832 vs random's 0.704; HIGHER not
    lower). The apparent low-DD property was a synthetic artifact.
  - PR #75's "highest composite score 0.7331 of any policy ever
    tested" — false on yfinance (-0.193, below random's -0.185).

**Operator implications:**

  - "Path C: ship low-DD selector behind operator approval" —
    invalidated. Set_ranker is NOT low-DD on real yfinance.
  - "D2: ratify set_ranker as SHADOW_ONLY shadow research lane" —
    invalidated as written. Either close or rewrite to ratify the
    "selection beyond priority order" property without the low-DD claim.
  - "D4 multi-cycle WF on existing artifacts" — even more important
    now. Run 2021/2022/2023/2024 on real yfinance to see whether
    random's apparent yfinance win generalizes or is regime-specific.

**State after this PR:** main is on canonical-yfinance numbers. CLAUDE.md
§2 reflects rebaselined verdicts. 6 diaries carry `[CORRECTION]` banners
pointing at the rebaseline. 289 tests still passing. No live deploys, no
in-flight compute. D4 multi-cycle becomes the next concrete piece of work.

## 2026-05-06 (Phase 3 step 1 PR-1b) — RESEARCH: set ranker B2 stabilization NO_GO at gate; PR-1's lucky checkpoint not reproducible, but DD-lower-than-random is

**Issue:** [#34](https://github.com/l2code/trading-bot-rl/issues/34) (PR-1b stabilization)
**Run:** local on Loki — 3 seeds × 30 epochs × 15s/seed; with feature standardization + LR warmup + grad clip
**Diary:** [`2026-05-06_v002_set_ranker_stabilized_NO_GO.md`](diary/2026-05-06_v002_set_ranker_stabilized_NO_GO.md)

Operator picked B2 before B1 after PR-1 SHADOW_ONLY: stabilize
supervised training before spending Kaggle on PPO. Implemented
gradient clipping, LR warmup, multi-seed loop, top-1 diagnostics,
save-best-by-val. The operator's listed asks alone did NOT
stabilize training — MSE still exploded into 10^9-10^11 across all
3 seeds. Root cause caught and fixed during the run: ctx features
include raw frame fields (prices, dollar volumes) that span many
orders of magnitude; linear layers in phi/rho produce massive
predictions; MSE on a target in [-1, 1] dominates everything else.
Fix: feature standardization. Train-set mean/std on ctx + slot
blocks; transform inputs at train time; persist stats with bundle
and apply at inference.

After standardization, training is monotonically decreasing across
all 3 seeds (val_loss in [0.0758, 0.0815]; top-1 in [0.74, 0.76]).
**Tight clustering — reproducible.**

**Three honest findings:**

  1. PR-1's "DD lower than random" claim **is real and
     reproducible**. PR-1 reported DD 0.1539; PR-1b reports DD
     0.1356 (even better). The architecture genuinely produces a
     low-DD trading policy on yfinance — first such result in the
     project's history.
  2. PR-1's "5-of-5 gate-pass on absolute return + sharpe" claim
     is **NOT reproducible**. PR-1's high-return numbers came from
     a take-everything operating point (94% take rate) that the
     unstable mid-training checkpoint happened to land on.
     Stable training converges to a selective operating point
     (53% take rate) instead.
  3. The architecture's stable operating point is **genuinely
     distinct from first_fired**. PR-1b per_strat [646, 75, 218]
     is 55% fewer Momentum trades than first_fired's [1423, 79,
     278] — the slate framing IS doing real selection.

**Verdict: NO_GO** at the per-metric Phase-24 gate (1-of-5
improved, 2 material regressions on absolute return -0.96 and
sharpe -1.20). But composite score 0.7331 is the highest of any
policy ever tested on this benchmark — composite formula's DD +
turnover components compensate for the absolute-return regression.
A composite-vs-per-metric framework gap is implicitly filed.

PR-2 (sb3 features-extractor + Kaggle PPO retrain) **still
justified**: PPO optimizes total accumulated reward, which weighs
differently than per-slot regression on risk-adj return. The
supervised checkpoint's selective operating point isn't the only
thing the encoder can express. Strict acceptance for PR-2 set at
the gate level (beat random ≥2 of 5; not bit-identical to
first_fired; ≥2 of 3 seeds usable) — not at PR-1's lucky-checkpoint
level.

286 tests still passing (no test changes). Trainer code now has:
multi-seed loop, LR warmup, grad clip, feature standardization,
top-1 diagnostics, per-seed summary in artifact metadata.

## 2026-05-06 (Phase 3 step 1 PR-1) — RESEARCH: set/slate encoder cheap diagnostic SHADOW_ONLY (first trained policy to beat random on every gate metric)

**Issue:** [#34](https://github.com/l2code/trading-bot-rl/issues/34) (PR-1 only)
**Run:** local on Loki — torch DeepSets-style encoder, ~30k packs, fit 11s, early-stopped at epoch 21 (val_loss 41.46)
**Diary:** [`2026-05-06_v002_set_ranker_SHADOW_ONLY.md`](diary/2026-05-06_v002_set_ranker_SHADOW_ONLY.md)

**Phase 3 has its first positive result.** Reversed the framing
hypothesis from PR #73's NO_GO: the slate framing's prior collapse
to first_fired was an MlpPolicy artifact, not a feature-level
exhaustion. A permutation-equivariant DeepSets encoder over the
same yfinance data produces a policy that:

  - Beats `selector_baseline_random` on **5 of 5 Phase-24 gate
    metrics** including LOWER max_drawdown (0.1539 vs 0.1557).
    First trained policy on yfinance to clear this bar.
  - Beats the FEAT-7 HistGB ranker on **5 of 5** (return +4.17,
    sharpe +3.41, PF +1.77, DD -0.035, take_rate +0.37).
  - Genuinely escapes first_fired's [1423, 79, 278] pattern with
    [1325, 73, 289] — a real reallocation, not a tie-break artifact.

Operator's cheap-diagnostic acceptance (from issue scope before
the run): "material improvement OR distinct strategy distribution
from first_fired" — both conditions met. PR-2 (sb3 features-
extractor wiring + Kaggle MaskablePPO retrain) is justified.

Honest caveats in the diary: composite score 0.7066 trails random's
0.7186 because of saturation and turnover-component formula
artifacts, even though every individual metric improves; training
loss diverged after epoch 23 (PR-2 should add gradient clipping +
LR warmup); single seed. None of these change the gate-output
verdict, but PR-2 is the multi-seed information.

Architecture: \`src/rl_swing/rl/agents/slate_encoder.py\` (DeepSets
phi → sum/max/mean pool → rho_slot per-slot logit + rho_skip head;
slot weights shared, aggregate order-invariant — cannot trivially
encode "always pick slot 0"). Scorer: \`SetRankerSelectorScorer\`
(lazy torch load + masked argmax). Trainer: \`scripts/train_set_ranker.py\`
(per-slate MSE on realized risk-adj return, skip target = -best-
signal CF). Wiring: auto-included in v002 / v002_masked evaluate()
on the "set_ranker" tag when the artifact exists.

286 tests passing (282 + 4 new: permutation-equivariance, mask
zeroing, empty-pack short-circuit, registry pickup).

## 2026-05-06 (Phase 1 closure) — RESEARCH: masked-PPO FEAT-7 tie-breaker NO_GO; v002 selector closes for further compute

**Issue:** [#29](https://github.com/l2code/trading-bot-rl/issues/29) (decisive evidence)
**Run:** Kaggle private kernel `crazypenguin/rl-swing-v002-masked-feat7-tiebreak` — 3 seeds × 500k timesteps, 45 min wall-time
**Diary:** [`2026-05-06_v002_masked_feat7_tiebreaker_NO_GO.md`](diary/2026-05-06_v002_masked_feat7_tiebreaker_NO_GO.md)

Phase 1 closure. Per the operator's "Path B, single tie-breaker
run, strict acceptance, information gain not rescue mission"
framing (chat 2026-05-06), retrained masked-PPO on the FEAT-7
observation (+9 dims of pack-level + per-slot agreement features).
Strict acceptance — all five required:

  [1] beat random on Phase-24 gate              → PASS (4-of-5)
  [2] beat first_fired on absolute composite    → FAIL (tied 0.690470)
  [3] NOT bit-identical to first_fired          → FAIL (every metric bit-equal)
  [4] nontrivial strategy distribution           → FAIL ([1423, 79, 278] = first_fired's exactly)
  [5] ≥2 of 3 seeds usable                       → FAIL (1 of 3, and seed 33 is take-everything)

**4 of 5 fail. NO_GO.**

The trained masked-PPO with FEAT-7 features is bit-identical to
selector_baseline_first_fired on every metric to 6 decimals — same
numbers as the pre-FEAT-7 masked-PPO (PR #70 SHADOW_ONLY). The +9
informative dims did nothing for PPO at default ent_coef=0.01.
Seed 11 actually did *worse* than its pre-FEAT-7 self (briefly took
trades at step 50k then collapsed; vs prior run's productive
checkpoint at step 300k). Seed 22: stayed all-skip across 500k
steps. Seed 33: take-everything from step 50k, never improved.

**v002 selector with default-hyperparam MlpPolicy is structurally
exhausted on yfinance.** Both the unmasked and masked variants
converge to first_fired regardless of feature set. Adding more
features without changing the model class or hyperparameters won't
fix this — the architecture trivially encodes "always pick slot 0
of fired slots" and that's the local optimum it lands in.

Per the diary's recommendation (operator's pre-agreed Path B):
close v002 PPO for further default-hyperparam compute. Pivot to
Phase 3 architectural work, ordered: #34 set/attention slate
encoder → #32 portfolio-aware chronological v3 → optionally #27
Optuna formally if anyone wants the "tuning alone cannot fix this"
negative result on record.

The supervised ranker baseline (FEAT-30 + FEAT-7) stays available
as a comparison floor for Phase 3 work; even though it's NO_GO vs
random, it carries the slight selectivity that PPO doesn't.

## 2026-05-06 (Phase 1 step 3) — RESEARCH: cross-strategy agreement features NO_GO; gap to random halved but not flipped

**Issue:** [#7](https://github.com/l2code/trading-bot-rl/issues/7) (agreement-features half closed; pairwise-one-hot + multi-day variants stay open)
**Run:** local on Loki — sklearn HistGB re-fit on 28,219 rows × 36 features (was 27); fit in 1.1s; in-sample MSE 0.01673
**Diary:** [`2026-05-06_v002_feat7_agreement_features_NO_GO.md`](diary/2026-05-06_v002_feat7_agreement_features_NO_GO.md)

Phase 1 step 3 — operator-authorized "Path B" feature engineering
before HPO, motivated by the bit-identity finding from PR #71.
Added the operator-specified pack-level + per-slot agreement
features (`pack_n_fired`, `pack_signal_max/mean/std/gap_top2`,
`pack_all_fired`, `pack_same_symbol_strategy_agreement`, per-slot
`is_top_signal` and `rank_by_signal`). Wired into both the
`MultiStrategyObservationBuilder` (so the next masked-PPO
retrain sees them) and the supervised ranker's
`PER_SLOT_FEATURE_NAMES`. Pure-function helpers in a new module
`agreement_features.py` so train and inference share the same
implementation.

Re-trained the cheap supervised ranker (28k rows × 36 features,
fit in 1.1s) and ran rl-swing validate to compute the leverage
test. Result: **marginal improvement, not material.** Composite
score 0.7107 → 0.7145 (Δ +0.0038); sharpe +4.376 → +4.645
(Δ +0.27); max_DD 0.1963 → 0.1889 (Δ -0.0074); n_trades 1088 →
1031 (slightly more selective). Gap to `selector_baseline_random`
(score 0.7186) closed from 0.0079 to 0.0041 — about half. But
the same 3 material regressions vs random remain (return -1.35,
sharpe -2.46, profit_factor -1.05). Verdict: NO_GO.

The features carry real-but-modest signal — they did move the
ranker along the right axis. But not enough to flip the gate. Per
the operator's pre-agreed framing (chat 2026-05-06): "If ranker
improves materially, PPO/HPO becomes worth run budget." A 0.5%
composite-score gain that doesn't flip any gate metric is hard to
call material. The diary's recommendation is **Path A** (pivot to
Phase 3 architectural #34/#32 instead of a #27 Optuna burn). Path
B alternative — one masked-PPO Kaggle retrain on the new obs as a
tie-breaker — is the operator's call.

282 tests passing (276 + 6 new agreement-feature tests).

## 2026-05-06 (Phase 1, late evening) — RESEARCH: supervised ranker NO_GO — masked-PPO bit-identical to first_fired baseline

**Issue:** [#30](https://github.com/l2code/trading-bot-rl/issues/30) (supervised half closed; LinUCB sub-RFC stays open)
**Run:** local on Loki — sklearn `HistGradientBoostingRegressor` fit on 28,219 (pack × fired strategy) rows in 0.8s; in-sample MSE 0.01671
**Diary:** [`2026-05-06_v002_masked_supervised_ranker_NO_GO.md`](diary/2026-05-06_v002_masked_supervised_ranker_NO_GO.md)

Phase 1 step 2. Trained the supervised ranker baseline per the
operator's FEAT-30 scope (slate features + per-slot fields →
realized risk-adjusted return; argmax with skip threshold at 0).
Two findings:

1. **Ranker is NO_GO vs `selector_baseline_random`** — 3 material
   regressions (return -1.35, sharpe -2.73, PF -1.17). The ranker
   is genuinely selective (60.85% take rate vs random's 69.41%)
   but its discrimination is wrong-directional — it filters out
   winners along with losers. Total return drops to +0.82 vs
   random's +2.17.

2. **Masked-PPO is bit-identical to `selector_baseline_first_fired`.**
   Same score (0.690470), same return (+4.875414), same per_strat
   [1423, 79, 278] to 6 decimals. The "trained" Kaggle 500k×3
   masked-PPO learned exactly the deterministic-priority rule
   "take the lowest-index strategy that fired." This refines the
   masked-PPO SHADOW_ONLY verdict: the strong gate-pass (4-of-5
   improved vs random) is a property of the first_fired rule, not
   of the RL machinery. Addendum on the SHADOW_ONLY diary flags
   this; verdict still SHADOW_ONLY (gate output + tier rules
   unchanged), but the implied "PPO learned something" reading is
   wrong. The supervised ranker also doesn't beat masked-PPO
   (NO_GO with 3 material regressions in that direction too), so
   the operator's "RL machinery isn't earning its complexity"
   framing on #30 doesn't fire either.

The much-more-important finding: **both selector architectures
(masked-PPO + supervised ranker) fail the Phase-24 gate vs random.**
Random's apparent dominance is almost entirely from lower max_DD
(0.156 vs 0.196-0.199). Random's lower DD is from skipping ~30% of
fired packs uniformly, which acts as portfolio-level noise
reduction. None of the trained policies beat that.

Recommendation per the diary, ordered by EV:

1. **#7 cross-strategy agreement features.** Highest-leverage
   intervention. The current observation gives the policy per-slot
   features but not "do strategies agree on this (symbol, date)."
   Cheaper than architectural work and could shift both PPO and
   ranker verdicts.
2. **#8 Optuna sweep against masked-PPO** with a *tightened*
   acceptance criterion: must clear the gate vs random AND beat
   `first_fired` on absolute composite score (currently bit-
   identical to PPO). Operator's #8 framing was "use ranker as
   context, not in isolation"; this diary IS that context.
3. **#34 set/attention or #32 chronological v3** only if (1) and
   (2) both fail.

## 2026-05-06 (Phase 1, evening) — RESEARCH: masked v2 SHADOW_ONLY — masking unsticks all-skip but seed-stability is poor

**Issue:** [#29](https://github.com/l2code/trading-bot-rl/issues/29)
**PRs:** scaffold [#67](https://github.com/l2code/trading-bot-rl/pull/67), plumbing [#69](https://github.com/l2code/trading-bot-rl/pull/69)
**Run:** Kaggle private kernel `crazypenguin/rl-swing-v002-maskableppo-phase-1-private` (43.8 min wall-time, 3 seeds × 500k timesteps + 3 evals each)
**Diary:** [`2026-05-06_v002_masked_SHADOW_ONLY.md`](diary/2026-05-06_v002_masked_SHADOW_ONLY.md)

First Phase-1 result. The post-Phase-0 unmasked v002 was trapped at
[0,0,0] all-skip across all 30 evaluation points. With sb3-contrib
MaskablePPO + `action_masks() = [True, fired_slot_0..N]` on the
otherwise-identical setup (same seeds 11/22/33, same windows, same
ent_coef=0.01, same reward), the trained alias diversifies across
all three strategies (per_strategy_take_counts=[1423, 79, 278]) and
clears the Phase-24 gate cleanly: 4 of 5 metrics improved vs
selector_baseline_random (return +4.88 vs +2.17, sharpe +7.46 vs
+7.11, PF 3.51 vs 3.19, take_rate 0.996 vs 0.694), no material
regressions (DD +0.0435, just under the 0.05 threshold). Phase-24
gate output: GO.

But the verdict is **SHADOW_ONLY**, not GO, for two reasons either
of which alone is sufficient: (1) data tier is exploratory yfinance
— yfinance can never earn decision-grade GO per CLAUDE.md §3.5;
WRDS replication (#4) gates promotion. (2) eval_history shows the
cross-seed best alias comes from seed 11 step 300k, where the
policy briefly escaped all-skip (1613 trades, val 0.5124) and then
**collapsed back to all-skip** for the remaining 200k steps. Seed
22 stayed bit-identical to all-skip across all 500k steps. Seed 33
broke into take-everything from step 50k onward. So masking is
necessary but not sufficient — only 1-of-3 seeds found a productive
policy and even that one was transient. Default ent_coef=0.01 is
likely too low; a textbook entropy-collapse signature.

Recommendation per the diary: do not spend more compute on masked
v2 at default hyperparams. Per CLAUDE.md §4 Phase 1: next is #30
supervised ranker baseline (task #47, local-only, no Kaggle quota
concern), then #8 Optuna sweep on ent_coef + lr against the masked
variant (acceptance: per_strategy_take_counts diversifies on
≥3-of-5 seeds). v2 unmasked PPO is closed for further compute.

## 2026-05-06 (late evening) — RESEARCH: Phase 0 FINAL closure — DRAFT diaries promoted to FINAL_NO_GO

**Issues:** [#2](https://github.com/l2code/trading-bot-rl/issues/2), [#3](https://github.com/l2code/trading-bot-rl/issues/3)
**Runs:** `crazypenguin/rl-swing-v001-rerun-audit-v2`, `…-v001-phase0-final`, `…-v002-rerun-audit-v2` (all complete; v002-phase0-final-retry never pushed due to Kaggle 5-session quota — audit-v2 carries the same FIX-AUDIT-V2/V3 patch set and is the canonical v2 final)
**Diaries:** [`2026-05-06_v001_filter_post_phase0_FINAL_NO_GO.md`](diary/2026-05-06_v001_filter_post_phase0_FINAL_NO_GO.md), [`2026-05-06_v002_selector_post_phase0_FINAL_NO_GO.md`](diary/2026-05-06_v002_selector_post_phase0_FINAL_NO_GO.md)

Pulled the audit-v2 / phase0-final validation summaries and
confirmed the qualitative verdict from the DRAFT entries holds
under the refined daily-P&L metrics (FIX-#36 idle-day window from
test_start/end, FIX-#57 trading-day calendar from bars). v1
trained PPO is bit-identical to `baseline_always_take_100` (score
0.6910, return +4.5134, sharpe +7.351, dd 0.1975); strongest
baseline `baseline_random` scores 0.7352 with dd 0.0904 — the
trained model fails the Phase-24 gate by a +0.107 max-drawdown
material regression. v2 trained PPO is bit-identical to
`selector_baseline_always_skip` ([0,0,0] per-strategy takes,
score 0.3250 across all 30 evaluation points); strongest baseline
`selector_baseline_random` scores 0.7186 — three material gate
regressions (return, sharpe, profit_factor). Both DRAFT diaries
renamed to `…_FINAL_NO_GO.md` with refined numbers and updated
banners; pre-Phase-0 predecessor PROVISIONAL banners replaced
with SUPERSEDED banners; CLAUDE.md §2 status table and
docs/scorecard.md research-state table updated to FINAL_NO_GO.
**Phase 0 is now fully closed.** Phase 1 leads with #29
MaskablePPO for v2, #30 supervised ranker baseline, #8 Optuna
sweep. No more compute on default-PPO v1/v2.

## 2026-05-06 (evening) — RESEARCH: Phase 0 closure DRAFT — both variants NO_GO under default PPO hyperparams

**Issues:** [#2](https://github.com/l2code/trading-bot-rl/issues/2), [#3](https://github.com/l2code/trading-bot-rl/issues/3)
**PRs:** all FIX-#22 through #62 merged
**Diaries:** the DRAFT entries linked here have been renamed to `…_FINAL_NO_GO.md` (see entry above)

After all 16 P1+P2 simulator/evaluation fixes from two operator
audits + one parallel pass were merged, both v1 and v2 were
re-trained on Kaggle (audit-bundle commit `eb32fba`). The
qualitative verdict is **NO_GO for both** under default PPO
hyperparams.

- **v1 collapse:** trained PPO is bit-identical to
  `baseline_always_take_100` (same trades, returns, sharpe, DD).
  Loses to `baseline_always_take_50` and `baseline_random` on
  the gate (material DD regression).
- **v2 collapse:** trained PPO is bit-identical to
  `selector_baseline_always_skip` (n_trades=0). Material
  regression on every metric vs `selector_baseline_random`. The
  collapse is a structural action-space issue (illegal-action
  penalty + low ent_coef), not reward calibration — wants action
  masking (#29 MaskablePPO).

The audit-v2 / phase0-final runs (still in flight) refine metrics
slightly via FIX-#56 / #57 / #61 / #62 but do not change the
qualitative verdict. Diary entries are DRAFT until those land
and we replace numbers in place.

**Phase 1 implications:** the diagnoses point hard at the next
moves. (1) #29 MaskablePPO directly targets v2's structural
collapse mode. (2) #30 supervised baseline tells us whether the
features have discriminating signal at all without RL exploration
noise. (3) #8 Optuna sweep tests whether v1's instant collapse to
always-take_100 is fixable via entropy/lr. Operator-blessed
sequence: #29 first, #30 second, #8 third. Do not burn more PPO
run budget on v1 default-hyperparam or v2 without masking.

## 2026-05-06 — OPS: CI switched to workflow_dispatch only (manual-only)

**Issue:** [#45](https://github.com/l2code/trading-bot-rl/issues/45)
**PR:** (this PR)

Operator request: stop CI from auto-running on push/PR. The workflow
ran across 3 Python versions per push, costing GitHub Actions
minutes on every commit including doc-only changes. Local
verification is the merge gate per CONTRIBUTING.md §6 anyway, so
the auto-trigger was duplicative. CI remains available as a
catch-net the operator can run manually from the UI.

## 2026-05-06 — FIX: P1 walk-forward warmup; first ~200 test days no longer use degraded features

**Issue:** [#24](https://github.com/l2code/trading-bot-rl/issues/24)
**PR:** [#44](https://github.com/l2code/trading-bot-rl/pull/44)

Third P1 simulator/eval fix. Both `walk_forward.py` and
`trainer._build_env` now load bars with 1.5 trading years of
warmup before the requested window. Long-lookback features (sma_200,
return_60d, atr_pct_14, etc.) populate before the in-window region
and frames are filtered to the eval window before candidate
generation. Three new unit tests in
`tests/unit/test_warmup_helpers.py`. One P1 (#36 portfolio
equity-curve eval) and one P2 (#26 hindsight skip-CF) remain
before v1/v2 re-runs can replace the PROVISIONAL diary entries.

## 2026-05-06 — FIX: P1 round-trip cost charged 2x per-side (fixed)

**Issue:** [#23](https://github.com/l2code/trading-bot-rl/issues/23)
**PR:** [#43](https://github.com/l2code/trading-bot-rl/pull/43)

Second P1 simulator fix. `cost_model.cost_bps()` documented as
per-side; simulator was subtracting once. Now multiplies by 2 for
round-trip explicitly. Combined with FIX-22 (size scaling), a 10%
sized trade with 50bps per-side cost on a flat asset now correctly
produces -10bps portfolio drag (was -5bps before FIX-23, was -50bps
before FIX-22+23).

## 2026-05-06 — OPS: roadmap restructured around operator's "Next Stage" framework

**Issues filed:** [#36](https://github.com/l2code/trading-bot-rl/issues/36) (P1 portfolio equity-curve eval), [#37](https://github.com/l2code/trading-bot-rl/issues/37) (promotion matrix), [#38](https://github.com/l2code/trading-bot-rl/issues/38) (baseline-dominance gate), [#39](https://github.com/l2code/trading-bot-rl/issues/39) (ablation harness), [#40](https://github.com/l2code/trading-bot-rl/issues/40) (shadow mode), and earlier #29–#35 from the RL-design review.

Operator review reframed the project trajectory as a phased
sequence: `fix correctness → prove baselines → run matrix →
ablate → shadow → paper`. Each phase gates on the previous
phase's evidence; more runs only have value as controlled
experiments inside the matrix harness, not as brute-force
training.

CLAUDE.md §4 rewritten to reflect this. Phase 0 (P1 simulator
fixes + v1/v2 re-runs) now explicitly gates all post-Phase-0
work. Earlier impact-per-effort tier ordering preserved as
"cheap diagnostics" runnable in parallel.

A new P1 surfaced from the same review: portfolio equity-curve
evaluation (#36). The current `validation_composite_score`
sums per-trade returns and computes Sharpe / DD on the trade
sequence — not on a date-ordered daily P&L. Adds to the
PROVISIONAL banner scope.

## 2026-05-06 — FIX: P1 size_pct now scales realized portfolio return

**Issue:** [#22](https://github.com/l2code/trading-bot-rl/issues/22)
**PR:** [#28](https://github.com/l2code/trading-bot-rl/pull/28)

First of the P1 simulator fixes. `ExecutionSimulator` now returns
`return_pct` as the portfolio contribution (sized + cost-net), not
the asset's standalone percent return. New regression test asserts
`take_25` and `take_100` produce proportionally different returns.
Backward-compat `raw_return_pct` alias preserved. Two more P1s
queued (#23 round-trip costs, #24 WF warmup) plus the new #36
portfolio equity-curve evaluation.

## 2026-05-06 — STRUCTURAL: 5 simulator/evaluation bugs identified by code review; current verdicts marked PROVISIONAL

**Issues filed:** [#22](https://github.com/l2code/trading-bot-rl/issues/22) (P1 size-scale), [#23](https://github.com/l2code/trading-bot-rl/issues/23) (P1 round-trip cost), [#24](https://github.com/l2code/trading-bot-rl/issues/24) (P1 WF warmup), [#25](https://github.com/l2code/trading-bot-rl/issues/25) (P2 selector runtime), [#26](https://github.com/l2code/trading-bot-rl/issues/26) (P2 hindsight skip-CF)

Operator code review after the v002 NO_GO and per-strategy EV diary
landed identified five real issues affecting both the simulator
(`return_pct` not scaled by `size_pct`; round-trip cost charged
once despite per-side docstring) and walk-forward evaluation
(no lookback warmup, so first ~200 days of test window run on
degraded long-lookback features). Two additional v2-specific
issues: selector_v002 not wired into the runtime DecisionPipeline,
and the skip-reward counterfactual uses hindsight-best (max-over-
noise bias). All three current diary entries (v001 NO_GO, v002
NO_GO, per-strategy EV) marked PROVISIONAL pending the P1 fixes.
Optuna sweep (#8) paused — running on a broken simulator would
burn compute. Sequence: P1 fixes (#22/#23/#24) → re-run v1 and v2
with corrected metrics → then resume #8 with confidence the gate
output is meaningful.

## 2026-05-06 — OPS: separate rolling findings (CHANGELOG.md) from CLAUDE.md operating brief

**Issue:** [#19](https://github.com/l2code/trading-bot-rl/issues/19)
**PR:** [#20](https://github.com/l2code/trading-bot-rl/pull/20)

CLAUDE.md was bloating with per-experiment narrative. Split: CLAUDE.md
stays a stable operating brief (variant-status table, rules,
debts); this CHANGELOG.md absorbs chronological findings; per-
experiment artifacts stay under `research/diary/`. CONTRIBUTING.md
§11 now codifies the rule (append to changelog on merge).

## 2026-05-06 — FEAT: hyperparam-override CLI plumbing for kaggle sweeps

**Issue:** [#28](https://github.com/l2code/trading-bot-rl/issues/28) (closed)
**PR:** [#21](https://github.com/l2code/trading-bot-rl/pull/21)

`scripts/kaggle_run.py` now accepts `--hyperparam-overrides='{...}'`
which propagates via env var into `kaggle_train.py` and finally
into `train_from_experiment(hyperparam_overrides=)`. Override dict
merges over `cfg.hyperparams` (override wins). Prerequisite for
the Optuna entropy sweep (#8) — but blocked by P1 simulator fixes.

## 2026-05-06 — RESEARCH: per-strategy training-EV analysis (PARTIAL-H2)

**Issue:** [#15](https://github.com/l2code/trading-bot-rl/issues/15)
**PR:** [#18](https://github.com/l2code/trading-bot-rl/pull/18)
**Diary:** [`2026-05-06_per_strategy_training_ev.md`](diary/2026-05-06_per_strategy_training_ev.md)

Pure data analysis (no RL) to test whether v2's "Momentum specialist"
collapse is rational on training data (H2) or pure entropy collapse
(H1). Result: PARTIAL-H2. Momentum has the highest mean risk-
adjusted return on training data (+0.327 vs Breakout +0.279 vs RSI
+0.151), so preferring it is rational. But specializing to *only*
Momentum is irrational — Breakout has 85% of Momentum's EV across
8,411 candidates that the trained model ignored. The collapse has
both a rational direction (H2) and an irrational severity (H1).
This refines the success criterion for #8: not just "trained beats
baseline" but "per_strategy_take_counts shows real diversification."
Filed #17 (take_all_fired baseline) as a parallel diagnostic.

## 2026-05-06 — RESEARCH: v002 selector NO_GO on yfinance starter_equities

**Issue:** [#3](https://github.com/l2code/trading-bot-rl/issues/3)
**PR:** [#16](https://github.com/l2code/trading-bot-rl/pull/16)
**Diary:** [`2026-05-06_v002_selector_NO_GO.md`](diary/2026-05-06_v002_selector_NO_GO.md)

500k×3 Kaggle run completed. Phase-24 gate returns NO_GO twice:
4-of-5-improved-but-material-DD-regression vs strongest selector
baseline (random); 1-of-5-improved-with-3-regressions vs v1 trained.
Notable wrinkle: v2 collapsed *differently* from v1 — to a Momentum
specialist (`per_strategy_take_counts = [323, 0, 0]`) rather than
"always take everything." Random selector beats trained on composite
score (0.7037 vs 0.6665), the textbook entropy-collapse signature.
Both variants now NO_GO under default PPO hyperparams; the framing
(filter vs selector) is not the lever.

## 2026-05-06 — OPS: codify critical self-review pass before merge

**Issue:** [#12](https://github.com/l2code/trading-bot-rl/issues/12)
**PR:** [#14](https://github.com/l2code/trading-bot-rl/pull/14)

CONTRIBUTING.md §7 + CLAUDE.md §3.7. Self-review checklist captured
verbatim from PR #10's live demonstration: ruff on touched files,
test counts in docs match reality, no aspirational tooling claims,
`Closes #N` only for fully-met AC, ambiguous design choices flagged
in code AND filed, no `__pycache__`/cache files in diff. Self-review
captured as a PR comment so the audit trail is visible. PR #10's
self-review caught 6 real issues; this codifies the practice as
permanent.

## 2026-05-06 — OPS: apply trading-bot2 SDLC lessons (foundational docs + acceptance gate)

**Issue:** [#1](https://github.com/l2code/trading-bot-rl/issues/1)
**PR:** [#10](https://github.com/l2code/trading-bot-rl/pull/10)
**Diary:** [`2026-05-06_v001_filter_loose_NO_GO.md`](diary/2026-05-06_v001_filter_loose_NO_GO.md)

Distilled the patterns from `SDLC_LESSONS_FOR_NEW_PROJECT.md` into
this repo: CLAUDE.md operating brief, CONTRIBUTING.md workflow
rules, docs/data_tiers.md, docs/acceptance_gates.md (≥2 of 5
metric improvement gate), docs/scorecard.md, issue templates.
First research diary entry written for the v1 loose run (NO_GO,
0 of 5 metrics improved). Acceptance gate module + 11 tests
landed but not yet wired into walk_forward output (filed as #11).
Self-review during this PR caught and fixed 6 real issues that
would have shipped uncaught (#13, #11 follow-ups filed during
the review).

## 2026-05-06 — STRUCTURAL: pluggable RL variant architecture + v2 multi-strategy selector

**PR:** `af86e06` (pre-issue-first discipline; legacy)

Refactored trainer + walk_forward to dispatch via a TrainingVariant
abstraction registered in the ComponentRegistry. Adding a new
variant is now a single new file in `rl_swing/rl/variants/` plus
a one-line entry in `configs/components/components.yaml`. v1 logic
extracted as `FilterV001Variant`; v2 multi-strategy selector
implemented as `SelectorV002Variant` (per-(symbol, date) decisions,
Discrete(N+1) action, no candidate dedupe so the agent sees the
full slate).

## 2026-05-06 — RESEARCH: v001 filter NO_GO confirmed at three intervention levels

**PR:** `c1ba1ed`, `f907dda`, `0f7c18b` (pre-issue-first discipline; legacy)

Three increasingly aggressive interventions (turnover penalty
0.02 → 0.30, skip mirror 0 → 1.0, candidate threshold loosening to
widen pool 309 → 477) all converged to "always take" — bit-identical
to baseline_always_take_100 across 30 evaluation points × 3 seeds.
Confirms the candidate-set-EV problem: the strategy stage produces
candidates with so much positive expected value that "always take"
is genuinely the EV-optimal policy under any reasonable reward.
The filter framing has no lift on this universe in this regime.
Recorded with verdict in `research/diary/2026-05-06_v001_filter_loose_NO_GO.md`.
