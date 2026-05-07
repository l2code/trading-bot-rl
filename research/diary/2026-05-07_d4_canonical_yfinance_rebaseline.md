# RESEARCH-78 — canonical 2022 yfinance rebaseline (FIX-#78 step 3)

**Date:** 2026-05-07
**Verdict:** **REBASELINE** — replaces the synthetic numbers in PR #70 / #71 / #72 / #73 / #74 / #75 / #76 diaries with real-yfinance numbers. **All five trained selectors are NO_GO vs random on real yfinance 2022.**
**Issue:** [#78](https://github.com/l2code/trading-bot-rl/issues/78) (step 3 of the 4-step recovery)
**Run:** local on Loki — `rl-swing validate` against the FEAT-7 `ppo_selector_v002_masked.yaml` with explicit `data_provider: yfinance_daily` (now structurally enforced by step 2's YAML edit + step 1's guardrail)
**Trainer commit:** existing artifacts trained pre-FIX-#78 (Kaggle masked-PPO from PR #73 tie-breaker; supervised HistGB FEAT-7; set_ranker PR-1b stabilized)

---

## Why this diary exists

Step 1 of FIX-#78 (PR #79) added a guardrail that made `validate_from_experiment` refuse to silently default to `synthetic_momentum` for selector variants. Step 2 (PR #80) added `data_provider: yfinance_daily` to the selector YAMLs. Step 3 — this diary — re-runs the canonical 2022 Phase-24 gate on real yfinance to produce the new ground-truth numbers that supersede the contaminated synthetic numbers from PR #70-#76.

## Side-by-side: synthetic (contaminated) vs yfinance (canonical)

| Policy | Synthetic 2022 (PR #75 et al) | yfinance 2022 (this rebaseline) |
|---|---|---|
| `selector_baseline_random` | score=0.7186  ret=+2.17  dd=0.156  trades=1241  per_strat=[839, 69, 333] | score=**-0.1849**  ret=**-0.70**  dd=**0.704**  trades=**839**  per_strat=**[600, 32, 207]** |
| `selector_baseline_first_fired` | score=0.6905  ret=+4.88  dd=0.199  per_strat=[1423, 79, 278] | score=-0.1999  ret=-0.85  dd=0.856  per_strat=[1127, 32, 40] |
| `selector_baseline_highest_signal` | score=0.6910  ret=+4.51  dd=0.197  per_strat=[1259, 116, 405] | score=-0.1999  ret=-0.84  dd=0.842  per_strat=[853, 53, 293] |
| `selector_baseline_supervised` (HistGB FEAT-7) | score=0.7145  ret=+0.82  dd=0.189  per_strat=[729, 84, 218] | score=-0.1891  ret=-0.77  dd=0.767  per_strat=[864, 19, 55] |
| `selector_baseline_set_ranker` (PR-1b stabilized) | score=**0.7331**  ret=+1.21  dd=**0.136**  per_strat=[646, 75, 218] | score=-0.1930  ret=-0.83  dd=**0.832**  per_strat=[946, 34, 54] |
| `ppo_selector_v002_masked` (FEAT-7 tie-breaker) | (synth: same as first_fired bit-identical) | score=-0.1999  ret=-0.85  dd=0.856  per_strat=**[1127, 32, 40]** |

**Buy-and-hold SPY 2022:** synthetic +0.187 (smoking gun) → yfinance -0.186 (matches reality: SPY -19% in 2022).

## Phase-24 gate output on yfinance (vs `selector_baseline_random`)

| Policy | Improved | Material regress | Verdict |
|---|---:|---:|---|
| `selector_baseline_first_fired` | 1 | 2 | NO_GO |
| `selector_baseline_highest_signal` | 1 | 2 | NO_GO |
| `selector_baseline_supervised` (HistGB) | **3** | 2 | NO_GO |
| `selector_baseline_set_ranker` (DeepSets) | 1 | 2 | NO_GO |
| `ppo_selector_v002_masked` (MaskablePPO) | 1 | 2 | NO_GO |

**Every trained selector is NO_GO vs random on yfinance.** Random itself has -0.18 composite / -0.70 return / 0.70 DD — and every fired-take policy makes things worse, not better.

## Findings — what survives, what doesn't

### Survives the synthetic→yfinance correction

- **MaskablePPO (PR #73 FEAT-7 tie-breaker artifact) is bit-identical to `selector_baseline_first_fired`** on yfinance just like it was on synthetic. Score -0.1999, return -0.8548, per_strat [1127, 32, 40] — both policies match to 6 decimals. The "trained masked-PPO learned only first_fired" finding from PR #71 was correct and is preserved on yfinance.
- **The slate framing's collapse to first_fired-class behavior** under default-hyperparam MlpPolicy. PR #73's "Phase 1 closed: NO_GO" is reaffirmed on real data.

### Does NOT survive the synthetic→yfinance correction

- **The "set_ranker has lowest DD of any trading policy" finding from PR #75 is FALSE on yfinance.** Synthetic: DD 0.136 (lower than synthetic random's 0.156). Yfinance: DD **0.832** (HIGHER than yfinance random's 0.704). The apparent low-DD property was an artifact of synthetic_momentum's structure, not a real architectural property of the DeepSets encoder.
- **The "first trained policy to beat random on every gate metric" finding from PR #74 is FALSE on yfinance.** That was 5-of-5 on synthetic; on yfinance the set_ranker is 1-of-5 with 2 material regressions. The supervised HistGB ranker actually does best (3-of-5) but is also NO_GO.
- **The "highest composite score 0.7331 of any policy ever tested" finding from PR #75 is FALSE on yfinance.** Yfinance set_ranker composite is -0.193, lower than random's -0.185.

### Reframed but not refuted

- **The set_ranker's per_strat distinctness from first_fired's pattern still holds.** Synthetic: [646, 75, 218] vs first_fired's [1423, 79, 278] — distinct. Yfinance: [946, 34, 54] vs first_fired's [1127, 32, 40] — also distinct (less so, but still real). The DeepSets architecture IS doing real selection beyond priority order; that selection just isn't profitable on real 2022 yfinance.

## What this means for Phase 1 + Phase 3

**Phase 1 closure verdict from PR #73 stands** — masked-PPO at default hyperparams collapses to first_fired regardless of features (synthetic or real). The diary entry's framing was right; the numbers were synthetic.

**Phase 3 step 1 PR-1 / PR-1b verdicts are MOSTLY OVERTURNED:**
- PR #74 PR-1 SHADOW_ONLY → should be **NO_GO** on yfinance.
- PR #75 PR-1b stabilized "lowest DD of any policy" → **false on yfinance**.
- PR #76 PR-1c threshold sweep → already documented NO_GO on yfinance via `--data-provider yfinance_daily` in the sweep script; that finding survives.

Step 4 will append [CORRECTION] entries to those diaries pointing here.

## Implications for the path forward

The operator's "Path C: ship the supervised set ranker as a low-DD selector behind explicit operator approval" — **invalidated**. The set_ranker is not low-DD on real yfinance.

D2 (ratify set ranker as SHADOW_ONLY shadow research lane) — **invalidated** as written. The set_ranker doesn't have the property D2 was meant to ratify. D2 should either be closed or rewritten to ratify the architecture's "produces a non-trivial selection pattern" property without the low-DD claim.

D4 multi-cycle WF — **even more important now**. Run 2021/2022/2023/2024 on real yfinance to see (a) whether random's apparent "win" on yfinance is consistent or regime-specific; (b) whether any policy beats random in any year; (c) whether the architectural "selection beyond priority" finding generalizes.

## Cross-references

- All affected verdicts (PR #70 / #71 / #72 / #73 / #74 / #75 / #76) — get [CORRECTION] entries via step 4.
- Step 1 (CLI flags + guardrail): PR #79
- Step 2 (YAML data_provider): PR #80
- Step 3 (this rebaseline diary): this PR
- Step 4 (corrections to old diaries + CHANGELOG/scorecard): upcoming PR
- Issue #78
