# RESEARCH-029b — masked-PPO retrain on FEAT-7 obs (Path B tie-breaker)

**Date:** 2026-05-06
**Verdict:** **NO_GO** (4 of 5 strict acceptance criteria fail; v002 masked-PPO direction closes for further compute)
**Issue:** [#29](https://github.com/l2code/trading-bot-rl/issues/29) (originally closed by PR #70 SHADOW_ONLY; this diary is the Path B tie-breaker requested by operator after PR #72 FEAT-7 ranker re-test was marginal-not-material)
**Variant:** `selector_v002_masked` on FEAT-7 obs (+9 dims vs prior masked-PPO run)
**Run:** Kaggle private kernel `crazypenguin/rl-swing-v002-masked-feat7-tiebreak`
**Trainer commit at run time:** `d43b88d` (post FEAT-7 merged)
**Wall time:** ~45 min, 3 seeds × 500k timesteps × n_envs=4

---

## Question

Per operator scope (chat 2026-05-06): one masked-PPO Kaggle private
retrain on the FEAT-7 agreement-features observation. **Information-
gain run, not a rescue mission.** Did the extra slate context help
PPO escape `first_fired`, or is v002 still capped by the framing?

**Strict acceptance — all five required, otherwise NO_GO:**

1. Beat `selector_baseline_random` on Phase-24 gate.
2. Beat `selector_baseline_first_fired` on absolute composite.
3. NOT bit-identical to `first_fired`.
4. Show nontrivial strategy distribution beyond priority ordering.
5. ≥2 of 3 seeds find usable policies (not a one-seed transient).

## Result

```
ppo_selector_v002_masked  score=0.690470  return=+4.875414  per_strat=[1423, 79, 278]
selector_baseline_first_fired  score=0.690470  return=+4.875414  per_strat=[1423, 79, 278]
selector_baseline_random  score=0.7186      return=+2.1697     per_strat=[839, 69, 333]
```

The trained masked-PPO is **bit-identical to `selector_baseline_first_fired`** on every metric to 6 decimals — same numbers as the *pre-FEAT-7* masked-PPO run (PR #70 SHADOW_ONLY). The extra slate context did nothing. **NO_GO.**

## Strict acceptance results

| Criterion | Result | Detail |
|---|---|---|
| [1] beat random on gate | **PASS** | 4-of-5 improved (return / sharpe / PF / take_rate); DD +0.0435 under 0.05 material; gate output GO. |
| [2] beat first_fired absolute | **FAIL** | composite 0.690470 vs first_fired 0.690470 — tied to 6 decimals. |
| [3] NOT bit-identical to first_fired | **FAIL** | every metric bit-equal; same per_strat counts. |
| [4] nontrivial distribution | **FAIL** | [1423, 79, 278] = exactly first_fired's pattern. |
| [5] ≥2/3 seeds usable | **FAIL** | seed 11: stayed all-skip after step 50k transient; seed 22: never escaped all-skip; seed 33: take-everything (best_val 0.4229, no improvement past step 50k). 1 of 3 usable. |

## Per-seed eval-history dynamics

| Seed | Best val | First non-zero n_trades | Sustained evals with trades | Notes |
|------|---------:|:-----------------------:|:---------------------------:|-------|
| 11 | 0.3250 | step 50_000 | 1 (only one eval) | Briefly took trades at step 50k, then collapsed to all-skip for the remaining 450k steps. **Worse than the pre-FEAT-7 run** (where seed 11 found peak val 0.5124 at step 300k). |
| 22 | 0.3250 | never | 0 | Stayed bit-identical to all-skip across all 500k steps. |
| 33 | 0.4229 | step 50_000 | 10 (every eval) | Take-everything from step 50k onward; never improved past 0.4229. Same as the pre-FEAT-7 seed 33. |

Seed 11 actually did **worse** with FEAT-7 features than without —
in the prior run it found a productive policy at step 300k (val
0.5124, 1613 trades); here it briefly took trades at step 50k,
then collapsed to all-skip. The agreement features did not give
PPO a path to a stable productive policy at default
`ent_coef=0.01`; if anything they made the early exploration
slightly worse (one possible reading: more dims → harder
exploration without compensating tuning).

## What this proves

- **The slate framing on yfinance is structurally exhausted at
  default PPO hyperparams.** The pre-FEAT-7 masked-PPO converged
  to `first_fired`. The FEAT-7 masked-PPO converged to `first_fired`.
  Adding +9 informative dims didn't change the destination.
- **The supervised ranker's marginal-not-material lift on FEAT-7
  features (PR #72) was the better-of-the-two responses to the
  new features.** PPO's training dynamics under default exploration
  can't even pick up the marginal signal that the ranker saw.
- **`selector_baseline_first_fired` is the de-facto ceiling for
  default-hyperparam selector PPO on yfinance.** All three seeds
  collapse to either first_fired (cross-seed alias picks this) or
  to take-everything; no seed found a productive selective policy.

This is the conclusive evidence the operator wanted from a single
information-gain run: **default-hyperparam masked-PPO on the v002
selector framing cannot escape first_fired on yfinance, with or
without agreement features.**

## What this does NOT prove

- That **#27 Optuna** would also fail. The Optuna sweep tunes
  exactly the hyperparam (`ent_coef`) that's most likely to be the
  bottleneck. Operator's pre-existing call (chat 2026-05-06): #27
  is gated on this run's verdict, and the verdict says don't run
  it. But the question "would Optuna unstick the collapse" remains
  formally open. If anyone wants the negative result on record,
  filing a follow-up issue is cheap.
- That **#34 set/attention encoder** would also fail. The slate-
  feature framing has been tested on (a) MlpPolicy at default
  hyperparams, (b) sklearn HistGB ranker. Neither carry slate-
  level structure beyond independent per-slot scoring; a set-
  permutation-invariant encoder is a different model class with
  different inductive biases.
- That the v0 supervised ranker is dominated. The ranker carries a
  small amount of selectivity (PR #72: composite 0.7145 with 1031
  trades vs PPO's 1780); under cost-stress conditions or
  out-of-sample windows that selectivity might matter. But it's
  still NO_GO vs random under the current Phase-24 gate.

## Known limitations / known-not-changing-the-verdict

1. **Three seeds.** A 5-10 seed re-run might catch a productive
   seed transient. But the cross-seed alias logic (FIX-#53) picks
   the best-by-val, and the best across seeds 11/22/33 was seed
   33's 0.4229 (worse than the pre-FEAT-7 best of 0.5124). The
   ceiling is 0.4229-ish at default hyperparams, well below
   first_fired's 0.6905.
2. **Default `ent_coef=0.01`.** Operator's #8 Optuna sweep tests
   exactly this. Out of scope per the operator's tie-breaker
   framing.
3. **Single test cycle.** 2022-only. Under different market
   regimes the priority ordering might be a worse rule and a
   genuinely-learning policy might pull ahead. Multi-cycle WF
   ([#5](https://github.com/l2code/trading-bot-rl/issues/5))
   would tell us; not blocking the decision.
4. **Exploratory tier.** WRDS canonical replication
   ([#4](https://github.com/l2code/trading-bot-rl/issues/4))
   could shift the calculus, but this NO_GO is structural —
   the policy is degenerating at the architecture level, not
   just at the noise level.

## What would change the verdict

- **None of the listed criteria would survive a hyperparameter
  fix that broke the first_fired attractor.** If a follow-up
  Optuna sweep or RecurrentPPO swap produced a masked-PPO with
  per_strategy_take_counts genuinely different from
  [1423, 79, 278] AND beat first_fired's composite, this NO_GO
  would be specific to default-hyperparam MlpPolicy and the
  framing decision would re-open.

## Recommendation

Per the operator's pre-agreed Path B framing: **close v002 PPO
(both unmasked and masked) for further default-hyperparam compute;
pivot to Phase 3 architectural work.**

Concrete next steps, ordered:

1. **#34 set/attention slate encoder** — the slate-shaped
   inductive bias is the most direct structural fix for
   "MlpPolicy converges to first_fired." A set-permutation-
   invariant encoder cannot trivially encode "always pick slot 0";
   the policy must learn a per-slot scoring rule that respects
   the slate structure. Ports cleanly into the masked-PPO + FEAT-7
   stack.
2. **#32 portfolio-aware chronological v3** — if #34 also fails,
   this is the bigger swing: full sequential RL over a portfolio
   state, not per-pack independent decisions. Higher EV, higher
   complexity.
3. **#27 Optuna** — formally not run yet. Operator can keep it
   filed as a "would tuning alone fix this" follow-up if the
   negative result is worth recording. Otherwise, close the issue
   with a link to this diary.

## Cross-references

- Predecessor (masked-PPO SHADOW_ONLY, pre-FEAT-7): [`2026-05-06_v002_masked_SHADOW_ONLY.md`](2026-05-06_v002_masked_SHADOW_ONLY.md) (with bit-identity addendum from PR #71)
- Predecessor (FEAT-7 ranker re-test, marginal-not-material): [`2026-05-06_v002_feat7_agreement_features_NO_GO.md`](2026-05-06_v002_feat7_agreement_features_NO_GO.md)
- Sibling (supervised ranker NO_GO): [`2026-05-06_v002_masked_supervised_ranker_NO_GO.md`](2026-05-06_v002_masked_supervised_ranker_NO_GO.md)
- Operator scope chat: 2026-05-06 ("Path B, single tie-breaker run, strict acceptance — information gain not rescue mission").
- Roadmap: CLAUDE.md §4 Phase 3 leads with #34, then #32. Phase 1 closes with this diary.
