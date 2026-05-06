# RESEARCH-003b — v002 selector on yfinance starter_equities (post-Phase-0)

> **DRAFT_NO_GO** as of 2026-05-06 evening. Phase 0 simulator/eval
> fixes are merged. Audit-bundle re-run completed; audit-v2 and
> phase0-final still in flight. Numbers below are from the audit-
> bundle run; refinements (#56 idle-day window, #57 trading-day
> calendar) will shift these slightly but the qualitative verdict
> is stable. This entry promotes from DRAFT to FINAL once those
> runs land.

**Date:** 2026-05-06
**Verdict:** **NO_GO** (draft — qualitative verdict stable; numbers pending final runs)
**Issue:** [#3](https://github.com/l2code/trading-bot-rl/issues/3) (PROVISIONAL banner from previous entry will be lifted by FINAL)
**Variant:** `selector_v002`
**Run (audit-bundle):** Kaggle `crazypenguin/rl-swing-v002-rerun-audit-bundle`
**Trainer commit at run time:** `eb32fba` (FIX-AUDIT-BUNDLE).
**Final-state metrics will use:** `crazypenguin/rl-swing-v002-rerun-audit-v2` or `…-phase0-final` once they complete.

---

## Question

After all Phase 0 simulator/evaluation fixes, does the v2 multi-
strategy selector PPO produce a policy that beats the strongest
baseline on the 2022 yfinance test window under the Phase-24 gate?

## Source

- **Provider:** yfinance.
- **Tier:** **exploratory** — same constraints as v1.
- **Universe:** `starter_equities`.

## Phase 0 fixes applied

Same set as v1 plus the v2-specific:
- **#26** v2 skip-CF mode = `highest_signal` (no hindsight peek;
  uses prior info only).

## Methodology

- **Variant:** `selector_v002` — per-(symbol, date) decisions over
  `Discrete(N+1)` actions (skip + N strategies, N=3).
- Same train/val/test windows, seeds, hyperparams as v1 — only the
  env shape differs.
- **`illegal_action_penalty = 0.5`** — picking a non-fired strategy
  slot pays this constant.
- **Reward params:** identical to v1 (turnover 0.05, holding 0.02,
  dd 0.10, skip mirror 1.0).

## Headline metrics — audit-bundle

Test window 2022-01-01..2022-12-31. Selector baselines + trained:

| model_id                                  | score   | n_trades | take_rate | total_return | sharpe | max_DD | per_strat |
|-------------------------------------------|--------:|---------:|----------:|-------------:|-------:|-------:|-----------|
| `selector_baseline_random` (strongest)    | 0.7177  | 1241     | 0.694     | +2.18        | +6.60  | 0.159  | [839, 69, 333] |
| `selector_baseline_always_skip`           | 0.3250  | 0        | 0         | 0            | 0      | 0      | [0, 0, 0] |
| **`ppo_selector_v002` (trained)**         | **0.3250** | **0** | **0** | **0** | **0** | **0** | **[0, 0, 0]** |

The trained PPO is **bit-identical to `selector_baseline_always_skip`**
across all 30 evaluation points (3 seeds × 10 checkpoints) — n_trades=0,
score=0.3250, no per-strategy takes anywhere.

## Gate output

vs **strongest baseline (`selector_baseline_random`, score 0.7177)**:

| Metric              | Trained | Baseline | Δ      | Status |
|---------------------|--------:|---------:|-------:|--------|
| total_return        | 0       | +2.18    | -2.18  | **✗ MATERIAL regression** |
| annualized_sharpe   | 0       | +6.60    | -6.60  | **✗ MATERIAL regression** |
| profit_factor       | n/a     | high     | -     | **✗ MATERIAL regression** |
| max_drawdown        | 0       | 0.159    | -0.159 | informational (skipping = no DD) |
| turnover_take_rate  | 0       | 0.694    | -0.694 | informational |

NO_GO by 3+ material regressions.

## Why v2 collapsed to "always skip"

The structural problem is the **action space + illegal-action
penalty interaction**, not reward calibration:

- Action space: `Discrete(4)` (skip + 3 strategies).
- On a typical pack, only 1 strategy fires (out of 3). So 2 of 3
  take-actions are "illegal" (pick a non-fired slot) and pay
  `-illegal_action_penalty = -0.5`.
- Random init at 25% per action → expected reward of "random take"
  ≈ -0.5×0.5 + 0.175×0.5 ≈ -0.16.
- Skip's expected reward ≈ -mean(cf risk_adj) × scale ≈ -0.18.
- These are very close; with `ent_coef=0.01` the policy can't reliably
  find "pick the *fired* strategy" before entropy collapses.
- Once skip-everywhere is reached, gradient flow stops — no
  exploration discovers the +0.175 reward from picking the
  fired strategy correctly.

This is not a reward-calibration issue. It's a **structural action-
space issue** that wants formal action masking (#29 MaskablePPO):
mask out non-fired strategies entirely so the policy can't pick
them, focusing exploration on legitimate alternatives.

## What this proves

Phase 0 fixes are **correct** — simulator + evaluator are now
self-consistent. v2's selector framing **cannot escape skip-
everywhere at default PPO hyperparams without action masking**.
The action space itself is currently teaching the agent the
wrong lesson (skipping is safer than picking a strategy slot at
random).

## Known limitations / known-not-changing-the-verdict

1. **Single test cycle** (#5).
2. **Exploratory tier** — yfinance cannot earn GO.
3. **Three seeds.** Bit-identical 0/0/0 across all 30 evaluations
   strongly suggests this is the policy's globally-optimal
   collapse, not seed noise.
4. **Default `ent_coef=0.01`.** A higher entropy might push the
   policy into the take-strategy-when-fired region long enough
   to reach a positive-reward attractor — but the more principled
   fix is action masking, not entropy.

## What would change the verdict

- **Evidence against:** MaskablePPO-trained v2 (#29) producing a
  policy with `per_strategy_take_counts > 0` on multiple
  strategies AND beating `selector_baseline_random` on the gate.
- **Evidence for:** even with masking, no v2 config beats random.
  That would close the v2 selector framing as a research direction.

## Recommendation

**Do not spend more PPO run budget on v2 without action masking.**
The current action space is a structural pathology. Per the
operator (chat 2026-05-06) and confirmed by this run:

1. Implement #29 MaskablePPO (sb3-contrib).
2. Re-train v2 with masking; expect non-zero
   `per_strategy_take_counts`.
3. Apply gate. If still NO_GO, file #30 (supervised ranker
   baseline) — if a simple ranker beats masked v2, the RL
   machinery isn't earning its complexity yet.

## Cross-references

- Predecessor (PROVISIONAL): `2026-05-06_v002_selector_NO_GO.md`
- Sibling: `2026-05-06_v001_filter_post_phase0_DRAFT_NO_GO.md`
- Roadmap: CLAUDE.md §4 — Phase 1 leads with #29.
- Operator audit chats: 2026-05-06 (Phase 0 fixes + Next Stage framing).
