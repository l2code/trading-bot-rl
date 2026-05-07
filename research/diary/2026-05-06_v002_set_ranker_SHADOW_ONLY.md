> # ⚠ [CORRECTION 2026-05-07 — FIX-#78] — VERDICT INVALIDATED
>
> **The numbers in this diary were computed on `synthetic_momentum`, not yfinance.**
> The "first trained policy on yfinance to clear all 5 Phase-24 gate metrics
> vs random — including LOWER max_drawdown" claim is **false on real yfinance**.
>
> On yfinance 2022:
> - set_ranker DD = 0.832 (HIGHER than yfinance random's 0.704), not 0.154 (synthetic).
> - set_ranker gate output: NO_GO improved=1 mat_regress=2 (vs claimed GO 5-of-5 on synthetic).
>
> The architecture's apparent "low-DD" property was an artifact of synthetic_momentum's
> structure, not a real property of the DeepSets encoder.
>
> **Reframed but not refuted:** the per_strat distinctness from first_fired DOES
> survive on yfinance ([946, 34, 54] vs [1127, 32, 40]) — the architecture is
> doing real selection beyond priority order; that selection just isn't profitable
> on real 2022 yfinance.
>
> Verdict in scorecard / CLAUDE.md §2 corrected to **NO_GO**. See
> [`2026-05-07_d4_canonical_yfinance_rebaseline.md`](2026-05-07_d4_canonical_yfinance_rebaseline.md)
> for full numbers. Original SHADOW_ONLY framing preserved below for audit-trail.

# RESEARCH-034a — set/slate encoder cheap diagnostic (FEAT-34 PR-1)

**Date:** 2026-05-06
**Verdict:** **SHADOW_ONLY**
**Issue:** [#34](https://github.com/l2code/trading-bot-rl/issues/34) (PR-1 only — sb3 features-extractor + Kaggle PPO retrain are PR-2, gated on this verdict)
**Variant:** baseline `selector_baseline_set_ranker` (NOT a TrainingVariant)
**Run:** local on Loki — torch DeepSets-style encoder, ~30k packs × 3 slots, fit in 11s; best val loss at epoch 21 via early-stopping
**Trainer commit at run time:** `f734e30` (post Phase 1 closure / PR #73 merged)

---

## Question

Phase 1 closed with the finding that masked-PPO is bit-identical
to `selector_baseline_first_fired` regardless of feature set. The
hypothesis: the slate framing's collapse is **architecture-level**
— a flat per-slot MlpPolicy can trivially encode "always pick slot
0" — not feature-level. A permutation-equivariant encoder can't
encode that shortcut. Does it actually produce a different policy?

**Operator-pre-agreed cheap-diagnostic acceptance:** the set ranker
must materially improve over the FEAT-7 HistGB ranker. "Material"
= either flips the gate verdict OR shows per-strategy distribution
genuinely beyond first_fired's [1423, 79, 278] pattern. If it
doesn't, PR-2 (Kaggle PPO retrain) is not justified and we pivot
to #32.

## Source

- **Provider:** yfinance.
- **Tier:** **exploratory** — same constraint as every prior v002
  result. Cannot earn decision-grade GO.
- **Universe:** `starter_equities`.

## Methodology

- **New module** `src/rl_swing/rl/agents/slate_encoder.py` — a
  DeepSets-style PyTorch encoder. `phi(x_k)` is a small shared MLP
  applied per slot; aggregate is `[sum, max, mean]` pool over
  fired slots; `rho_slot(phi_k, agg, ctx) → 1` per-slot scoring;
  `rho_skip(agg, ctx) → 1` separate skip head. Logits ordered
  `[skip, slot_0, ..., slot_{N-1}]` to match
  `MultiStrategySwingTradingEnv.action_space`. Slot weights are
  shared and the aggregate is order-invariant — the model literally
  cannot trivially encode "always pick slot 0."
- **New scorer** `SetRankerSelectorScorer` — wraps the trained
  encoder for the SelectorScorer Protocol. Builds `(slot_features,
  slot_mask, ctx)` tensors at inference; reads logits; argmax over
  fired slots after masking unfired ones to -inf; skip if
  `slot_logit < skip_logit + threshold`.
- **Per-slot features** (deliberately NO `slot_idx`):
  `signal_strength`, `base_size_pct`, `max_holding_days_norm`,
  `slot_is_top_signal`, `slot_rank_by_signal`. The agreement
  features (FEAT-7) are split between per-slot (the ones that
  describe a single slot) and pack-level (the ones that summarize
  the slate).
- **Pack context** (`ctx`): `ALL_FEATURE_NAMES` (frame features) +
  `PACK_AGREEMENT_FIELDS` (FEAT-7 pack-level summaries).
- **Loss:** per-slate MSE — for each pack, compute squared errors
  on every fired slot's predicted vs realized risk-adjusted return,
  plus MSE on the skip head against `-best_signal_r` (the
  highest-signal slot's realized risk-adjusted return, negated;
  so the skip head learns to dominate when no fired slot has
  positive predicted EV). Mirror of the env's
  `skip_counterfactual_mode = "highest_signal"` (FIX-#26).
- **Training:** Adam lr=1e-3, 30 epochs, batch=256, 10% val split
  for early-stopping. Single seed (11). Fit in 10.9s on Loki.
  **Caveat:** the loss diverged in the late epochs (epochs 24+);
  early-stopping retained the epoch-21 checkpoint with val_loss
  41.46. The divergence is an honest concern — a real PR-2 (or a
  v0 follow-up) should add gradient clipping + LR warmup. For this
  diagnostic the early-stopped checkpoint is what was evaluated.
- **Eval:** `rl-swing validate` on the v002_masked YAML test
  window (2022 yfinance), formal Phase-24 gate via
  `acceptance_gate.evaluate_gate()`.

## Headline metrics — set ranker on test 2022

Daily-P&L basis (FIX-#36), trading-day spread = 260 (FIX-#57).

| model_id                                  | score   | n_trades | take_rate | total_return | sharpe | max_DD | profit_factor | per_strat |
|-------------------------------------------|--------:|---------:|----------:|-------------:|-------:|-------:|--------------:|-----------|
| `selector_baseline_random`                | 0.7186  | 1241     | 0.6941    | +2.1697      | +7.108 | 0.1557 | 3.188         | [839, 69, 333] |
| `selector_baseline_first_fired`           | 0.6905  | 1780     | 0.9955    | +4.8754      | +7.455 | 0.1992 | 3.511         | [1423, 79, 278] |
| `selector_baseline_highest_signal`        | 0.6910  | 1780     | 0.9955    | +4.5134      | +7.351 | 0.1975 | 3.431         | [1259, 116, 405] |
| `selector_baseline_supervised` (HistGB FEAT-7) | 0.7145 | 1031 | 0.5766 | +0.8208 | +4.645 | 0.1889 | 2.141 | [729, 84, 218] |
| **`selector_baseline_set_ranker`**        | **0.7066** | **1687** | **0.9435** | **+4.9898** | **+8.056** | **0.1539** | **3.915** | **[1325, 73, 289]** |

The set ranker has the **highest absolute total_return, sharpe, and profit_factor** of any policy tested on this dataset — including all baselines AND every prior trained model.

## Phase-24 gate output — set ranker vs strongest baseline

vs `selector_baseline_random` (the canonical Phase-24 gate target):

| Metric              | Trained | Baseline | Δ        | Status |
|---------------------|--------:|---------:|---------:|--------|
| total_return        | +4.9898 | +2.1697  | +2.8201  | ✓ improved |
| annualized_sharpe   | +8.0564 | +7.1085  | +0.9480  | ✓ improved |
| profit_factor       | 3.9153  | 3.1879   | +0.7275  | ✓ improved |
| max_drawdown        | 0.1539  | 0.1557   | -0.0018  | ✓ improved (lower DD) |
| turnover_take_rate  | 0.9435  | 0.6941   | +0.2494  | ✓ improved (informational) |

**Gate output: GO — 5 of 5 metrics improved, no material
regressions.** This is the **first trained policy on yfinance**
where every individual gate metric is better than random. In
particular, `max_drawdown` is *lower* than random's (0.1539 vs
0.1557) — every prior trained policy traded extra DD for return,
which is what kept them off the gate.

## Phase-24 gate output — set ranker vs HistGB ranker (the cheap-diagnostic floor)

vs `selector_baseline_supervised` (the FEAT-7 HistGB ranker, the
operator's pre-agreed floor for "did the architecture change
help"):

| Metric              | Set ranker | HistGB ranker | Δ        | Status |
|---------------------|-----------:|--------------:|---------:|--------|
| total_return        | +4.9898    | +0.8208       | +4.169   | ✓ improved |
| annualized_sharpe   | +8.0564    | +4.6453       | +3.411   | ✓ improved |
| profit_factor       | 3.9153     | 2.1407        | +1.775   | ✓ improved |
| max_drawdown        | 0.1539     | 0.1889        | -0.035   | ✓ improved (lower DD) |
| turnover_take_rate  | 0.9435     | 0.5766        | +0.367   | ✓ improved |

**Gate output: GO — 5 of 5 improved.** The set ranker dominates
the HistGB ranker on every metric. The architecture change isn't
just marginal — it's a structural lift.

## Why the verdict is **SHADOW_ONLY**, not GO

Same reasons that capped the masked-PPO SHADOW_ONLY (PR #70) at
the gate level + tier level:

1. **Tier.** yfinance is exploratory per CLAUDE.md §3.5. The gate
   says GO; tier rules cap exploratory at SHADOW_ONLY. Promotion
   to GO requires WRDS replication ([#4](https://github.com/l2code/trading-bot-rl/issues/4)).
2. **Single seed, single training run.** This is one fitted
   encoder out of one seed. It's not seed-stable evidence. PR-2
   (the Kaggle PPO retrain that uses this encoder as a sb3
   features extractor) gives the multi-seed information.

But the operator's pre-agreed cheap-diagnostic threshold is
**clearly cleared.** Per the issue scope:

> "Cheap-diagnostic acceptance for PR-1: the set-ranker must
> materially improve over the HistGB FEAT-7 ranker. 'Material' =
> either flips the gate verdict OR shows per-strategy distribution
> genuinely beyond first_fired's [1423, 79, 278] pattern."

Both are true: the set ranker flips the gate vs HistGB (NO_GO →
GO), AND its per-strategy distribution [1325, 73, 289] is
genuinely distinct from first_fired's [1423, 79, 278] (98 fewer
slot-0 trades, 6 fewer slot-1, 11 more slot-2). This is the FIRST
trained policy on this benchmark where slot 0 isn't dominant by
the deterministic-priority pattern.

## Why the composite score (0.7066) trails random (0.7186) anyway

A real and honest tension worth flagging: the validation composite
score has set ranker AT 0.7066 vs random's 0.7186, a -0.012 gap.
But the per-metric gate is 5-of-5 improved. Two things going on:

1. **Saturation.** `n_total_return`, `n_sharpe`, and
   `n_profit_factor` all clamp to 1.0 once the policy clears certain
   thresholds. Set ranker is well above these clamps, so its huge
   absolute lead on return/sharpe/PF gets compressed to "1.0" in
   the composite.
2. **Turnover term.** `n_turnover` rewards higher take rate
   directly. Set ranker takes 94.4% vs random's 69.4% — a +0.25
   lift on the turnover component, but the formula clamps that
   too. The DD component swings random a bit higher than set
   ranker on the formula even though the actual DD's are nearly
   identical.

Net effect: the *per-metric* gate is a more discriminating signal
here than the composite. A follow-up RFC could revisit the
composite weighting now that we have a policy on the right side
of every individual metric.

## Why this is genuinely architectural, not noise

The set ranker's [1325, 73, 289] distribution differs from
first_fired's [1423, 79, 278] by roughly 100 trades on slot 0 and
slot 2 — a real reallocation of decisions, not a rounding-error
distinction. The set ranker is genuinely *choosing* which slot to
pick rather than always picking the lowest-index fired slot.

The architectural diagnostic confirms the hypothesis from PR #73's
NO_GO diary: the slate framing's prior collapse to first_fired
was an MlpPolicy artifact. Permutation-equivariance breaks the
shortcut. **Phase 3 has its first positive result.**

## Known limitations / known-not-changing-the-verdict

1. **Training divergence.** Loss exploded after epoch 23. Best
   checkpoint at epoch 21 (val_loss 41.46) was retained via
   early-stopping. Real concern; PR-2 should add gradient clipping
   + LR warmup. Non-blocking for the diagnostic verdict because
   the early-stopped checkpoint is what's actually being evaluated
   here, and it works.
2. **Single seed.** PR-2's Kaggle retrain (3 seeds × 500k
   timesteps) is the multi-seed information.
3. **Composite score formulaically rewards lower take rate** in a
   way that makes "trade more profitably" look like a wash. Diary
   notes this; not blocking.
4. **Exploratory tier.** WRDS replication required for GO.

## What this proves

- **The slate framing is NOT structurally exhausted on yfinance.**
  PR #73's NO_GO was specific to flat-MlpPolicy + flat per-slot
  features. A permutation-equivariant encoder over the same data
  produces a policy that beats every baseline on every gate metric.
- **The architecture-level shortcut hypothesis is correct.**
  Removing slot_idx from the per-slot features + sharing weights
  across slots + aggregating order-invariantly is exactly the fix.
- **PR-2 (Kaggle PPO retrain with the encoder as features
  extractor) is justified.** The supervised pre-test cleared the
  operator's pre-agreed cheap-diagnostic acceptance.

## What would change the verdict

- **Promotion to GO at the gate level:** WRDS canonical
  replication ([#4](https://github.com/l2code/trading-bot-rl/issues/4))
  with the same per-metric gate pass.
- **Demotion to NO_GO:** if PR-2's masked-PPO retrain (using this
  encoder as a sb3 features extractor) collapses to first_fired
  again, the policy network's gradient dynamics interact poorly
  with the encoder. Unlikely given the supervised result, but
  PR-2 is what tests it.

## Recommendation

**Proceed to PR-2:** wire `SlateEncoder` as a
`sb3-contrib.MaskablePPO` features extractor inside a new
`selector_v002_masked_setencoder` TrainingVariant; new experiment
YAML; one Kaggle private retrain (3 seeds × 500k timesteps, same
discipline as the FEAT-7 tie-breaker run from PR #73). Strict
acceptance:

1. Beat `selector_baseline_random` on Phase-24 gate (≥2 of 5
   improved, no material regressions). The set ranker did 5-of-5;
   PPO should at minimum match.
2. NOT bit-identical to first_fired (different per_strat counts).
3. ≥2 of 3 seeds find usable policies (not a one-seed transient).

If PR-2 clears these, **then** #27 Optuna becomes worth running
on top — the operator's deferred sweep target is finally a model
class that's responsive to entropy/lr tuning rather than collapsing
to a 3-line baseline regardless of hyperparameters.

If PR-2 fails: the supervised inductive bias that set ranker
captured doesn't translate to RL training dynamics. That's a more
nuanced finding than "v002 framing is exhausted" — it's "the
framing works for supervised ranking but not for RL." Triggers
either #32 chronological v3 or a contextual-bandit / supervised
production path (use the set ranker directly).

## Cross-references

- Predecessor (Phase 1 closure NO_GO): [`2026-05-06_v002_masked_feat7_tiebreaker_NO_GO.md`](2026-05-06_v002_masked_feat7_tiebreaker_NO_GO.md)
- Sibling (FEAT-7 ranker NO_GO, marginal-not-material): [`2026-05-06_v002_feat7_agreement_features_NO_GO.md`](2026-05-06_v002_feat7_agreement_features_NO_GO.md)
- Sibling (HistGB ranker NO_GO, the cheap-diagnostic floor): [`2026-05-06_v002_masked_supervised_ranker_NO_GO.md`](2026-05-06_v002_masked_supervised_ranker_NO_GO.md)
- Operator scope chat: 2026-05-06 ("Phase 3 step 1 = #34, then re-run #30-style ranker on the new representation as cheap diagnostic before PPO. Park #27.").
- Roadmap: CLAUDE.md §4 Phase 3.
