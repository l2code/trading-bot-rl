# RESEARCH-030 — supervised ranker baseline for v002 (Phase 1 step 2)

**Date:** 2026-05-06
**Verdict:** **NO_GO**
**Issue:** [#30](https://github.com/l2code/trading-bot-rl/issues/30) (supervised half — LinUCB stays open as a follow-up)
**Variant:** baseline `selector_baseline_supervised` (NOT a TrainingVariant)
**Run:** local on Loki (i7-1360P, 16 threads); 28,219 training rows; sklearn HistGradientBoostingRegressor fit in 0.8s; in-sample MSE 0.01671
**Trainer commit at run time:** `f590f08` (post FEAT-29 + FIX-68 + RESEARCH-29 SHADOW_ONLY merged)

---

## Question

Does a simple supervised contextual-bandit ranker — same features
as the v002 PPO observation, gradient-boosted regression on
realized risk-adjusted returns — beat the masked-PPO selector and
the strongest random baseline on the Phase-24 gate?

If yes: RL machinery is overkill on this decision shape; v3 should
be a bandit / supervised model.
If no: the slate framing itself is the bottleneck; v2 selector-
class architectures are unlikely to beat random on yfinance
without richer features.

## Source

- **Provider:** yfinance (daily bars, auto-adjusted).
- **Tier:** **exploratory** — same constraint as the v002 / masked
  variants. Cannot earn GO regardless of result; the strongest
  reachable verdict is SHADOW_ONLY.
- **Universe:** `starter_equities`.

## Methodology

- **Form factor:** SelectorScorer baseline (not a TrainingVariant).
  Plugs into the v002 + v002_masked evaluate() lineup next to
  `selector_baseline_random`, etc., and is auto-included when the
  artifact exists at `data/models/selector_baseline_supervised/model.joblib`.
- **Target / label:** per-slot **realized risk-adjusted return** =
  `outcome.return_pct / target_risk_pct` with target_risk_pct=0.02.
  Same target the RewardModel optimizes; computed by simulating
  every fired (pack × strategy) trade with the SAME
  `ExecutionSimulator` + `EquityExecutionModel` + cost layer the
  v2 env uses. No train/eval simulator drift.
- **Features:** `ALL_FEATURE_NAMES` (the v2 obs builder's frame
  features) + `[slot_idx, signal_strength, base_size_pct,
  max_holding_days_norm]` per fired slot. Total 27 features per row.
- **Model:** `sklearn.ensemble.HistGradientBoostingRegressor`
  (max_iter=200, max_depth=8, random_state=11). Default sklearn
  hyperparameters otherwise; not tuned via Optuna or cross-val.
- **Training set:** 28,219 (pack × fired strategy) rows from
  2014-01-01..2020-12-31 yfinance (warmup back to 2012-12-19 so
  long-lookback features populate from day 1 of the train window).
  22 rows skipped because the simulator returned None (boundary
  edge cases — bars unavailable for the candidate's `as_of`).
- **Inference policy:** for each pack at eval time, predict
  `risk_adj_return` for every fired slot, take argmax. If max
  prediction < 0 (skip threshold), return 0 (skip — predicted
  negative-EV trade). Threshold-at-0 is principled; matches the
  reward model's "skip when no fired strategy has predicted
  positive EV."
- **Train/test split:** train 2014-2020, test 2022. Same windows
  as the audit-v2 unmasked + masked-PPO runs. **No 2021
  validation tuning** at v0 — the model has no hyperparameter
  selection, so 2021 is effectively unused.

## Headline metrics — supervised ranker on test 2022

Daily-P&L basis (FIX-#36), trading-day spread = 260 (FIX-#57).
Compared against the canonical selector baselines + the trained
masked-PPO from PR #67 / kaggle run `…-maskableppo-phase-1-private`:

| model_id                             | score   | n_trades | take_rate | total_return | sharpe | max_DD | per_strat |
|--------------------------------------|--------:|---------:|----------:|-------------:|-------:|-------:|-----------|
| `selector_baseline_random` (strongest) | 0.7186  | 1241    | 0.6941    | +2.1697      | +7.108 | 0.1557 | [839, 69, 333] |
| `selector_baseline_first_fired`      | 0.6905  | 1780    | 0.9955    | +4.8754      | +7.455 | 0.1992 | [1423, 79, 278] |
| `selector_baseline_highest_signal`   | 0.6910  | 1780    | 0.9955    | +4.5134      | +7.351 | 0.1975 | [1259, 116, 405] |
| **`selector_baseline_supervised`**   | **0.7107** | **1088** | **0.6085** | **+0.8214** | **+4.376** | **0.1963** | **[778, 77, 233]** |
| `ppo_selector_v002_masked` (canonical Kaggle run) | 0.6905  | 1780 | 0.9955 | +4.8754 | +7.455 | 0.1992 | [1423, 79, 278] |

## Phase-24 gate output — `selector_baseline_supervised` vs strongest baseline

vs `selector_baseline_random` (score 0.7186), the operator-mandated
gate target:

| Metric              | Trained | Baseline | Δ        | Status |
|---------------------|--------:|---------:|---------:|--------|
| total_return        | +0.8214 | +2.1697  | -1.3483  | **✗ MATERIAL regression** (>0.05 threshold) |
| annualized_sharpe   | +4.376  | +7.108   | -2.733   | **✗ MATERIAL regression** (>0.5 threshold) |
| profit_factor       | 2.020   | 3.188    | -1.168   | **✗ MATERIAL regression** (>0.3 threshold) |
| max_drawdown        | 0.1963  | 0.1557   | +0.0406  | (under 0.05 threshold; not material) |
| turnover_take_rate  | 0.6085  | 0.6941   | -0.0856  | (informational) |

**Verdict: NO_GO by 3 material regressions.** The ranker is genuinely
selective (60.85% take rate vs random's 69.41%) but its discrimination
is wrong-directional — it's filtering out winners along with losers.
Total return drops to +0.82 vs random's +2.17.

## Phase-24 gate output — `selector_baseline_supervised` vs masked-PPO

The operator framing on #30: *"if a simple ranker beats masked-PPO
v2, RL machinery isn't earning its complexity yet."* Computed
explicitly:

| Metric              | Trained | Baseline (masked PPO) | Δ        | Status |
|---------------------|--------:|----------------------:|---------:|--------|
| total_return        | +0.8214 | +4.8754               | -4.054   | **✗ MATERIAL regression** |
| annualized_sharpe   | +4.376  | +7.455                | -3.079   | **✗ MATERIAL regression** |
| profit_factor       | 2.020   | 3.511                 | -1.490   | **✗ MATERIAL regression** |
| max_drawdown        | 0.1963  | 0.1992                | -0.003   | ✓ improved (DD lower) |
| turnover_take_rate  | 0.6085  | 0.9955                | -0.387   | (informational) |

**Verdict: NO_GO** — ranker does NOT beat masked-PPO. The
operator's "RL machinery isn't earning its complexity" framing
doesn't trigger here.

## The much more important finding: masked-PPO is bit-identical to `selector_baseline_first_fired`

The canonical Kaggle 500k×3 masked-PPO trained model produces
exactly these numbers on test 2022:

```
score=0.690470  total_return=+4.875414  per_strat=[1423, 79, 278]
```

`selector_baseline_first_fired` — a 3-line rule that returns
`1 + min(k for k where pack.candidates[k] is not None)` and
otherwise returns 0 — produces:

```
score=0.690470  total_return=+4.875414  per_strat=[1423, 79, 278]
```

**Bit-identical to 6 decimals on every metric.** The "trained"
masked-PPO learned exactly the deterministic-priority rule:
*always take the lowest-index strategy that fired.*

This refines the SHADOW_ONLY verdict on the masked-PPO diary
([`2026-05-06_v002_masked_SHADOW_ONLY.md`](2026-05-06_v002_masked_SHADOW_ONLY.md)):
the policy that passes the Phase-24 gate vs random isn't merely
"undertrained" or "transiently good" — it's *bit-equivalent to a
3-line baseline*. The strong gate-pass (4-of-5 improved) is a
property of the first_fired rule, not of the RL machinery. An
addendum on the SHADOW_ONLY diary now flags this; the SHADOW_ONLY
verdict still stands (the gate output is what it is, and the data
tier is still exploratory) but the implied "PPO learned something"
reading is incorrect.

## What this proves

- **The supervised ranker as designed (slate-feature HistGB on
  realized risk-adjusted returns) doesn't carry useful signal on
  yfinance starter_equities.** The features predict trade
  outcomes worse than random selection. Either the features are
  the wrong features, or the underlying noise on yfinance is too
  high for any cheap supervised approach to extract signal at this
  scale (N=28k examples, high label variance σ=0.21).
- **Masked-PPO at default `ent_coef=0.01` collapses to the
  simplest deterministic baseline.** Masking opened the door, but
  default exploration didn't push the policy past `first_fired`.
- **Both selector architectures (PPO + supervised) fail the
  Phase-24 gate vs random.** Random's apparent dominance is
  almost entirely from lower max_drawdown (0.156 vs 0.196-0.199).
  Random's drawdown is lower because it skips ~30% of fired packs
  uniformly, which acts as a portfolio-level noise reducer.

## Known limitations / known-not-changing-the-verdict

1. **No hyperparameter sweep** on the ranker. HistGB defaults are
   fine for v0 but probably not optimal. A meaningful tuned ranker
   would need a separate RFC with cross-val on 2021 + held-out 2022
   eval. **Filed as a follow-up** if anyone wants to revisit; not
   blocking other Phase 1 work.
2. **Same features as the v002 PPO obs.** If the features
   themselves don't carry signal (and they apparently don't on
   yfinance), no model class trained on them will beat random.
   **#7 cross-strategy agreement features** is a higher-priority
   intervention.
3. **Single test cycle ([#5](https://github.com/l2code/trading-bot-rl/issues/5)).**
   2022-only. The ranker's behavior could shift markedly across years.
4. **Single train/eval window.** No bootstrap or k-fold on the
   training data — the in-sample MSE 0.0167 is just plumbing
   evidence that the fit ran, not a generalization claim.

## What would change the verdict

- **Promotion above NO_GO:** swap features. **#7 cross-strategy
  agreement features** is the highest-leverage intervention — if
  three strategies agree on a (symbol, date), that's a signal not
  visible in any individual strategy's feature row. Re-train the
  ranker on `[per-slot features, agreement features]` and re-run
  the gate.
- **Or:** WRDS canonical replication ([#4](https://github.com/l2code/trading-bot-rl/issues/4)).
  yfinance's noise floor may be high enough that NO model trained
  on it can beat random; canonical data could change that calculus.
- **Closing v002 selector framing entirely:** if `#7 + #8 + this
  ranker` all fail to clear random, the slate framing itself is
  exhausted. Move to Phase 3 architectural work
  ([#34 set/attention encoder](https://github.com/l2code/trading-bot-rl/issues/34))
  or [#32 portfolio-aware chronological v3](https://github.com/l2code/trading-bot-rl/issues/32).

## Recommendation

The operator's Phase 1 sequence had `#30` ahead of `#27` (Optuna).
The original logic: prove a simple ranker first; if RL doesn't beat
it, RL machinery is wasted. **The ranker doesn't beat random and
doesn't beat masked-PPO.** Both directions of that test produce
NO_GO. The bit-identity of masked-PPO with `first_fired` adds the
finding that masked-PPO's gate pass is also from a non-RL rule.

**Three concrete next steps, ordered by EV:**

1. **#7 cross-strategy agreement features.** Filed but not yet
   implemented. The current observation gives the policy per-slot
   features but not "do strategies agree on this (symbol, date)."
   If three strategies fire simultaneously, that's high-conviction
   information. The ranker would benefit; PPO would benefit; this
   one feature set could materially shift both verdicts. Cheaper
   than #34 architectural work.
2. **#8 Optuna sweep against masked-PPO.** Still worth running
   once #7 is in. The operator's framing for #8 (chat 2026-05-06)
   was: "use the ranker result as context, not in isolation."
   This diary IS that context: the ranker fails. So #8's
   acceptance criterion should be tightened — masked-PPO with
   tuned hyperparams must produce a policy that BOTH (a) clears
   the gate vs random AND (b) beats `selector_baseline_first_fired`
   on absolute composite score (currently bit-identical to PPO).
3. **#34 or #32** if (1) and (2) both fail. The slate framing
   itself is then exhausted.

## Cross-references

- Predecessor (masked-PPO SHADOW_ONLY): [`2026-05-06_v002_masked_SHADOW_ONLY.md`](2026-05-06_v002_masked_SHADOW_ONLY.md)
  — addendum-amended in PR with this diary to flag bit-identity
  with `selector_baseline_first_fired`.
- Sibling diaries: v1 + v2 unmasked FINAL_NO_GO entries (see
  [`docs/scorecard.md`](../../docs/scorecard.md) research-state table).
- RFC: [#30](https://github.com/l2code/trading-bot-rl/issues/30)
  — supervised half closed by this diary; LinUCB sub-RFC stays open.
- Roadmap: CLAUDE.md §4 Phase 1.
- Operator scope chat: 2026-05-06 (Phase 1 sequence #29 → #30 → #8;
  #30's framing: "ranker result as context for #8, not in isolation").
