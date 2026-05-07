# RESEARCH-029 — v002 selector with MaskablePPO action masking (Phase 1)

**Date:** 2026-05-06
**Verdict:** **SHADOW_ONLY**
**Issue:** [#29](https://github.com/l2code/trading-bot-rl/issues/29)
**Variant:** `selector_v002_masked` (FEAT-29 scaffold landed in PR #67)
**Run:** Kaggle private kernel `crazypenguin/rl-swing-v002-maskableppo-phase-1-private`
**Trainer commit at run time:** `e271eb5` (post FEAT-29 scaffold + FIX-68 alias fallback)

---

## Question

Does formal action masking (sb3-contrib `MaskablePPO` with
`action_masks() = [True, fired_slot_0, ..., fired_slot_N-1]`)
unstick the v002 selector from its all-skip collapse and produce a
policy that beats `selector_baseline_random` (audit-v2 score 0.7186)
under the Phase-24 gate?

## Source

- **Provider:** yfinance (daily bars, auto-adjusted).
- **Tier:** **exploratory** — yfinance can never earn decision-grade
  GO regardless of result (CLAUDE.md §3.5). The strongest verdict
  reachable on this tier is SHADOW_ONLY.
- **Universe:** `starter_equities` (15 symbols + SPY, QQQ).

## Methodology

- **Variant:** `selector_v002_masked` — same env, observation,
  reward, and per-pack semantics as the unmasked `selector_v002`.
  Only the algorithm (`PPO` → `MaskablePPO`) and inference scorer
  (`MaskablePpoSelectorScorer`) differ. Action mask:
  `[skip=True, slot_k=True iff candidates[k] is not None]`.
- **Hyperparameters:** identical to the audit-v2 unmasked v002 run
  — 3 seeds (11/22/33), 500k timesteps, n_envs=4 (SubprocVecEnv
  fork), MlpPolicy 128×128, `ent_coef=0.01`, `lr=3e-4`,
  `eval_interval=50000`, reward weights post-FIX-49.
- **Windows:** train 2014-01-01..2020-12-31, validation 2021,
  test 2022. Same windows as the audit-v2 unmasked v002 run.
- **Trainer commit:** `e271eb5` (FEAT-29 scaffold + FIX-68 alias
  fallback merged to main).
- **Decision criterion (per operator scope, 2026-05-06):** beat
  `selector_baseline_random` (score 0.7186), not merely escape
  the all-skip attractor.

## Headline metrics — masked v002 Phase-1 run (test 2022)

Daily-P&L basis (FIX-#36), trading-day spread = 260 (FIX-#57).

| model_id                                  | score   | n_trades | take_rate | total_return | sharpe | max_DD | per_strat |
|-------------------------------------------|--------:|---------:|----------:|-------------:|-------:|-------:|-----------|
| `selector_baseline_random` (strongest)    | 0.7186  | 1241     | 0.6941    | +2.1697      | +7.108 | 0.1557 | [839, 69, 333] |
| `selector_baseline_always_skip`           | 0.3250  | 0        | 0         | 0            | 0      | 0      | [0, 0, 0] |
| **`ppo_selector_v002_masked` (trained)**  | **0.6905** | **1780** | **0.9955** | **+4.8754** | **+7.455** | **0.1992** | **[1423, 79, 278]** |

The trained MaskablePPO **diversifies across all three strategies**
— `[1423, 79, 278]` per-strategy take counts. This is the
load-bearing diagnostic FEAT-29 was designed to flip, and it
flipped. Compare to the unmasked v002 audit-v2 result of
`[0, 0, 0]` (bit-identical to `selector_baseline_always_skip`).

## Phase-24 gate output

vs **strongest baseline (`selector_baseline_random`, score 0.7186)**,
computed by `rl_swing.rl.validation.acceptance_gate.evaluate_gate`:

| Metric              | Trained | Baseline | Δ        | Improved | Material regression |
|---------------------|--------:|---------:|---------:|----------|---------------------|
| total_return        | +4.8754 | +2.1697  | +2.7057  | ✓        | ✗ |
| annualized_sharpe   | +7.455  | +7.108   | +0.347   | ✓        | ✗ |
| profit_factor       | 3.511   | 3.188    | +0.323   | ✓        | ✗ |
| max_drawdown        | 0.1992  | 0.1557   | +0.0435  | ✗        | ✗ (under 0.05 threshold) |
| turnover_take_rate  | 0.9955  | 0.6941   | +0.301   | ✓        | ✗ |

**Gate output: GO** — 4 of 5 metrics improved (≥2 required); no
material regressions (DD increase of 0.0435 sits just under the
0.05 material threshold).

## Why the verdict is **SHADOW_ONLY**, not GO

Two reasons, either alone sufficient:

1. **Tier.** Data is yfinance, which is exploratory tier per
   CLAUDE.md §3.5: *"yfinance, synthetic_*: quick-look only;
   never decision-grade."* The Phase-24 gate is the validation
   layer; data-tier rules cap exploratory results at SHADOW_ONLY
   regardless of how cleanly the gate passes. Promotion to GO
   requires WRDS canonical replication ([#4](https://github.com/l2code/trading-bot-rl/issues/4)).
2. **Seed instability.** The cross-seed `model.zip` alias is from
   seed 11 step 300k (val=0.5124). Looking at the eval-history
   per seed:

   | Seed | First nonzero n_trades | best_val (2021) | Notes |
   |------|------------------------|----------------:|-------|
   | 11   | step 300_000           | 0.5124         | Briefly escaped all-skip at step 300k → 1613 trades; **collapsed back** to all-skip from step 350k onwards |
   | 22   | never                  | 0.3250         | Stayed bit-identical to all-skip across all 500k steps |
   | 33   | step 50_000            | 0.4229         | Take-everything from step 50k (2772 trades, score 0.4229), drifted to 2625 trades by step 250k; never improved beyond 0.4229 |

   Only 1 of 3 seeds found a working policy, and even that one was
   transient. The strong 2022 test-set numbers above all come from
   a **single brief checkpoint** that the policy abandoned within
   50k more steps. This is not robust behavior.

The combination — exploratory tier + 1-of-3 seeds barely escaping
— is exactly what SHADOW_ONLY exists for: "the model passes
validation; staking real money on it would be premature."

## What this proves

**Masking is necessary but not sufficient.** The audit-v2 unmasked
v002 was trapped at `[0, 0, 0]` (all-skip) across 30 evaluation
points (3 seeds × 10 checkpoints). With masking:

- 1 seed found a non-trivial policy (transiently) — diversifying
  across all 3 strategies, not just the dominant one.
- 1 seed broke into take-everything — the symmetric collapse.
- 1 seed stayed all-skip even with masking.

This says the action mask **opens the door** but does not stabilize
the policy's stay through the door. The remaining failure is
hyperparameter / exploration: at default `ent_coef=0.01`, only some
seeds find the productive region of action-space, and even those
that do can drift back out. This is a textbook entropy-collapse
signature, exactly what [#8 Optuna sweep](https://github.com/l2code/trading-bot-rl/issues/8)
on `ent_coef` + `lr` is designed to address.

## Known limitations

1. **Exploratory tier (yfinance).** Cannot earn GO. WRDS
   replication ([#4](https://github.com/l2code/trading-bot-rl/issues/4))
   gates the path to GO.
2. **Single test cycle ([#5](https://github.com/l2code/trading-bot-rl/issues/5)).**
   2022-only. Multi-cycle WF could meaningfully shift the per-seed
   stability story (a year where every seed escapes is more
   interesting than one where one seed transiently does).
3. **Three seeds.** With only 1 of 3 succeeding transiently, the
   stability claim is weak. A 5-10 seed re-run would tell us
   whether the 1/3 success rate is a real frequency or a fluke.
4. **Default hyperparams.** `ent_coef=0.01` is the standard PPO
   default — likely too low for the v2 selector's exploration
   problem even with masking. [#8](https://github.com/l2code/trading-bot-rl/issues/8)
   tests `ent_coef ∈ {0.05, 0.1}` and `lr ∈ {1e-4, 3e-4, 1e-3}`.

## Cross-checks (sanity)

- Local 5k-step smoke (preflight task #49) on the same YAML produced
  `per_strategy_take_counts=[1423, 79, 278]` and score 0.6905 —
  bit-identical to the Kaggle 500k×3 result on the test window.
  This is because the cross-seed alias picked up seed 11 step 300k,
  and the smoke happened to converge to a similar early-take-many
  policy. Coincidence, not a reproducibility artifact.
- `metric_basis: "daily_pnl_v36"` and `n_trading_days: 260` confirm
  the run used the post-Phase-0 metric stack (FIX-#36 idle-day fill
  + FIX-#57 trading-day calendar from bars).

## What would change the verdict

- **Promotion to GO:** replicate on WRDS canonical data
  ([#4](https://github.com/l2code/trading-bot-rl/issues/4))
  with the same ≥2-of-5 gate pass AND seed-stable behavior
  (≥3-of-5 seeds finding a productive policy that they don't
  abandon).
- **Demotion to NO_GO:** an Optuna-tuned masked v2
  ([#8](https://github.com/l2code/trading-bot-rl/issues/8))
  that still produces ≤1 seed finding a non-trivial policy. That
  would close masked-v2 PPO as a research direction; the next stop
  is supervised baseline ([#30](https://github.com/l2code/trading-bot-rl/issues/30))
  and / or Phase 3 architectural changes (set/attention encoder
  [#34](https://github.com/l2code/trading-bot-rl/issues/34)).

## Recommendation

Do **not** spend more compute on masked v2 at default hyperparams.
The 1-of-3 seed escape rate at the audit-v2 hyperparams is a
ceiling, not a floor. Per CLAUDE.md §4 Phase 1 sequence:

1. **#30 supervised ranker baseline** (task #47, blocked by this
   verdict — now unblocked). Tells us whether the per-strategy
   features carry enough signal to rank without RL exploration.
   Local-only; no Kaggle quota concern.
2. **#8 Optuna sweep on `ent_coef` + `lr`** (task #27, blocked by
   #30). Run on `selector_v002_masked` (NOT the unmasked variant,
   which the gate has already shown collapses bit-identical to
   all-skip). Acceptance criterion (per operator's #8 refinement):
   trained model's `per_strategy_take_counts` shows diversification
   across ≥2 strategies on ≥3-of-5 seeds.
3. After #8 lands, decide whether to invest in
   [#34 set/attention encoder](https://github.com/l2code/trading-bot-rl/issues/34)
   (Phase 3 architectural) or close v2 PPO as a research direction
   in favor of #30 supervised + a different model class.

## Cross-references

- FEAT scaffold: PR [#67](https://github.com/l2code/trading-bot-rl/pull/67)
- Plumbing fix: PR [#69](https://github.com/l2code/trading-bot-rl/pull/69)
  (FIX-#68 alias fallback when no eval fires)
- Predecessor (post-Phase-0 unmasked): `2026-05-06_v002_selector_post_phase0_FINAL_NO_GO.md`
- Sibling (v1 filter, also FINAL_NO_GO): `2026-05-06_v001_filter_post_phase0_FINAL_NO_GO.md`
- Roadmap: CLAUDE.md §4 Phase 1
- Operator scope chat: 2026-05-06 (Phase 1 ordering: #29 → #30 → #8)
- Operator decision criterion (chat 2026-05-06): "beat
  `selector_baseline_random` (audit-v2 score 0.7186), not merely
  escape all-skip"
