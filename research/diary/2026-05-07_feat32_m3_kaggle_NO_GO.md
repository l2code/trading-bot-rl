# FEAT-32 M3 — Kaggle masked-PPO 500k×3 on v003 chronological env

**Date:** 2026-05-07
**Verdict:** **NO_GO** — trained MaskablePPO is bit-identical to `portfolio_baseline_no_op` across all 3 seeds.
**Issue:** [#89](https://github.com/l2code/trading-bot-rl/issues/89) FEAT-32 M3
**PR (masking infra):** [#90](https://github.com/l2code/trading-bot-rl/pull/90) — merged at SHA `cb60d3b`
**PR (this verdict):** TBD
**Kernel:** `crazypenguin/rl-swing-v003-masked-m3` (private)
**Run:** Kaggle private, repo @ `main` SHA `cb60d3b`, 3 seeds × 500k × n_envs=4
**Wall-time:** 5689s (~95 min) — about 2× the operator's ~45 min estimate; the chronological env's per-day stepping has more overhead than v002's per-pack stepping (portfolio bookkeeping every step, even on no-op days).

---

## Why this diary exists

M2 (PR #88) PASSED — the supervised classifier recovered a non-trivial state-dependent target with val_acc=1.00, ruling out env degeneracy. M3 was the RL training step: does PPO discover a profitable policy by RL, given that supervised methods can fit one by imitation? The bit-identity check vs each portfolio baseline (per the PR #71 fingerprint pattern) is the primary diagnostic; the Phase-24 metric gate is secondary.

## Numbers

```
model_id                                    composite     return    sharpe       PF       DD     turn   n_t   per_action
portfolio_baseline_no_op                    +0.325000   +0.0000   +0.0000   1.000   0.000   0.000     0   [511,   0,   0]
portfolio_baseline_top1                     -0.038124   -0.3009   -6.8310   0.294   0.316   0.260   133   [  0, 511,   0]
portfolio_baseline_top2                     -0.049512   -0.3159   -6.7304   0.274   0.337   0.255   139   [  0,   0, 511]
portfolio_baseline_random_action            -0.018194   -0.2779   -6.0697   0.293   0.284   0.217   116   [192, 148, 171]
ppo_portfolio_v003 (trained MaskablePPO)    +0.325000   +0.0000   +0.0000   1.000   0.000   0.000     0   [511,   0,   0]
```

**Per-seed `best_validation_score` from `summary.json`:**
- seed 11: 0.32499999999999996
- seed 22: 0.32499999999999996
- seed 33: 0.32499999999999996

All three seeds converged to *identical* validation composite score, exact to 16 decimals. The eval history within each seed (5 evaluations at 50k/100k/150k/200k/250k timesteps) is also identical: every checkpoint scored 0.325000.

## Bit-identity check (the gate criterion that fails)

The trained MaskablePPO is bit-identical to `portfolio_baseline_no_op`:

```
                              trained == no_op    ?
composite (6 decimals)        0.325000 == 0.325000     YES
total_return                  +0.0000 == +0.0000        YES
max_drawdown                  +0.0000 == +0.0000        YES
per_action_counts             [511,0,0] == [511,0,0]    YES
                                                     ━━━━━━━━━━━━━
                                                     BIT-IDENTICAL
```

Vs every other baseline: NOT bit-identical (different actions, different metrics).

## Phase-24 metric gate (secondary, since bit-identity fails the primary)

Vs `portfolio_baseline_random`:

| metric        | trained | random  | delta   | improved? |
|---------------|---------|---------|---------|-----------|
| composite     | +0.325  | -0.018  | +0.343  | yes       |
| total_return  | 0.000   | -0.278  | +0.278  | yes       |
| sharpe        | 0.000   | -6.070  | +6.070  | yes       |
| profit_factor | 1.000   | 0.293   | +0.707  | yes       |
| max_drawdown  | 0.000   | 0.284   | -0.284  | yes (lower DD) |

5 of 5 metrics improved, 0 material regressions. **By the metric gate alone, M3 would PASS.** But the gate also requires "**AND not bit-identical to any portfolio_baseline**" — and the trained policy IS bit-identical to no_op. The metrics improved trivially by zeroing them out via all-skip.

## Verdict reasoning

The M3 acceptance criteria explicitly require: composite > random's, ≥2 of 5 metrics improved, no material regression, **AND not bit-identical to any portfolio_baseline**. The last clause is the v002 lesson encoded: in v002, masked-PPO's per-metric gate also "passed" but the policy was bit-identical to `selector_baseline_first_fired`, which made the win meaningless. The same pattern repeats here in v003: **MaskablePPO learned the simplest possible policy (no_op) and stuck with it** for all 3 seeds across 500k steps each.

This is consistent with v002's hyperparameter exhaustion finding (Phase 1 closure, FEAT-7 tie-breaker NO_GO). At default `ent_coef=0.01`, PPO has insufficient exploration on a 3-element action space to escape the all-skip attractor when most days have negative reward expectations under the configured cost model (random action gives -0.278 return; top1 gives -0.301; top2 gives -0.316). The Bellman-optimal action *is* no_op on this benchmark — and PPO finds it without ever sampling the alternative actions enough to discover any profitable subset of states.

Three readings:

1. **Masking didn't matter.** The action lattice is `Discrete(3)` and the env's `step` already clamps invalid actions to no-op. Masking changed the policy's gradient signal slightly but didn't alter the convergence point. v002's masked-PPO was bit-identical to first_fired; v003's is bit-identical to no_op. Both "won" the metric gate trivially.
2. **Default exploration is insufficient on small action lattices over negative-EV environments.** When the no-trade option is one of only 3 choices and has the highest expected reward (because trading at this cost level is unprofitable), PPO converges to it within 50k steps and never leaves. The eval history confirms: seed 11/22/33 hit 0.325000 by their first eval and stayed there.
3. **The architectural shift (per-day chronological vs per-pack contextual-bandit) does not solve the v002 problem.** v002 NO_GO was attributed to architectural mimicry of first_fired. v003 NO_GO is now architectural mimicry of no_op. The structural complexity (per-day stepping, portfolio bookkeeping, drawdown penalty) buys nothing if PPO can't be coaxed to explore. The primary obstacle is exploration, not architecture.

## Wall-time honesty

The operator's stated estimate was ~45 min Kaggle. Actual: ~95 min. Two factors:

- v003 chronological env steps once per trading day with portfolio bookkeeping (mark-to-market every open position, drawdown computation). v002 stepped once per pack with stateless portfolio. v003 per-step is roughly 2× the work of v002.
- The packer + slate-by-day pre-computation runs once per env construction; with sb3 SubprocVecEnv n_envs=4 + 3 seeds = 12 env constructions across the run.

This timing should be carried forward as a calibration update — v003 Kaggle runs should budget ~90-100 min for 500k×3 with n_envs=4, not 45 min.

## What this does NOT close

- **Does not say "v003 is bad."** It says "v003 + default-hyperparam MaskablePPO is bad." The same architecture with different exploration (higher ent_coef, lower lr, or distributional-RL alternatives) might still find a profitable policy. M2 PASS established that the obs space is rich enough; what's missing is the optimization signal to use it.
- **Does not refute M2.** Supervised BC fit a deterministic target with val_acc=1.00; the learnability floor is real. PPO's failure here is policy gradient + exploration, not representation.
- **Does not eliminate v003.** Closes M3 default-hyperparam compute. M3.b (Optuna ent_coef + lr sweep) is the natural next step per the FEAT-32 plan.

## Code artifacts

- `data/kaggle/rl-swing-v003-masked-m3/output/validation_summary.json` — full per-policy metric block (gitignored under `data/kaggle/`).
- `data/kaggle/rl-swing-v003-masked-m3/output/summary.json` — per-seed train history.
- Kernel: `crazypenguin/rl-swing-v003-masked-m3` (private, will remain on Kaggle for traceability).

## Decision points

1. **(Recommended) M3.b: Optuna sweep on ent_coef + lr.** Same shape as the open issue #27 RESEARCH-8 for v002. Specifically: sweep `ent_coef ∈ {0.01, 0.05, 0.1, 0.2}` and `learning_rate ∈ {1e-4, 3e-4, 1e-3}` × 3 seeds × 200k steps each; cheap diagnostic to test whether higher entropy unlocks the exploration the default failed at. ~3-4 hr Kaggle.
2. **Alternative: shape the reward to penalize all-skip more aggressively.** Currently the no-trade reward is 0; turnover penalty is 0.05. If the no-trade absorbing state is too cheap, PPO won't have a gradient pushing it to try anything. Could add a small "stagnation penalty" to the no-trade case. Risk: introduces another tuning knob and could mask the real problem.
3. **Accept v003 NO_GO under default hyperparams; close FEAT-32.** Mirrors how Phase 1 closed v002 default-hyperparam masked-PPO as NO_GO. v003's architectural value (per-day stepping, portfolio state) becomes a research artifact rather than a deployed strategy. M4 (multi-cycle yfinance per D4-b harness) is moot under this option since 2022 already shows NO_GO.

**Recommendation: option 1 (M3.b Optuna sweep), gated on operator approval.** The exploration-vs-architecture diagnostic is what M2 PASS earned us — we now know that any v003 collapse is hyperparam, not architecture. M3.b would either confirm exploration-only or demonstrate that no hyperparam configuration in the swept range escapes no_op, which would close v003 cleanly with multi-config evidence.

## Honest annotations

- **Why all 3 seeds got identical 0.325000 to 16 decimals:** the no-op action produces *exactly* zero P&L (`daily_pnl = 0` because no positions opened, no DD, no turnover penalty), and the validation composite components map zero-pnl to a fixed point: `n_total_return=0.5`, `n_sharpe=0.333`, `n_profit_factor=0.333`, `n_max_drawdown=0.0`, `n_turnover=0.0`, weighted sum = 0.325. There's no floating-point drift across seeds because the deterministic no-op policy produces deterministic env outputs. This is actually a clean diagnostic, not a bug — it means three independent training trajectories all converged to the *same* attractor.
- **Why the operator's ~45 min estimate was off:** see "Wall-time honesty" above. Calibration: v003 ≈ 2× v002 per-step.
- **Why this isn't an "exploration is too low" hand-wave:** the per-action distribution is exactly [511, 0, 0] — the policy never tried action 1 or action 2 at any point in the 251 evaluation days. With ent_coef=0.01 over 500k training steps, the policy entropy collapsed before sampling any productive trajectories.
