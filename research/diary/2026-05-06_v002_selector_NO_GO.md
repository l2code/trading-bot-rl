# RESEARCH-003 — v002 selector on yfinance starter_equities

> **PROVISIONAL** as of 2026-05-06 evening. Five simulator/evaluation
> issues were identified by code review after this verdict was
> recorded — three P1 affecting both v1 and v2
> ([#22](https://github.com/l2code/trading-bot-rl/issues/22),
> [#23](https://github.com/l2code/trading-bot-rl/issues/23),
> [#24](https://github.com/l2code/trading-bot-rl/issues/24)) and two
> P2 affecting v2 specifically ([#25](https://github.com/l2code/trading-bot-rl/issues/25)
> selector not in runtime,
> [#26](https://github.com/l2code/trading-bot-rl/issues/26) skip
> reward uses hindsight-best max-over-noise counterfactual).
> Quantitative metrics here are NOT trustworthy until at least the
> three P1s land. The "Momentum specialist" qualitative collapse
> finding is independent of cost/warmup bugs and may survive, but
> the per_strategy_take_counts may shift after #26.

**Date:** 2026-05-06
**Verdict:** **NO_GO** (provisional)
**Issue:** [#3](https://github.com/l2code/trading-bot-rl/issues/3)
**Variant:** `selector_v002`
**Run:** Kaggle `crazypenguin/rl-swing-v002-selector-500k-3seeds`
**Trainer commit:** `af86e06` (variant + selector landed); evaluated
on `8d1d91b` main.

---

## Question

Does the v2 multi-strategy selector produce a policy that beats
(a) v1's `filter_v001` and (b) `selector_baseline_highest_signal`
(the cross-strategy-aware baseline) on the 2022 yfinance test
window?

## Source

- **Provider:** yfinance (daily bars, auto-adjusted).
- **Tier:** **exploratory** — per `docs/data_tiers.md`, yfinance
  cannot earn a GO verdict regardless of result. NO_GO at this
  tier is conclusive against the framing; GO would require
  canonical (WRDS) replication before any decision authority.
- **Universe:** `starter_equities` (15 names + SPY, QQQ).

## Methodology

- **Variant:** `selector_v002` — per-(symbol, date) decisions over
  `Discrete(N+1)` actions (skip + N strategies, N=3). Strategies
  packed without dedupe so confluence info reaches the agent.
- **Window:** train 2014-01-01..2020-12-31, validation 2021,
  test 2022. Same as v001 for clean comparison.
- **Seeds:** 11, 22, 33.
- **Hyperparams:** PPO MlpPolicy 128×128, n_envs=4 (SubprocVecEnv
  fork), 500k timesteps, default sb3 PPO (`ent_coef=0.01`,
  `lr=3e-4`).
- **Reward params:** `skip_counterfactual_scale=1.0`,
  `turnover_penalty_weight=0.30`, `illegal_action_penalty=0.5`.
- **Strategy-pack config:** same loose thresholds as v1 (Momentum
  with `min_r20=-0.02`, no SMA-200 gate, RS≥-0.05; Mean-reversion
  RSI≤35; Breakout dist_high≥-0.02, vol≥0.7×).
- **Cost layer:** `EquityExecutionModel` active.
- **Test packs:** 480 (each pack = one `(symbol, date)` where
  ≥1 strategy fired).

## Headline metrics (Phase-24 5-vector)

Comparison **against the strongest selector baseline** by
`validation_composite_score`:
- `selector_baseline_random`: **0.7037** ← strongest
- `selector_baseline_always_skip`: 0.3250 (trivial)
- `ppo_selector_v002` (trained): **0.6665** ← below the random baseline

Gate output (programmatic, via `acceptance_gate.evaluate_gate`):

| Metric              | Trained v2 | Strongest baseline (random) | Delta    | Status |
|---------------------|------------|------------------------------|----------|--------|
| total_return        | +438.99    | +255.09                      | +183.90  | ✓ improved |
| annualized_sharpe   | +2.48      | +2.33                        | +0.15    | ✓ improved |
| profit_factor       | +2.35      | +2.29                        | +0.07    | ✓ improved |
| max_drawdown        | 0.333      | 0.212                        | +0.121   | **✗ MATERIAL regression** (>0.05 threshold) |
| turnover_take_rate  | 0.673      | 0.652                        | +0.021   | ✓ improved (informational) |

**4 of 5 improved, 1 material regression. Phase-24 gate caps
verdict at NO_GO regardless of improvement count.**

Cross-variant comparison vs v1 trained (`ppo_filter_v001`):

| Metric              | v2     | v1     | Status |
|---------------------|--------|--------|--------|
| total_return        | 438    | 1109   | **✗ material** (-670, threshold 0.05) |
| annualized_sharpe   | 2.48   | 2.04   | ✓ improved (+0.44) |
| profit_factor       | 2.35   | ~999   | **✗ material** (PF regression) |
| max_drawdown        | 0.333  | 0.202  | **✗ material** (+0.131) |
| turnover_take_rate  | 0.673  | 0.973  | informational |

**1 of 5 improved (sharpe only), 3 material regressions. NO_GO
vs v1 by a wide margin.**

## What v2 actually learned (the wrinkle)

Unlike v1 (which collapsed to "always take everything"), v2
collapsed to **"Momentum specialist"**:

```
ppo_selector_v002:   per_strategy_take_counts = [323, 0, 0]
```

- 100% of taken actions go to strategy 0 (Momentum).
- Zero to Breakout, zero to RSI Mean Reversion.
- Skip ~33% of packs (the ones where Momentum doesn't fire).

This is a real, non-trivial behavior. The agent learned a
discriminating policy — just one that's worse than random across
strategies. `selector_baseline_random` distributes
`[145, 41, 127]` across the three strategies and beats the
trained model on composite score.

## Eval-history tell

The training-time eval history shows characteristic entropy
collapse:

| Seed | First "stuck" step | Stuck score | Stuck n_trades | Stuck take_rate |
|------|---------------------|-------------|----------------|-----------------|
| 11   | 100000              | 0.6787      | 336            | 1.000           |
| 22   | 100000              | 0.6787      | 336            | 1.000           |
| 33   | 50000               | 0.6787      | 336            | 1.000           |

After the convergence step, every metric is bit-identical across
all subsequent checkpoints. Note: the training-eval `take_rate=1.0`
differs from the walk-forward `take_rate=0.673` because the
training eval runs on the *validation* window 2021 with a
different candidate distribution than test 2022.

Best-checkpoint score (0.6887) is slightly higher than the stuck
score (0.6787) because at step 50000 seeds 11 and 22 had a
take_rate < 1.0 (n_trades=257); they then converged upward to
take_rate=1.0 and lost a small amount of score.

## Known limitations

1. **Single test cycle** (#5). Result may be 2022-specific.
2. **Default PPO hyperparams** — Optuna sweep on `ent_coef` (#8)
   has not run; entropy collapse hypothesis untested.
3. **Exploratory tier.** yfinance cannot earn GO; canonical
   replication (#4) is a prerequisite for any decision authority.
4. **Three seeds.** Variance estimates weak.
5. **No data analysis of per-strategy training EV.** Why
   Momentum specifically? Filed as RFC #15.

## Root-cause framing

Two competing hypotheses, in order of evidence:

**H1 (entropy collapse, strong evidence).** The data signature is
textbook: convergence at step 50000-100000, zero variance after,
random policy beats trained policy. PPO's default `ent_coef=0.01`
is insufficient for this domain. Issue #8 (Optuna sweep) tests
this directly.

**H2 (Momentum-rationality on training data, less evidence).** If
Momentum-only candidates have markedly higher EV on the 2014-2020
training window than the other two strategies, the agent's
specialization is rational on its training distribution and just
doesn't generalize to 2022. Issue #15 (RFC) proposes a quick data
analysis to test this *before* any RL change.

These are not mutually exclusive. Both could be true.

## What would change the verdict

- **Reverse to GO:** canonical (WRDS) replication of v2 with
  hyperparams from #8 sweep, multi-cycle WF (#5), trained model
  beats strongest baseline by ≥2 of 5 metrics with no material
  regression on any cycle's max_drawdown.
- **Reverse to SHADOW_ONLY:** the same conditions on yfinance,
  pending canonical replication.

## Cross-variant summary so far

| Variant | Tier | Verdict | Collapse mode |
|---------|------|---------|----------------|
| `filter_v001` (loose) | exploratory (yfinance) | NO_GO | "always take everything" |
| `selector_v002`       | exploratory (yfinance) | NO_GO | "always take Momentum" |

Both variants converged to degenerate policies that fail to
discriminate within the positive-EV candidate pool. Different
shapes of failure, same wall: the framing alone (filter vs
selector) does not unlock learning on this candidate distribution
under default PPO hyperparams.

## Recommendation

The data points entropy-collapse over architecture-deficiency:
the random-beats-trained signature is more diagnostic than the
v1-vs-v2 divergence. Next experiment is **#8 — Optuna sweep on
`ent_coef` and `learning_rate`** against v2 (the variant with
more interesting policy structure). If Optuna unsticks the
collapse and the post-tuning trained model beats baseline_random,
write a SHADOW_ONLY diary entry and proceed to canonical
replication (#4). If Optuna fails to unstick, the framing itself
is the wall — pivot to Tier 2 (action masking via MaskablePPO,
RecurrentPPO, per-symbol embeddings).

Issue #15 (RFC on Momentum specialization) is a cheap parallel
diagnostic — run the per-strategy training-EV analysis before
or during the Optuna sweep.
