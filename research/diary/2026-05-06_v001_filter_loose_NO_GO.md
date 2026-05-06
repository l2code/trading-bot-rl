# RESEARCH-001 — v001 filter (loose candidate config) on yfinance starter_equities

> **PROVISIONAL** as of 2026-05-06 evening. Three P1 simulator/
> evaluation bugs were identified by code review after this verdict
> was recorded ([#22](https://github.com/l2code/trading-bot-rl/issues/22)
> position-size doesn't scale return,
> [#23](https://github.com/l2code/trading-bot-rl/issues/23) round-trip
> costs subtracted once,
> [#24](https://github.com/l2code/trading-bot-rl/issues/24) walk-
> forward lacks lookback warmup). Quantitative metrics in this entry
> are NOT trustworthy until those land and the run is repeated.
> Qualitative conclusion ("trained model is statistically identical
> to baseline_always_take_100") may survive but is not confirmed.

**Date:** 2026-05-06
**Verdict:** **NO_GO** (provisional)
**Issue:** [#2](https://github.com/l2code/trading-bot-rl/issues/2)
**Variant:** `filter_v001`
**Run:** Kaggle `crazypenguin/rl-swing-v001-500k-3seeds-loose`
**Trainer commit:** `0f7c18b`

---

## Question

Does the v1 trade-filter PPO produce a policy that beats
`baseline_always_take_100` on the 2022 yfinance test window after
the loose-candidate + symmetric-mirror reward changes?

## Source

- **Provider:** yfinance (daily bars, auto-adjusted).
- **Tier:** **exploratory** — per `docs/data_tiers.md`, yfinance
  cannot earn a GO verdict regardless of result. A NO_GO at this
  tier is conclusive against the framing; a GO would require
  canonical (WRDS) replication before any decision authority.
- **Universe:** `starter_equities` (15 names + SPY, QQQ).

## Methodology

- **Variant:** `filter_v001` — per-candidate filter, dedupe by
  `(symbol, date)` keeping highest signal_strength, action space
  `Discrete(4)` (skip / take_25 / take_50 / take_100).
- **Window:** train 2014-01-01..2020-12-31, validation 2021,
  test 2022.
- **Seeds:** 11, 22, 33.
- **Hyperparams:** PPO MlpPolicy 128×128, n_envs=4 (SubprocVecEnv
  with `start_method='fork'`), 500k timesteps, default sb3 PPO
  hyperparams (`ent_coef=0.01`, `lr=3e-4`, `n_steps=2048`).
- **Reward params:** `skip_counterfactual_scale=1.0` (full mirror),
  `turnover_penalty_weight=0.30` (per-trade cost),
  `drawdown_penalty_weight=0.10`, `holding_period_penalty_weight=0.05`,
  `target_risk_pct=0.02`, `reward_clip=5.0`.
- **Candidate strategies (loose config):**
  - `MomentumStrategy(min_relative_strength=-0.05, min_r20=-0.02, require_sma200_above=False)`
  - `RsiMeanReversionStrategy(rsi_threshold=35.0)`
  - `BreakoutStrategy(min_relative_volume=0.7, max_distance_below_high=-0.02)`
- **Cost layer:** `EquityExecutionModel` active (base spread 3 bps,
  slippage 5 bps, market impact 0.10, adverse selection 2 bps).
- **Aggregation:** `StrategyAggregator` deduplication.
- **Number of test candidates:** 477 (vs 311 before loosening,
  ~54% wider pool).

## Headline metrics (Phase-24 5-vector)

Test window 2022-01-01..2022-12-31. Comparison against the
**strongest baseline** by validation_composite_score, which on
this test window is `baseline_always_take_100` (score 0.6907).
`baseline_random` scores 0.6672 (a much weaker bar) and
`baseline_never_take` scores 0.3250 (trivially weak).

| Metric              | Trained PPO | Baseline | Delta | Improved? |
|---------------------|-------------|----------|-------|-----------|
| total_return        | +1109.47    | +1109.05 | +0.42 | tie       |
| annualized_sharpe   | +2.04       | +2.04    | 0     | tie       |
| profit_factor       | (clipped)   | (clipped)| 0     | tie       |
| max_drawdown        | 0.202       | 0.202    | 0     | tie       |
| turnover_take_rate  | 0.973       | 0.973    | 0     | tie       |
| **n_trades**        | 467         | 467      | 0     | (matches) |
| **score (composite)** | 0.6907    | 0.6907   | 0     | tie       |

**0 of 5 metrics improved.** Phase-24 gate (≥2 of 5) not met by
any reasonable margin. Verdict: **NO_GO**.

## Eval-history tell

The training-time eval history confirms the policy converges at
the very first checkpoint (step 50000) and never moves:

- All 30 evaluation points (3 seeds × 10 checkpoints) report
  `n_trades=477`, `take_rate=1.000`, `score=0.6900`.
- Returns rotate through only **3 floating-point values** (5081.06,
  5083.03 across the 30 evals) — entirely accounted for by the
  random-window sampler picking deterministic episodes.
- Zero seed-level variance.

That is a textbook entropy-collapse signature. PPO landed at "always
take" on rollout 1, the gradient pushed it deeper into that point,
and the default `ent_coef=0.01` was insufficient to keep exploration
alive.

## Known limitations

1. **Single test cycle.** One year (2022) is one realization. Until
   we have multi-cycle walk-forward (issue #5), this NO_GO might be
   year-specific. yfinance 2022 in particular is a notably bearish
   regime (QQQ -33%) which could push results either way.
2. **Default hyperparams.** No Optuna sweep on `ent_coef` or `lr`
   yet. The collapse pattern suggests entropy is at fault more than
   architecture; Optuna sweep is filed as #8 to test that hypothesis.
3. **Exploratory tier.** yfinance has dividends/splits adjustments
   that may differ from CRSP, plus survivorship in the
   starter_equities list.
4. **Three seeds.** Variance estimates are weak.

## Root-cause framing

Two competing hypotheses for why the filter framing produces no
lift:

**H1 — candidate-set EV is too positive.** The strategy stage
already filters aggressively enough that the residual candidate
distribution has avg per-trade return ≈ +1.3%. Under any reasonable
cost-and-risk-adjusted reward, "always take" is the EV-optimal
policy. The filter has nothing meaningful to do.

**H2 — entropy collapse.** The default `ent_coef=0.01` is not high
enough to keep the policy stochastic during early training. PPO
converges to a deterministic point at step 50000 before exploring
alternatives. Higher entropy or a learning-rate schedule might
unlock learning that genuinely improves on always-take.

These are not mutually exclusive. The `selector_v002` run pending
in [#3](https://github.com/l2code/trading-bot-rl/issues/3) tests
H1 (different framing on same data). Optuna sweep #8 tests H2.

## What would change my mind

- **Reverse to GO:** A canonical (WRDS) replication of this run
  showing the trained PPO beats baseline by ≥2 of 5 metrics on
  multi-cycle walk-forward.
- **Reverse to SHADOW_ONLY:** A v2 selector or post-Optuna v1 run
  on the same yfinance data that beats the gate; we'd need WRDS
  replication before any further authority.

## Verdict

**NO_GO** for `filter_v001` as currently configured on yfinance
starter_equities. Recommend:

1. Wait for `selector_v002` results (#3).
2. Run Optuna sweep on `ent_coef` and `lr` (#8).
3. Wire WRDS canonical data (#4) before any GO call on either
   variant.
4. Begin multi-cycle walk-forward (#5) before claiming any
   year-specific result generalizes.
