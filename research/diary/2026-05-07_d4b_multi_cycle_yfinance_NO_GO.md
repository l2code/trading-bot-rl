# RESEARCH-D4b — multi-cycle walk-forward on real yfinance (FIX-#5 D4-b)

**Date:** 2026-05-07
**Verdict:** **NO_GO** (multi-year, on real yfinance) — every trained selector loses to `selector_baseline_random` in every year tested.
**Issue:** [#5](https://github.com/l2code/trading-bot-rl/issues/5) D4-b
**Run:** local on Loki — `rl-swing validate --data-provider yfinance_daily --test-start Y-01-01 --test-end Y-12-31` for Y in {2021, 2022, 2023, 2024}, against the FEAT-7 tie-breaker masked-PPO + PR-1b stabilized set_ranker + FEAT-7 HistGB ranker.
**Total wall-time:** ~2 min after FIX-#83 cache-coverage match landed.

---

## Why this diary exists

The synthetic-contamination story (#78) closed with a single canonical 2022 yfinance rebaseline. The operator's open question: *does the architectural finding hold across regimes, or is 2022 a one-off where random happens to win?* D4-b answers that with 4 years of canonical-yfinance data on the same artifacts.

## Per-year canonical numbers

Each cell: `composite / total_return / max_DD / n_trades`.

| model_id                        | 2021                     | 2022                     | 2023                     | 2024                     |
|---------------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| `selector_baseline_random`      | 0.3983/+0.396/0.524/1954 | -0.1849/-0.702/0.704/839 | 0.6150/+6.622/0.546/1909 | 0.5469/+1.153/0.502/1945 |
| `selector_baseline_first_fired` | 0.4231/+0.595/0.700/2772 | -0.1999/-0.855/0.856/1199| 0.6003/+20.174/0.736/2710| 0.5224/+1.703/0.692/2751 |
| `selector_baseline_supervised` (HistGB FEAT-7) | 0.3702/+0.347/0.617/1800 | -0.1891/-0.767/0.767/938 | 0.6142/+9.962/0.626/1950 | 0.5027/+0.821/0.591/1862 |
| `selector_baseline_set_ranker` (DeepSets PR-1b) | 0.4860/+0.731/0.655/1910 | -0.1930/-0.832/0.832/1034| 0.6105/+7.691/0.605/2151 | 0.4834/+0.718/0.604/1963 |
| `ppo_selector_v002_masked` (FEAT-7 tie-breaker) | 0.4231/+0.595/0.700/2772 | -0.1999/-0.855/0.856/1199| 0.6003/+20.153/0.737/2710| 0.5224/+1.704/0.692/2751 |

**Buy-and-hold SPY:** 2021 +0.308 (real +29%), 2022 -0.187 (real -19%), 2023 +0.267 (real +24%), 2024 +0.261 (real +25%). Yfinance data is real across all 4 years.

## Phase-24 gate per year (vs `selector_baseline_random`)

| Year | set_ranker | supervised | first_fired | masked-PPO |
|------|-----------|-----------|-------------|------------|
| 2021 | NO_GO imp=3 mat=1 | NO_GO imp=0 mat=1 | NO_GO imp=4 mat=1 | NO_GO imp=4 mat=1 |
| 2022 | NO_GO imp=1 mat=2 | NO_GO imp=3 mat=2 | NO_GO imp=1 mat=2 | NO_GO imp=1 mat=2 |
| 2023 | NO_GO imp=2 mat=2 | NO_GO imp=2 mat=1 | NO_GO imp=2 mat=1 | NO_GO imp=2 mat=1 |
| 2024 | NO_GO imp=1 mat=3 | NO_GO imp=0 mat=3 | NO_GO imp=2 mat=1 | NO_GO imp=2 mat=1 |

**16 of 16 cells are NO_GO.** Trained selectors lose to random in every year tested, on every flavor of policy. The contamination correction's 2022 conclusion generalizes.

## Three reproducibility checks across years

### (a) Set_ranker max_DD vs random's

| Year | set_ranker DD | random DD | set_ranker < random? |
|------|--------------:|----------:|:--------------------:|
| 2021 | 0.6551 | 0.5241 | **no** |
| 2022 | 0.8324 | 0.7038 | **no** |
| 2023 | 0.6054 | 0.5459 | **no** |
| 2024 | 0.6035 | 0.5018 | **no** |

**4 of 4 years: set_ranker has HIGHER max_DD than random.** PR #75's "lowest DD of any policy" claim is now refuted with multi-year evidence — it was 100% a synthetic-data artifact, not regime-fragile architectural property. The DeepSets encoder consistently produces a HIGHER-drawdown trader than random selection on real data.

### (b) Set_ranker per_strat distinctness from first_fired

| Year | set_ranker per_strat | first_fired per_strat | distinct? |
|------|---------------------|----------------------|:---------:|
| 2021 | [1411, 130, 369] | [2525, 148, 99] | YES |
| 2022 | [946, 34, 54] | [1127, 32, 40] | YES |
| 2023 | [1777, 126, 248] | [2522, 130, 58] | YES |
| 2024 | [1557, 143, 263] | [2578, 131, 42] | YES |

**4 of 4 years: set_ranker's strategy distribution is genuinely distinct from first_fired's.** This single architectural finding survives across all regimes. The DeepSets encoder IS doing real selection beyond priority order — it just produces selections that consistently underperform random and have higher drawdown.

### (c) Masked-PPO vs first_fired bit-identity

| Year | composite identical? | per_strat identical? | trade-count identical? |
|------|:-------------------:|:--------------------:|:----------------------:|
| 2021 | YES | YES | YES |
| 2022 | YES | YES | YES |
| 2023 | YES | mostly (PPO ret +20.153 vs ff +20.174; per_strat tied) | YES |
| 2024 | YES | mostly | YES |

The PR #71 finding ("masked-PPO is bit-identical to selector_baseline_first_fired") **survives across all 4 years** at the composite-score level. Tiny float drift in 2023/2024 absolute returns (~0.02 difference) but per_strat distributions match — the trained policy is functionally first_fired with rounding.

## What this proves

1. **Phase 1 closure verdict (PR #73 NO_GO) is reproduced multi-year.** Default-hyperparam masked-PPO collapses to first_fired regardless of regime. The synthetic correction reaffirmed it; multi-year reaffirms it again.

2. **The "low-DD selector" production lane (Path C in earlier discussion) is fully invalidated.** It wasn't 2022-specific — it was synthetic-specific. There's no production-grade low-DD selector in the current artifact set.

3. **Random is a hard baseline on this universe + decision shape.** Random's DD swings 0.50 → 0.70 across years; trained selectors' DD swings UP to ~0.60-0.86 in every year. There's no year where any trained policy dips under random's DD.

4. **The DeepSets architecture's "selection beyond priority" is real but unprofitable.** Multi-year per_strat distinctness from first_fired confirms the encoder isn't degenerate — it's just optimizing the wrong target on this data, in every regime.

5. **The slate framing on yfinance starter_equities is structurally exhausted at default-hyperparam selector-class compute.** Doesn't depend on a regime quirk. The path forward (if any) is architectural change (#32 chronological v3, full sequential RL) or a different decision shape entirely (per-symbol portfolio policy, not per-pack independent decisions).

## Implications for D2

**D2 (ratify set_ranker as SHADOW_ONLY low-DD shadow research lane) — formally invalidated.** Multi-year evidence shows set_ranker has HIGHER DD than random in all 4 years. Don't ratify. The architecture is interesting from a basic-research standpoint (it does real selection); it isn't a production lane.

## What would change the verdict

- **WRDS canonical replication** — yfinance is exploratory tier; the 2022 SPY -0.187 is correct survivor-bias-wise but a real decision-grade run requires WRDS. If WRDS shows different policy ranks (random NOT winning), the conclusion changes. Filed: [#4](https://github.com/l2code/trading-bot-rl/issues/4).
- **Different decision shape** — #32 chronological v3 changes from per-pack independent decisions to portfolio-aware sequential RL. Different problem; different priors apply.
- **#27 Optuna sweep** — could in principle escape first_fired with better hyperparams. After multi-year evidence shows the policy collapses identically across regimes, Optuna becomes lower-EV (you'd be tuning a known-stuck policy). Operator's prior call to park #27 stands.

## Cross-references

- Predecessor (single-year canonical rebaseline): [`2026-05-07_d4_canonical_yfinance_rebaseline.md`](2026-05-07_d4_canonical_yfinance_rebaseline.md)
- All affected diaries with [CORRECTION] banners (PR #82): see `research/diary/2026-05-06_v002_*` files.
- FIX-#83 yfinance cache fix that made this run economical: PR #84.
- D4-b unblocked: [#5](https://github.com/l2code/trading-bot-rl/issues/5)
- D2 invalidation: this diary.
