# Data tier policy

Distilled from `SDLC_LESSONS_FOR_NEW_PROJECT.md` §1.4 and §4.1.
Mixing tiers without labeling them is the single most common
research-integrity failure. Every research artifact and every
Kaggle run names both its data **provider** and its **tier**.

## Tiers

| Tier                  | Providers                              | Decision authority                                |
|-----------------------|----------------------------------------|---------------------------------------------------|
| **canonical**         | `wrds_parquet` (CRSP via WRDS)         | Decision-grade. GO verdicts must come from here.  |
| **execution-realism** | (future: Databento intraday + bid/ask) | Slippage & cost-model calibration; not strategy decisions. |
| **exploratory**       | `yfinance_daily`, `synthetic_*`        | Quick-look only. Cannot earn a GO verdict.        |

## Rules

1. **Provider name encodes the tier.** Don't introduce providers
   that span tiers. If a provider can serve multiple tiers, name
   the variants explicitly (e.g., `wrds_parquet` vs
   `yfinance_daily` — both are daily bars, but the tier is
   different).

2. **Diary entries declare tier.** The research diary template
   (`research/diary/`) requires a "Source" section that names
   provider and tier. A diary entry without a tier label is
   incomplete.

3. **Exploratory tier verdict ladder is capped.** A run on
   exploratory data can produce {NO_GO, SHADOW_ONLY} but never GO.
   GO requires canonical replication.

4. **Synthetic data is exploratory.** `synthetic_momentum`,
   `synthetic_mean_reversion`, `synthetic_random_walk` are useful
   for smoke tests and unit tests. They cannot validate a strategy.

## Why this matters

A backtest that mixes tiers silently is the kind of thing that
becomes a year-long sunk cost. trading-bot2 §1.4 makes this point
plainly:

> The cost of skipping: a year of accumulated research that nobody
> can retroactively trust. Re-running everything on canonical data
> is the only fix; expensive.

We avoid this by making the tier label part of every artifact
from day one.

## Current state

- `yfinance_daily` is wired and works on Kaggle (auto-downloads at
  run time).
- `synthetic_*` providers are wired and work everywhere.
- `wrds_parquet` is wired locally (reads from
  `/home/rissac/projects/trading-bot2/cache/wrds`) but **not
  accessible to Kaggle workers**. This is structural debt #1 in
  CLAUDE.md §6 and is tracked as issue #4.

Until #4 lands, every Kaggle run is exploratory. NO_GO results
from Kaggle are conclusive against the framing; GO results require
canonical replication before any decision authority.

## When to escalate to canonical

Per `SDLC_LESSONS_FOR_NEW_PROJECT.md` §4.1:

| Stage | Gate to advance |
|-------|-----------------|
| Exploratory | Operator says "interesting; let's investigate" |
| Methodology | Single canonical artifact in `research/diary/` |
| Decision-grade | Canonical + walk-forward + cost layer + ≥2-of-5 gate met |

Each stage costs more compute and more operator time. Escalate only
when the previous stage's NO_GO/SHADOW_ONLY/GO has been recorded
durably.
