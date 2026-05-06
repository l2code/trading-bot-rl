# ADR 0006 — Research, shadow, paper, and live share one pipeline

Status: Accepted (Phase 0A)
Date: 2026-05-06

## Context

If research and production take divergent code paths, "but it worked
in backtest" becomes the standard incident root cause. The reference
bot's monolithic-script style makes this trap easy to fall into.

## Decision

The 12-step decision pipeline runs identically in every runtime mode
(see `services/pipeline.py`):

1. Resolve config / universe.
2. Load bars through the configured `MarketDataProvider`.
3. Build `FeatureFrame`s through the configured `FeaturePipeline`.
4. Generate `CandidateTrade`s through configured `CandidateStrategy`s.
5. Score each candidate through the configured `PolicyScorer`.
6. Run `RiskPolicy` rules in the configured order.
7. Emit `OrderIntent`s for approved trades.
8. Submit to the configured `BrokerAdapter`.
9. Persist all domain events.
10. Reconcile.
11. Emit alerts.
12. Daily report.

Differences between modes are only:

* The `MarketDataProvider` adapter (historical-replay vs. latest data).
* The `BrokerAdapter` adapter (simulated vs. no-op vs. Alpaca).
* The active risk profile.
* The two safety flags (`place_orders`, `allow_live_trading`).

## Consequences

* Adding a new mode is a config; no new pipeline code.
* Reconciliation tests written once cover all modes.
* Walk-forward validation is "research mode with a chronological
  episode sampler"; shadow is "research mode pointing at recent
  data"; paper is "research mode with the Alpaca adapter".
