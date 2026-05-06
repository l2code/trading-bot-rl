# ADR 0001 — Use ports-and-adapters architecture

Status: Accepted (Phase 0A)
Date: 2026-05-06

## Context

The system needs to swap data providers (yfinance, WRDS, Alpaca,
synthetic, parquet cache), brokers (simulated, no-op shadow, Alpaca
paper, Alpaca live), and RL algorithms (PPO, DQN, ensemble) without
rewriting the surrounding code. It also needs the *same* decision
pipeline to run in research, shadow, paper, and live modes.

## Decision

We organize the codebase as a hexagonal (ports-and-adapters)
architecture:

* `rl_swing.domain` holds frozen dataclasses for the handoff types
  (MarketBar, FeatureFrame, CandidateTrade, PolicyDecision, …).
* `rl_swing.ports` holds `typing.Protocol` interfaces for everything
  replaceable (MarketDataProvider, FeaturePipeline, CandidateStrategy,
  PolicyScorer, RiskPolicy, BrokerAdapter, repositories, EventBus,
  ModelRegistry).
* `rl_swing.adapters` holds concrete implementations grouped by
  category (`adapters/data/*`, `adapters/broker/*`, `adapters/storage/*`).
* `rl_swing.runtime` wires it all up via a YAML component registry.
  Service code asks the registry for `policy_scorers.ppo_filter_v001`
  rather than importing the class directly.

## Consequences

* Adding WRDS or a new broker is a new adapter file plus a registry
  entry — service code does not change.
* Baselines (random, always-take, momentum-rule) are just other
  `PolicyScorer` adapters, so validation compares them through the
  same harness as the RL policies.
* No vendor SDK (yfinance, WRDS, Alpaca, alpaca-py) may be imported
  outside `rl_swing.adapters.*`. Contract tests guard this.
