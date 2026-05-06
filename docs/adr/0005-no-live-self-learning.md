# ADR 0005 — No live self-learning

Status: Accepted (Phase 0A)
Date: 2026-05-06

## Context

A self-learning agent that updates from live broker fills mixes two
hard problems: distributional shift in market data and credit
assignment under noisy rewards. It also makes incident recovery much
harder — a buggy reward could destabilize the policy in hours.

## Decision

* Models only update inside the offline RL training environment.
* In shadow / paper / live runtimes the policy is **frozen**:
  PPO/DQN run in deterministic inference mode with no gradient steps.
* Promotion from `TRAINED` → `VALIDATED` → `SHADOW_APPROVED` →
  `PAPER_APPROVED` → `LIVE_APPROVED` is manual; the `ModelRegistry`
  records who approved what when.

## Consequences

* The runtime never imports an optimizer at scoring time.
* A regression in production is rolled back by registry transition,
  not by retraining.
* The `allow_live_retraining` flag exists in the config schema and is
  hard-failed by tests; keeping it in the schema makes the prohibition
  explicit rather than implicit.
