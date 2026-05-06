# ADR 0004 — Parquet for bulk numerics, relational DB for events

Status: Accepted (Phase 0)
Date: 2026-05-06

## Context

We have two very different storage workloads:

1. Bulk numerics (bars, feature matrices, backtest output) — append-
   heavy, read-heavy, columnar-friendly.
2. Operational records (candidates, decisions, orders, fills,
   positions, audit events, reconciliation breaks) — relational,
   need ACID, need cross-table joins.

## Decision

* Bulk numerics live as parquet files under `data/cache/`.
* Operational records live in SQLite (development) or PostgreSQL
  (production). The relational schema referenced in the spec is
  authoritative.
* The relational store does not duplicate raw numerics — it stores
  hashes (`feature_id`, `observation_hash`, `snapshot_id`) that point
  back to parquet files / model artifacts.
* Repository adapters under `rl_swing.adapters.storage.*` hide the
  storage engine behind the port interfaces; swapping SQLite for
  Postgres is one config change.

## Consequences

* Audit queries stay cheap because they don't carry feature payloads.
* Backtest batch I/O is fast because it's columnar.
* The same code runs against SQLite locally and Postgres in
  production — only the connection URL changes.
