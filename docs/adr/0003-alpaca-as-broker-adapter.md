# ADR 0003 — Alpaca as a broker adapter, not a runtime

Status: Accepted (Phase 0A)
Date: 2026-05-06

## Context

The reference bot wires its training and execution code directly to
MetaTrader 5 / MetaAPI. That coupling makes the broker hard to swap
and forces test environments to mock SDK calls that have nothing to
do with the trading logic.

## Decision

* Alpaca lives only behind the `BrokerAdapter` port.
* Concrete adapters (`AlpacaPaperBrokerAdapter`,
  `AlpacaLiveBrokerAdapter`) translate between domain `OrderIntent` /
  `BrokerOrder` / `PositionSnapshot` / `AccountSnapshot` and Alpaca's
  REST / streaming objects.
* No Alpaca import is allowed outside `rl_swing.adapters.broker.*`
  and (for historical bars) `rl_swing.adapters.data.alpaca_data_provider`.
* Paper and live use **separate** keys, separate config files, and
  two independent feature flags before live trades may be sent.

## Consequences

* Tests can substitute `SimulatedBrokerAdapter` /
  `NoOpShadowBrokerAdapter` by changing one config line.
* Switching to a different broker later means writing a new adapter,
  not changing the decision pipeline.
* The reconciliation service compares Alpaca-shaped data through the
  same domain types as simulated data, so reconciliation tests are
  identical in research, shadow, paper, and live modes.
