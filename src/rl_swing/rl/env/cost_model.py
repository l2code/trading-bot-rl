"""Equity execution / cost model.

Implements the spec's `equity_execution_model` configuration knobs. A
``EquityExecutionModel.cost_bps(...)`` call returns total **per-side**
basis points to debit from a simulated fill.

A round-trip trade involves TWO fills (entry + exit), so callers
that simulate a complete round-trip (e.g.,
``ExecutionSimulator.simulate``) must multiply this number by 2 to
get the round-trip cost drag. The convention is per-side here so
the cost-stress 2x multiplier is intuitive against per-side
numbers, and so callers that genuinely want only one side
(e.g., partial fills, hedging legs) get the right answer without
dividing.

Component meanings (all per-side):
    base_spread_bps              — half the bid-ask cost per fill
    base_slippage_bps            — execution slippage per fill
    market_impact (computed)     — notional-dependent impact per fill
    adverse_selection_bps        — informational disadvantage per fill

This module is intentionally adapter-free — the env passes pure
numbers in. Cost-stress tests work by passing a multiplier
(`cost_stress_multiplier`).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EquityExecutionModel:
    base_spread_bps: float = 3.0
    base_slippage_bps: float = 5.0
    high_volatility_slippage_multiplier: float = 2.0
    event_window_slippage_multiplier: float = 2.0
    market_impact_coef: float = 0.10        # bps per (notional / adv) %
    adverse_selection_bps: float = 2.0
    cost_stress_multiplier: float = 1.0     # set >1 for doubled-cost tests

    def cost_bps(
        self,
        atr_pct: float = 0.0,
        volatility_percentile: float = 0.5,
        in_event_window: bool = False,
        notional: float = 0.0,
        avg_dollar_volume: float = 0.0,
    ) -> float:
        spread = self.base_spread_bps
        slippage = self.base_slippage_bps

        if volatility_percentile >= 0.8:
            slippage *= self.high_volatility_slippage_multiplier
        elif atr_pct >= 0.04:
            slippage *= 1.5

        if in_event_window:
            slippage *= self.event_window_slippage_multiplier

        if avg_dollar_volume > 0 and notional > 0:
            participation = notional / avg_dollar_volume
            impact = self.market_impact_coef * participation * 10_000.0
        else:
            impact = 0.0

        adverse = self.adverse_selection_bps

        total = (spread + slippage + impact + adverse) * self.cost_stress_multiplier
        return float(total)
