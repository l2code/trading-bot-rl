#!/usr/bin/env python3
"""Per-strategy training-EV analysis (RESEARCH-15).

Pure data analysis — no RL. For the v002 training window
(2014-2020 yfinance starter_equities), group candidates by their
originating strategy and compute per-strategy descriptive stats:

    count, mean per-trade return, win rate, mean risk-adjusted return,
    sharpe-ish (mean / std), median holding days, mean cost bps.

Output answers a single question: was the trained PPO selector's
collapse to "Momentum specialist" rational on the training
distribution (H2) or did it converge there for non-EV reasons (H1)?

Usage:
    python3 scripts/per_strategy_training_ev.py
        [--output research/diary/...md]

If ``--output`` is given, a Markdown table is appended to that file.
Otherwise the table is printed to stdout.
"""
from __future__ import annotations

import argparse
import math

# Adjust path so this works whether we run via ``python3 scripts/...``
# from the repo root.
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from rl_swing.adapters.data.yfinance_provider import YFinanceProvider
from rl_swing.domain import PortfolioState
from rl_swing.features.pipelines import CoreDailyPipeline
from rl_swing.rl.env.cost_model import EquityExecutionModel
from rl_swing.rl.env.execution_simulator import ExecutionSimulator
from rl_swing.strategies.breakout import BreakoutStrategy
from rl_swing.strategies.mean_reversion import RsiMeanReversionStrategy
from rl_swing.strategies.momentum import MomentumStrategy


# ---------------------------------------------------------------------
def _load_universe(name: str) -> list[str]:
    path = Path(__file__).resolve().parents[1] / "configs" / "universes" / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        return list((yaml.safe_load(f) or {}).get("universe", {}).get("symbols", []))


def _build_strategies():
    """Same loose config used by trainer.py and selector_v002.py — must
    match exactly so this analysis describes the candidate distribution
    the agent actually saw.
    """
    return [
        ("momentum", MomentumStrategy(
            min_relative_strength=-0.05,
            min_r20=-0.02,
            require_sma200_above=False,
        )),
        ("rsi_mean_reversion", RsiMeanReversionStrategy(rsi_threshold=35.0)),
        ("breakout", BreakoutStrategy(
            min_relative_volume=0.7,
            max_distance_below_high=-0.02,
        )),
    ]


# ---------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", default="starter_equities")
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end",   default="2020-12-31")
    ap.add_argument("--output", default=None,
                    help="Markdown file to append the table to (default: stdout).")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    universe = _load_universe(args.universe)
    print(f"[per_strategy_ev] universe={args.universe} ({len(universe)} symbols)")
    print(f"[per_strategy_ev] window={start}..{end}")

    # 1. Pull bars and build features.
    provider = YFinanceProvider()
    bars = list(provider.get_bars(universe, start, end, "1d", True))
    print(f"[per_strategy_ev] loaded {len(bars):,} bars")
    pipeline = CoreDailyPipeline()
    frames = list(pipeline.build(bars))
    print(f"[per_strategy_ev] built {len(frames):,} feature frames")

    # 2. Generate candidates per strategy (no dedupe).
    portfolio = PortfolioState(
        as_of=datetime(end.year, end.month, end.day),
        cash=100_000.0, equity=100_000.0,
    )
    strategies = _build_strategies()
    candidates_by_strategy: dict[str, list] = {}
    for name, strat in strategies:
        cands = list(strat.generate(frames, portfolio))
        candidates_by_strategy[name] = cands
        print(f"[per_strategy_ev] {name}: {len(cands):,} candidates")

    # 3. Simulate each candidate's trade outcome with the standard
    #    cost model. We use the same ExecutionSimulator the env uses
    #    so this analysis matches what the agent's reward signal saw.
    cost_model = EquityExecutionModel()
    sim = ExecutionSimulator()
    by_symbol_bars: dict[str, list] = defaultdict(list)
    for b in bars:
        by_symbol_bars[b.symbol].append(b)
    for sym in by_symbol_bars:
        by_symbol_bars[sym].sort(key=lambda b: b.timestamp)
    frames_by_key = {(f.symbol, f.as_of): f for f in frames}

    stats: dict[str, dict] = {}
    for name, cands in candidates_by_strategy.items():
        returns: list[float] = []
        raw_returns: list[float] = []
        cost_bps: list[float] = []
        holding_days: list[int] = []
        wins = 0
        n_no_data = 0
        for c in cands:
            frame = frames_by_key.get((c.symbol, c.as_of))
            if frame is None:
                n_no_data += 1
                continue
            atr_pct = float(frame.values.get("atr_pct_14", 0.02))
            rv20 = float(frame.values.get("realized_vol_20", 0.20))
            vol_pct = min(1.0, max(0.0, rv20 / 0.6))
            adv = float(frame.values.get("dollar_volume", 0.0))
            notional = 100_000.0 * c.base_size_pct
            bps = cost_model.cost_bps(
                atr_pct=atr_pct, volatility_percentile=vol_pct,
                in_event_window=False, notional=notional,
                avg_dollar_volume=adv,
            )
            sym_bars = by_symbol_bars.get(c.symbol) or []
            entry_idx = -1
            for i, b in enumerate(sym_bars):
                if b.timestamp == c.as_of:
                    entry_idx = i
                    break
            if entry_idx < 0:
                n_no_data += 1
                continue
            outcome = sim.simulate(
                bars=sym_bars, entry_index=entry_idx,
                size_pct=c.base_size_pct,
                max_holding_days=c.max_holding_days,
                cost_bps=bps, atr_pct=atr_pct,
                starting_equity=100_000.0,
            )
            if outcome is None:
                n_no_data += 1
                continue
            returns.append(outcome.return_pct)
            raw_returns.append(outcome.raw_return_pct)
            cost_bps.append(outcome.cost_bps)
            holding_days.append(outcome.holding_days)
            if outcome.return_pct > 0:
                wins += 1

        n = len(returns)
        if n == 0:
            stats[name] = {"n": 0}
            continue
        mean_ret = sum(returns) / n
        var = sum((r - mean_ret) ** 2 for r in returns) / max(1, n - 1)
        std = math.sqrt(var) if var > 0 else 0.0
        # "Sharpe-ish": per-trade mean / std (annualization is a
        # holding-day correction we omit since the comparison is
        # within-strategy and equally affected).
        sharpe_ish = (mean_ret / std) if std > 0 else 0.0
        # Risk-adjusted return per the env's reward conventions:
        # return / target_risk_pct (=0.02), clipped at +/-5.
        risk_adj = [max(-5.0, min(5.0, r / 0.02)) for r in returns]
        mean_risk_adj = sum(risk_adj) / n
        stats[name] = {
            "n": n,
            "mean_return": mean_ret,
            "median_return": sorted(returns)[n // 2],
            "std_return": std,
            "win_rate": wins / n,
            "mean_risk_adj": mean_risk_adj,
            "sharpe_ish": sharpe_ish,
            "mean_cost_bps": sum(cost_bps) / n,
            "median_holding_days": sorted(holding_days)[n // 2],
            "n_no_data": n_no_data,
        }

    # 4. Print / write the table.
    lines = [
        "",
        "## Per-strategy training-window EV (yfinance, 2014-01-01..2020-12-31, starter_equities, loose config)",
        "",
        "| Strategy | n | mean ret | median ret | win rate | mean risk-adj | sharpe-ish | mean cost (bps) | median hold (d) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("momentum", "rsi_mean_reversion", "breakout"):
        s = stats.get(name, {"n": 0})
        if s["n"] == 0:
            lines.append(f"| **{name}** | 0 | - | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| **{name}** | {s['n']:,} | {s['mean_return']:+.4f} | {s['median_return']:+.4f} | "
            f"{s['win_rate']:.3f} | {s['mean_risk_adj']:+.3f} | {s['sharpe_ish']:+.3f} | "
            f"{s['mean_cost_bps']:.1f} | {s['median_holding_days']} |"
        )
    lines.append("")

    out_text = "\n".join(lines)
    if args.output:
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(out_text)
        print(f"[per_strategy_ev] appended to {args.output}")
    print(out_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
