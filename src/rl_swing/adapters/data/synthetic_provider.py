"""SyntheticProvider — deterministic fake markets for sanity testing.

Three regimes are supported:
    * ``momentum``       — symbols with persistent multi-week trends.
                           A naive momentum strategy should make money.
    * ``mean_reversion`` — symbols with mean-reverting noise around a
                           slow drift. RSI-style signals should work.
    * ``random_walk``    — pure GBM noise. Nothing should produce a
                           durable edge; if RL claims one, the simulator
                           is leaking.

This is the substrate for the spec's mandatory synthetic sanity tests
(spec §12.5.9).
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np

from rl_swing.domain import MarketBar


def _trading_dates(start: date, end: date) -> list[date]:
    out = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri
            out.append(d)
        d += timedelta(days=1)
    return out


def _seeded_rng(symbol: str, regime: str, seed: int) -> np.random.Generator:
    h = hashlib.sha256(f"{symbol}|{regime}|{seed}".encode()).digest()
    derived_seed = int.from_bytes(h[:8], "big") % (2**31 - 1)
    return np.random.default_rng(derived_seed)


@dataclass
class SyntheticProvider:
    provider_id: str = "synthetic"
    regime: str = "momentum"      # "momentum" | "mean_reversion" | "random_walk"
    seed: int = 11
    base_price: float = 100.0
    drift_per_day: float = 0.0003   # ~7.5%/year
    annual_vol: float = 0.20

    def __post_init__(self) -> None:
        self.provider_id = f"synthetic_{self.regime}"

    # ------------------------------------------------------------------
    def get_bars(
        self,
        symbols: list[str],
        start: date,
        end: date,
        timeframe: str = "1d",
        adjusted: bool = True,
    ) -> Iterable[MarketBar]:
        if timeframe != "1d":
            return
        dates = _trading_dates(start, end)
        if not dates:
            return
        sigma_d = self.annual_vol / np.sqrt(252.0)

        for symbol in symbols:
            rng = _seeded_rng(symbol, self.regime, self.seed)
            n = len(dates)
            log_returns = self._log_returns(n, sigma_d, rng)
            prices = self.base_price * np.exp(np.cumsum(log_returns))
            # Build OHLC around the close path. Use intraday range
            # ~ daily vol * 1.5 to give the env something to work with.
            for i, d in enumerate(dates):
                c = float(prices[i])
                rng_intraday = rng.normal(0.0, sigma_d * 0.6, size=2)
                hi = c * (1 + abs(rng_intraday[0]))
                lo = c * (1 - abs(rng_intraday[1]))
                op = float(prices[i - 1]) if i > 0 else c
                vol = float(max(1e6, rng.normal(5e6, 1e6)))
                ts = datetime(d.year, d.month, d.day)
                yield MarketBar(
                    symbol=symbol,
                    timestamp=ts,
                    timeframe="1d",
                    open=op, high=hi, low=lo, close=c,
                    volume=vol,
                    adjusted_close=c,
                    source=self.provider_id,
                    quality_flags=(),
                )

    def get_snapshot_id(
        self,
        symbols: list[str],
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> str:
        key = f"{self.provider_id}|{self.regime}|{self.seed}|{','.join(sorted(symbols))}|{start}|{end}|{timeframe}"
        return hashlib.sha1(key.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    def _log_returns(
        self, n: int, sigma_d: float, rng: np.random.Generator
    ) -> np.ndarray:
        if self.regime == "random_walk":
            return rng.normal(self.drift_per_day, sigma_d, size=n)
        if self.regime == "momentum":
            # AR(1) with positive autocorrelation — multi-week trends.
            phi = 0.65
            eps = rng.normal(0.0, sigma_d, size=n)
            r = np.zeros(n)
            r[0] = self.drift_per_day + eps[0]
            for i in range(1, n):
                r[i] = self.drift_per_day + phi * (r[i - 1] - self.drift_per_day) + eps[i]
            return r
        if self.regime == "mean_reversion":
            # OU around a drifting mean — pullbacks revert.
            mean = self.drift_per_day
            theta = 0.20
            r = np.zeros(n)
            r[0] = mean + rng.normal(0.0, sigma_d)
            for i in range(1, n):
                r[i] = r[i - 1] + theta * (mean - r[i - 1]) + rng.normal(0.0, sigma_d)
            return r
        raise ValueError(f"unknown regime: {self.regime!r}")
