"""MarketDataProvider port.

Adapters: yfinance, WRDS-parquet, Alpaca historical, CSV/parquet,
synthetic. No yfinance/WRDS/Alpaca-specific objects may cross this
interface.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Protocol, runtime_checkable

from rl_swing.domain import MarketBar


@runtime_checkable
class MarketDataProvider(Protocol):
    provider_id: str

    def get_bars(
        self,
        symbols: list[str],
        start: date,
        end: date,
        timeframe: str = "1d",
        adjusted: bool = True,
    ) -> Iterable[MarketBar]: ...

    def get_snapshot_id(
        self,
        symbols: list[str],
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> str: ...
