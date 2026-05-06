"""ParquetProvider — reads pre-saved parquet files keyed by symbol.

Used for unit tests with synthetic fixtures and as a way to ship
pre-baked bar data without re-hitting yfinance/WRDS.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from rl_swing.domain import MarketBar

_log = logging.getLogger(__name__)


@dataclass
class ParquetProvider:
    provider_id: str = "parquet_cache"
    cache_dir: str = "data/cache/bars"

    def get_bars(
        self,
        symbols: list[str],
        start: date,
        end: date,
        timeframe: str = "1d",
        adjusted: bool = True,
    ) -> Iterable[MarketBar]:
        if timeframe != "1d":
            raise NotImplementedError("ParquetProvider only supports 1d.")
        try:
            import pandas as pd
        except ImportError:
            return
        for symbol in symbols:
            path = Path(self.cache_dir) / f"{symbol}.parquet"
            if not path.exists():
                continue
            try:
                df = pd.read_parquet(path)
            except Exception as e:  # pragma: no cover
                _log.warning("parquet read failed for %s: %s", symbol, e)
                continue
            for ts, row in df.iterrows():
                d = ts.date() if isinstance(ts, datetime) else ts
                if d < start or d > end:
                    continue
                yield MarketBar(
                    symbol=symbol,
                    timestamp=ts if isinstance(ts, datetime) else datetime(ts.year, ts.month, ts.day),
                    timeframe="1d",
                    open=float(row.get("open", 0.0)),
                    high=float(row.get("high", 0.0)),
                    low=float(row.get("low", 0.0)),
                    close=float(row.get("close", 0.0)),
                    volume=float(row.get("volume", 0.0)),
                    adjusted_close=float(row.get("adjusted_close", row.get("close", 0.0))),
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
        key = f"{self.provider_id}|{','.join(sorted(symbols))}|{start}|{end}|{timeframe}|{self.cache_dir}"
        return hashlib.sha1(key.encode()).hexdigest()[:16]
