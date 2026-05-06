"""YFinanceProvider — yfinance-backed ``MarketDataProvider``.

Used for prototyping and sanity checks (per spec §6.1). Caches results
to parquet under ``cache_dir`` so we don't re-hit yfinance on every
backtest.

yfinance is the only third-party import allowed in this file. No
yfinance objects may leak past ``get_bars()``.
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
class YFinanceProvider:
    provider_id: str = "yfinance_daily"
    auto_adjust: bool = True
    cache_dir: str = "data/cache/yfinance"
    use_cache: bool = True

    def get_bars(
        self,
        symbols: list[str],
        start: date,
        end: date,
        timeframe: str = "1d",
        adjusted: bool = True,
    ) -> Iterable[MarketBar]:
        if timeframe != "1d":
            raise NotImplementedError("YFinanceProvider only supports 1d in this build.")
        cache_path = Path(self.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        for symbol in symbols:
            yield from self._bars_for_symbol(symbol, start, end, cache_path, adjusted)

    def get_snapshot_id(
        self,
        symbols: list[str],
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> str:
        key = f"{self.provider_id}|{','.join(sorted(symbols))}|{start}|{end}|{timeframe}|adj={self.auto_adjust}"
        return hashlib.sha1(key.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    def _cache_file(self, symbol: str, start: date, end: date, cache_dir: Path) -> Path:
        name = f"{symbol}__{start.isoformat()}__{end.isoformat()}__adj={int(self.auto_adjust)}.parquet"
        return cache_dir / name

    def _bars_for_symbol(
        self,
        symbol: str,
        start: date,
        end: date,
        cache_dir: Path,
        adjusted: bool,
    ) -> Iterable[MarketBar]:
        import pandas as pd  # heavy; lazy
        cache_file = self._cache_file(symbol, start, end, cache_dir)
        df: pd.DataFrame | None = None
        if self.use_cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
            except Exception as e:  # pragma: no cover
                _log.warning("yfinance cache read failed for %s: %s", symbol, e)
                df = None

        if df is None or df.empty:
            df = self._download(symbol, start, end)
            if df is not None and not df.empty and self.use_cache:
                try:
                    df.to_parquet(cache_file, index=True)
                except Exception as e:  # pragma: no cover
                    _log.warning("yfinance cache write failed for %s: %s", symbol, e)

        if df is None or df.empty:
            return

        # yfinance returns columns Open/High/Low/Close/Adj Close/Volume.
        # auto_adjust=True collapses Close into the adjusted price.
        for ts, row in df.iterrows():
            try:
                t = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                if isinstance(t, datetime):
                    t = datetime(t.year, t.month, t.day)
                else:
                    t = datetime(t.year, t.month, t.day)  # type: ignore[arg-type]
            except Exception:
                continue
            close = float(row.get("Close", 0.0))
            adj_close = float(row.get("Adj Close", close)) if "Adj Close" in row else close
            yield MarketBar(
                symbol=symbol,
                timestamp=t,
                timeframe="1d",
                open=float(row.get("Open", 0.0)),
                high=float(row.get("High", 0.0)),
                low=float(row.get("Low", 0.0)),
                close=close,
                volume=float(row.get("Volume", 0.0)),
                adjusted_close=adj_close,
                source=self.provider_id,
                quality_flags=(),
            )

    def _download(self, symbol: str, start: date, end: date):
        try:
            import yfinance as yf
        except ImportError as e:  # pragma: no cover
            _log.error("yfinance not installed: %s", e)
            return None
        try:
            df = yf.download(
                symbol,
                start=start.isoformat(),
                end=(end).isoformat(),
                auto_adjust=self.auto_adjust,
                progress=False,
                threads=False,
            )
            # yfinance >= 0.2.40 sometimes returns a MultiIndex column
            # frame even for a single symbol; flatten if so.
            if (
                df is not None and not df.empty
                and hasattr(df.columns, "nlevels") and df.columns.nlevels > 1
            ):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception as e:  # pragma: no cover
            _log.error("yfinance download failed for %s: %s", symbol, e)
            return None
