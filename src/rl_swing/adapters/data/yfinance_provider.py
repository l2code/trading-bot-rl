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

    def _find_covering_cache_file(
        self, symbol: str, start: date, end: date, cache_dir: Path,
    ) -> Path | None:
        """FIX-#83: range-coverage cache match.

        The legacy lookup hits only on exact-filename match, which
        causes redundant yfinance fetches when a slightly-shifted
        warmup window (off by one day) misses an otherwise-covering
        cached file. Scan the cache for any
        ``{symbol}__{cached_start}__{cached_end}__adj=N.parquet``
        whose range fully contains the requested ``(start, end)``,
        and return the path of the cheapest such file (smallest
        coverage = least filtering on read). Returns None if none
        of the cached files cover the requested range — caller then
        falls through to fresh fetch.
        """
        adj = int(self.auto_adjust)
        prefix = f"{symbol}__"
        suffix = f"__adj={adj}.parquet"
        candidates: list[tuple[int, Path]] = []
        try:
            for p in cache_dir.glob(f"{prefix}*{suffix}"):
                stem = p.name[len(prefix):-len(suffix)]
                parts = stem.split("__")
                if len(parts) != 2:
                    continue
                try:
                    cs = date.fromisoformat(parts[0])
                    ce = date.fromisoformat(parts[1])
                except ValueError:
                    continue
                if cs <= start and ce >= end:
                    span_days = (ce - cs).days
                    candidates.append((span_days, p))
        except OSError as e:  # pragma: no cover
            _log.warning("yfinance cache scan failed for %s: %s", symbol, e)
            return None
        if not candidates:
            return None
        candidates.sort(key=lambda kv: kv[0])
        return candidates[0][1]

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
        # Fast path: exact-filename match.
        if self.use_cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
            except Exception as e:  # pragma: no cover
                _log.warning("yfinance cache read failed for %s: %s", symbol, e)
                df = None
        # FIX-#83: if exact match missed, try a covering cache file.
        # Filter the loaded DataFrame to the requested range so the
        # caller sees identical bars to what an exact-match cache
        # would have produced.
        if (df is None or df.empty) and self.use_cache:
            covering = self._find_covering_cache_file(symbol, start, end, cache_dir)
            if covering is not None:
                try:
                    full_df = pd.read_parquet(covering)
                    # Index is datetime-like; filter inclusive to (start, end).
                    df = full_df[(full_df.index.date >= start) & (full_df.index.date <= end)]
                    _log.info(
                        "yfinance cache hit via covering file for %s: %s "
                        "(requested %s..%s)",
                        symbol, covering.name, start, end,
                    )
                except Exception as e:  # pragma: no cover
                    _log.warning(
                        "yfinance covering-cache read failed for %s (%s): %s",
                        symbol, covering, e,
                    )
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
