"""WrdsParquetProvider — reads the parquet cache produced by the
``trading-bot2`` project's ``scripts/research/wrds_refresh.py``.

This adapter does NOT import the ``wrds`` Python package. Live WRDS
pulls happen out-of-band; this adapter is the read side. That keeps
the package importable in environments without the ``wrds`` extras
installed (Colab, CI).

Expected cache layout (the trading-bot2 default at
``cache/wrds/``):

    cache_dir/
      crsp_dsf.parquet      # daily security file: PERMNO/PRC/RET/...
      crsp_stocknames.parquet
      ...

We map CRSP PRC/RET to OHLCV. Only ``c`` (close) and ``v`` (volume)
are available in CRSP DSF; we synthesize ``o/h/l`` as the close so
downstream features that read those fields don't crash. For
intra-day-sensitive features, prefer yfinance for now and roll over
to WRDS for return/feature calculations.
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
class WrdsParquetProvider:
    provider_id: str = "wrds_parquet"
    cache_dir: str = "/home/rissac/projects/trading-bot2/cache/wrds"

    def get_bars(
        self,
        symbols: list[str],
        start: date,
        end: date,
        timeframe: str = "1d",
        adjusted: bool = True,
    ) -> Iterable[MarketBar]:
        if timeframe != "1d":
            raise NotImplementedError("WrdsParquetProvider only supports 1d.")
        df = self._load_dsf()
        if df is None:
            return
        # Normalize tickers to upper.
        symbols_set = {s.upper() for s in symbols}

        for symbol in sorted(symbols_set):
            sub = self._slice_for_symbol(df, symbol, start, end)
            if sub is None or sub.empty:
                continue
            for _, row in sub.iterrows():
                try:
                    raw_close = float(row["PRC"])
                    if raw_close == 0:
                        continue
                    # CRSP encodes bid/ask-average closes as negative PRC.
                    flags: tuple[str, ...] = ()
                    if raw_close < 0:
                        flags = ("price_unreliable",)
                        raw_close = abs(raw_close)
                    cfacpr = float(row.get("CFACPR", 1.0) or 1.0)
                    adj_close = raw_close / cfacpr if cfacpr else raw_close
                    c = adj_close if adjusted else raw_close
                    vol = float(row.get("VOL", 0.0) or 0.0)
                    d = row["date"]
                    if not isinstance(d, datetime):
                        d = datetime.combine(d, datetime.min.time())
                    yield MarketBar(
                        symbol=symbol,
                        timestamp=d,
                        timeframe="1d",
                        open=c, high=c, low=c, close=c,   # CRSP DSF only has close
                        volume=vol,
                        adjusted_close=adj_close,
                        source=self.provider_id,
                        quality_flags=flags,
                    )
                except (KeyError, TypeError, ValueError) as e:  # pragma: no cover
                    _log.debug("skipping bad WRDS row for %s: %s", symbol, e)

    def get_snapshot_id(
        self,
        symbols: list[str],
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> str:
        key = f"{self.provider_id}|{','.join(sorted(symbols))}|{start}|{end}|{timeframe}|{self.cache_dir}"
        return hashlib.sha1(key.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    _df_cache = None  # type: ignore[var-annotated]

    def _load_dsf(self):
        if WrdsParquetProvider._df_cache is not None:
            return WrdsParquetProvider._df_cache
        try:
            import pandas as pd  # lazy
        except ImportError:
            return None
        path = Path(self.cache_dir) / "crsp_dsf.parquet"
        if not path.exists():
            _log.warning(
                "WRDS DSF cache not found at %s. "
                "Run trading-bot2/scripts/research/wrds_refresh.py to populate it.",
                path,
            )
            return None
        df = pd.read_parquet(path)
        # Resolve ticker symbol via stocknames if available.
        names_path = Path(self.cache_dir) / "crsp_stocknames.parquet"
        if "TICKER" not in df.columns and names_path.exists():
            try:
                names = pd.read_parquet(names_path)
                if "PERMNO" in df.columns and "PERMNO" in names.columns:
                    # Take the latest TICKER per PERMNO. CRSP allows ticker
                    # reuse so this is approximate — see trading-bot2's
                    # resolve_permno() for the precise version.
                    names_sorted = names.sort_values(["PERMNO", "NAMEDT"])
                    last = names_sorted.drop_duplicates("PERMNO", keep="last")
                    df = df.merge(last[["PERMNO", "TICKER"]], on="PERMNO", how="left")
            except Exception as e:  # pragma: no cover
                _log.warning("WRDS stocknames merge failed: %s", e)
        if "TICKER" not in df.columns:
            return None
        # Standardize date column.
        if "date" not in df.columns and "DATE" in df.columns:
            df = df.rename(columns={"DATE": "date"})
        if "date" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        WrdsParquetProvider._df_cache = df
        return df

    def _slice_for_symbol(self, df, symbol: str, start: date, end: date):
        s = df["TICKER"].astype(str).str.upper()
        mask = (s == symbol) & (df["date"] >= str(start)) & (df["date"] <= str(end))
        sub = df.loc[mask].sort_values("date")
        return sub
