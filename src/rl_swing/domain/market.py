"""Market-data domain types.

These objects are how data crosses the ``MarketDataProvider`` boundary.
They are deliberately small and provider-agnostic — no yfinance/WRDS/
Alpaca types may leak past this layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class MarketBar:
    """A single OHLCV bar for one symbol/timeframe.

    ``adjusted_close`` is preferred for signal/return calculations.
    ``source`` records which provider produced the bar so downstream
    consumers can apply provider-specific quality assumptions.
    ``quality_flags`` is a sorted tuple of zero or more short flag
    strings such as ``"price_unreliable"`` or ``"volume_zero"``.
    """

    symbol: str
    timestamp: datetime  # bar close (or session date at 00:00 UTC for daily)
    timeframe: str       # "1d", "1h", ...
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: float | None = None
    source: str = "unknown"
    quality_flags: tuple[str, ...] = ()

    def with_quality_flag(self, flag: str) -> MarketBar:
        if flag in self.quality_flags:
            return self
        new_flags = tuple(sorted({*self.quality_flags, flag}))
        return MarketBar(
            symbol=self.symbol,
            timestamp=self.timestamp,
            timeframe=self.timeframe,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            adjusted_close=self.adjusted_close,
            source=self.source,
            quality_flags=new_flags,
        )


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Identifier for a frozen view of market data used by a run.

    The snapshot id should be deterministic given (provider, universe,
    start, end, timeframe). Including it in feature/model artifacts is
    what makes a research run reproducible.
    """

    snapshot_id: str
    provider_id: str
    timeframe: str
    universe_version: str
    start: datetime
    end: datetime
    metadata: dict = field(default_factory=dict)
