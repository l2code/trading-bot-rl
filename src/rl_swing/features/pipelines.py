"""``CoreDailyPipeline`` — features_v001_core_daily.

Implements the spec's MVP feature tier: ~25 daily technical features
plus a few market-regime features when SPY/QQQ are present in the
input bars. Strategy signal features and portfolio-state features
join later via ``services/feature_build_service``.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from rl_swing.domain import FeatureFrame, MarketBar
from rl_swing.features import technical as tf

_log = logging.getLogger(__name__)

# Authoritative feature list. Order matters: ObservationBuilder uses this
# ordering when packing observation vectors.
TECHNICAL_FEATURE_NAMES: tuple[str, ...] = (
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",
    "close_vs_sma_10",
    "close_vs_sma_20",
    "close_vs_sma_50",
    "close_vs_sma_200",
    "rsi_2",
    "rsi_5",
    "rsi_14",
    "atr_pct_14",
    "realized_vol_20",
    "relative_volume_20",
    "zscore_close_20",
    "distance_from_20d_high",
    "distance_from_20d_low",
    "dollar_volume",
    "log_dollar_volume",
)

REGIME_FEATURE_NAMES: tuple[str, ...] = (
    "spy_return_20d",
    "spy_above_sma_50",
    "spy_above_sma_200",
)

ALL_FEATURE_NAMES: tuple[str, ...] = TECHNICAL_FEATURE_NAMES + REGIME_FEATURE_NAMES


@dataclass
class CoreDailyPipeline:
    feature_version: str = "features_v001_core_daily"
    feature_names: tuple[str, ...] = ALL_FEATURE_NAMES

    def build(
        self,
        bars: Iterable[MarketBar],
        context: dict[str, Any] | None = None,
    ) -> Iterable[FeatureFrame]:
        # Group bars by symbol, sorted by timestamp.
        by_symbol: dict[str, list[MarketBar]] = defaultdict(list)
        for b in bars:
            by_symbol[b.symbol].append(b)
        for s in by_symbol:
            by_symbol[s].sort(key=lambda x: x.timestamp)

        # Pre-compute SPY market context if available.
        spy_bars = by_symbol.get("SPY") or []
        spy_dates: list[float] = []
        spy_ret_20d: np.ndarray | None = None
        spy_above_50: np.ndarray | None = None
        spy_above_200: np.ndarray | None = None
        if len(spy_bars) >= 200:
            spy_close = np.array(
                [b.adjusted_close or b.close for b in spy_bars], dtype=float
            )
            spy_dates = [b.timestamp.toordinal() for b in spy_bars]
            spy_ret_20d = tf.returns(spy_close, 20)
            sma50 = tf.sma(spy_close, 50)
            sma200 = tf.sma(spy_close, 200)
            spy_above_50 = (spy_close > sma50).astype(float)
            spy_above_200 = (spy_close > sma200).astype(float)

        snapshot_id = (context or {}).get("source_snapshot_id", "unknown")

        for symbol, sym_bars in by_symbol.items():
            if len(sym_bars) < 60:  # need at least 60 bars to compute returns_60d
                continue
            close = np.array(
                [b.adjusted_close or b.close for b in sym_bars], dtype=float
            )
            high = np.array([b.high for b in sym_bars], dtype=float)
            low = np.array([b.low for b in sym_bars], dtype=float)
            volume = np.array([b.volume for b in sym_bars], dtype=float)
            dollar_vol = close * volume

            r1 = tf.returns(close, 1)
            r5 = tf.returns(close, 5)
            r10 = tf.returns(close, 10)
            r20 = tf.returns(close, 20)
            r60 = tf.returns(close, 60)

            sma10 = tf.sma(close, 10)
            sma20 = tf.sma(close, 20)
            sma50 = tf.sma(close, 50)
            sma200 = tf.sma(close, 200) if len(close) >= 200 else tf.sma(close, len(close) // 2 or 1)

            close_vs_sma10 = np.where(sma10 != 0, close / sma10 - 1.0, 0.0)
            close_vs_sma20 = np.where(sma20 != 0, close / sma20 - 1.0, 0.0)
            close_vs_sma50 = np.where(sma50 != 0, close / sma50 - 1.0, 0.0)
            close_vs_sma200 = np.where(sma200 != 0, close / sma200 - 1.0, 0.0)

            rsi2 = tf.rsi(close, 2)
            rsi5 = tf.rsi(close, 5)
            rsi14 = tf.rsi(close, 14)

            atr14 = tf.atr(high, low, close, 14)
            atr_pct = np.where(close != 0, atr14 / close, 0.0)
            rv20 = tf.realized_vol(close, 20)
            rel_vol_20 = tf.relative_volume(volume, 20)
            zclose20 = tf.zscore(close, 20)
            dist_high20 = tf.distance_from_high(close, 20)
            dist_low20 = tf.distance_from_low(close, 20)

            # Avoid log(0).
            log_dollar_vol = np.log(np.maximum(dollar_vol, 1.0))

            # Build per-bar frames starting at index 200 (or the first
            # index where SMA200 is meaningful) so all features are
            # well-defined.
            start_idx = max(60, min(200, len(sym_bars) - 1))
            for i in range(start_idx, len(sym_bars)):
                vals: dict[str, float] = {
                    "return_1d": float(r1[i]),
                    "return_5d": float(r5[i]),
                    "return_10d": float(r10[i]),
                    "return_20d": float(r20[i]),
                    "return_60d": float(r60[i]),
                    "close_vs_sma_10": float(close_vs_sma10[i]),
                    "close_vs_sma_20": float(close_vs_sma20[i]),
                    "close_vs_sma_50": float(close_vs_sma50[i]),
                    "close_vs_sma_200": float(close_vs_sma200[i]),
                    "rsi_2": float(rsi2[i]),
                    "rsi_5": float(rsi5[i]),
                    "rsi_14": float(rsi14[i]),
                    "atr_pct_14": float(atr_pct[i]),
                    "realized_vol_20": float(rv20[i]),
                    "relative_volume_20": float(rel_vol_20[i]),
                    "zscore_close_20": float(zclose20[i]),
                    "distance_from_20d_high": float(dist_high20[i]),
                    "distance_from_20d_low": float(dist_low20[i]),
                    "dollar_volume": float(dollar_vol[i]),
                    "log_dollar_volume": float(log_dollar_vol[i]),
                }
                vals.update(self._regime_values(
                    sym_bars[i].timestamp.toordinal(),
                    spy_dates, spy_ret_20d, spy_above_50, spy_above_200,
                ))
                yield FeatureFrame(
                    as_of=sym_bars[i].timestamp,
                    symbol=symbol,
                    feature_version=self.feature_version,
                    values=vals,
                    feature_names=self.feature_names,
                    source_snapshot_id=str(snapshot_id),
                )

    @staticmethod
    def _regime_values(
        ordinal: int,
        spy_dates: list[float],
        spy_ret_20d: np.ndarray | None,
        spy_above_50: np.ndarray | None,
        spy_above_200: np.ndarray | None,
    ) -> dict[str, float]:
        if not spy_dates or spy_ret_20d is None:
            return {
                "spy_return_20d": 0.0,
                "spy_above_sma_50": 0.0,
                "spy_above_sma_200": 0.0,
            }
        # Find the most recent SPY index <= ordinal.
        # spy_dates is sorted; bisect.
        import bisect
        idx = bisect.bisect_right(spy_dates, ordinal) - 1
        if idx < 0:
            return {
                "spy_return_20d": 0.0,
                "spy_above_sma_50": 0.0,
                "spy_above_sma_200": 0.0,
            }
        return {
            "spy_return_20d": float(spy_ret_20d[idx]),
            "spy_above_sma_50": float(spy_above_50[idx]) if spy_above_50 is not None else 0.0,
            "spy_above_sma_200": float(spy_above_200[idx]) if spy_above_200 is not None else 0.0,
        }
