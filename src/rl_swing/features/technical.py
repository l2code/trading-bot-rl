"""Pure technical-feature helpers operating on numpy arrays.

All helpers return arrays of the same length as the input close series.
The first ``window`` values are filled with the *first valid* lookback
result (rather than NaN) so downstream feature consumers don't have
to handle NaNs. The leakage check still asserts there are bars at or
before each frame's ``as_of``.
"""
from __future__ import annotations

import numpy as np


def returns(close: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(close, dtype=float)
    if len(close) <= window:
        return out
    out[window:] = close[window:] / close[:-window] - 1.0
    return out


def sma(close: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        return close.copy()
    out = np.zeros_like(close, dtype=float)
    cumsum = np.cumsum(np.insert(close, 0, 0.0))
    sums = cumsum[window:] - cumsum[:-window]
    out[window - 1:] = sums / float(window)
    out[: window - 1] = out[window - 1] if len(close) >= window else close[: window - 1]
    return out


def rsi(close: np.ndarray, window: int) -> np.ndarray:
    """Wilder's RSI."""
    if len(close) <= window:
        return np.full_like(close, 50.0, dtype=float)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros_like(close, dtype=float)
    avg_loss = np.zeros_like(close, dtype=float)
    avg_gain[window] = float(gain[1:window + 1].mean())
    avg_loss[window] = float(loss[1:window + 1].mean())
    for i in range(window + 1, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (window - 1) + gain[i]) / window
        avg_loss[i] = (avg_loss[i - 1] * (window - 1) + loss[i]) / window
    # Standard RSI conventions:
    #   avg_loss == 0 and avg_gain > 0  -> RSI = 100
    #   avg_loss == 0 and avg_gain == 0 -> RSI = 50 (flat)
    out = np.full_like(close, 50.0, dtype=float)
    safe_loss = np.where(avg_loss == 0, 1.0, avg_loss)
    rs = avg_gain / safe_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    no_loss = avg_loss == 0
    out = np.where(no_loss & (avg_gain > 0), 100.0, out)
    out = np.where(no_loss & (avg_gain == 0), 50.0, out)
    out[:window + 1] = 50.0
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    """Average True Range over ``window`` days."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    out = np.zeros_like(close, dtype=float)
    if len(close) >= window:
        out[window - 1] = float(tr[:window].mean())
        for i in range(window, len(close)):
            out[i] = (out[i - 1] * (window - 1) + tr[i]) / window
        out[: window - 1] = out[window - 1]
    return out


def realized_vol(close: np.ndarray, window: int, annualize: bool = True) -> np.ndarray:
    if len(close) < 2:
        return np.zeros_like(close, dtype=float)
    log_ret = np.zeros_like(close, dtype=float)
    log_ret[1:] = np.log(close[1:] / close[:-1])
    out = np.zeros_like(close, dtype=float)
    for i in range(window, len(close)):
        out[i] = float(log_ret[i - window + 1:i + 1].std(ddof=0))
    if annualize:
        out *= np.sqrt(252.0)
    out[:window] = out[window] if len(close) > window else 0.0
    return out


def zscore(values: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    for i in range(window - 1, len(values)):
        seg = values[i - window + 1:i + 1]
        m = seg.mean()
        s = seg.std(ddof=0)
        out[i] = 0.0 if s == 0 else (values[i] - m) / s
    out[: window - 1] = 0.0
    return out


def relative_volume(volume: np.ndarray, window: int) -> np.ndarray:
    avg = sma(volume, window)
    out = np.zeros_like(volume, dtype=float)
    nonzero = avg != 0
    out[nonzero] = volume[nonzero] / avg[nonzero]
    return out


def distance_from_high(close: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(close, dtype=float)
    for i in range(window - 1, len(close)):
        peak = float(close[i - window + 1:i + 1].max())
        if peak > 0:
            out[i] = (close[i] - peak) / peak  # negative until breakout
    return out


def distance_from_low(close: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(close, dtype=float)
    for i in range(window - 1, len(close)):
        trough = float(close[i - window + 1:i + 1].min())
        if trough > 0:
            out[i] = (close[i] - trough) / trough  # positive after bounce
    return out
