"""Technical indicators implemented natively on pandas/numpy.

All functions accept lowercase-column OHLCV DataFrames (open, high, low,
close, volume) and are side-effect free. Wilder smoothing is used for RSI
and ATR, matching ta-lib conventions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # No losses at all -> RSI 100; flat series -> neutral 50.
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return out


def vwap(df: pd.DataFrame, tz: str = "America/New_York") -> pd.Series:
    """Session-anchored VWAP: cumulative typical-price * volume, reset daily.

    Sessions are grouped by calendar date in `tz` when the index is
    timezone-aware (yfinance intraday indexes are), else by naive date.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        session = idx.tz_convert(tz).date
    elif isinstance(idx, pd.DatetimeIndex):
        session = idx.date
    else:  # non-datetime index: single session
        session = np.zeros(len(df), dtype=int)
    pv = (typical * df["volume"]).groupby(session).cumsum()
    vv = df["volume"].groupby(session).cumsum()
    return pv / vv.replace(0, np.nan)


def bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands. Returns columns: bb_mid, bb_upper, bb_lower, bb_pctb, bb_width."""
    mid = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid.replace(0.0, np.nan)
    pctb = (series - lower) / (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
         "bb_pctb": pctb, "bb_width": width}
    )


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range with Wilder smoothing."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line, histogram. Columns: macd, macd_signal, macd_hist."""
    fast_ema = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = fast_ema - slow_ema
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def relative_volume(volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Current bar volume vs the mean of the *prior* `lookback` bars (RVOL)."""
    baseline = volume.shift(1).rolling(lookback).mean()
    return volume / baseline.replace(0, np.nan)


def rolling_resistance(high: pd.Series, lookback: int = 20) -> pd.Series:
    """Highest high of the prior `lookback` bars (excludes the current bar)."""
    return high.shift(1).rolling(lookback).max()


def rolling_support(low: pd.Series, lookback: int = 20) -> pd.Series:
    """Lowest low of the prior `lookback` bars (excludes the current bar)."""
    return low.shift(1).rolling(lookback).min()


@dataclass
class VolumeProfile:
    """Volume-by-price histogram with point of control and 70% value area."""

    profile: pd.DataFrame       # columns: price_low, price_high, price_mid, volume
    poc_price: float            # point of control (highest-volume bin midpoint)
    value_area_low: float
    value_area_high: float
    total_volume: float


def volume_profile(df: pd.DataFrame, bins: int = 24, value_area_pct: float = 0.70) -> VolumeProfile:
    """Build a volume profile by binning each bar's volume at its typical price."""
    typical = ((df["high"] + df["low"] + df["close"]) / 3.0).to_numpy()
    volumes = df["volume"].to_numpy(dtype=float)
    lo = float(df["low"].min())
    hi = float(df["high"].max())
    if hi <= lo:
        hi = lo + 1e-9
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(typical, edges) - 1, 0, bins - 1)
    vol_by_bin = np.zeros(bins)
    np.add.at(vol_by_bin, idx, volumes)

    mids = (edges[:-1] + edges[1:]) / 2.0
    total = float(vol_by_bin.sum())
    poc = int(np.argmax(vol_by_bin))

    # Expand around the POC, greedily adding the higher-volume neighbor,
    # until the value area holds `value_area_pct` of total volume.
    lo_i = hi_i = poc
    covered = vol_by_bin[poc]
    while covered < value_area_pct * total and (lo_i > 0 or hi_i < bins - 1):
        below = vol_by_bin[lo_i - 1] if lo_i > 0 else -1.0
        above = vol_by_bin[hi_i + 1] if hi_i < bins - 1 else -1.0
        if above >= below:
            hi_i += 1
            covered += max(above, 0.0)
        else:
            lo_i -= 1
            covered += max(below, 0.0)

    profile = pd.DataFrame(
        {"price_low": edges[:-1], "price_high": edges[1:],
         "price_mid": mids, "volume": vol_by_bin}
    )
    return VolumeProfile(
        profile=profile,
        poc_price=float(mids[poc]),
        value_area_low=float(edges[lo_i]),
        value_area_high=float(edges[hi_i + 1]),
        total_volume=total,
    )


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of an OHLCV frame enriched with the standard indicator set.

    Adds: ema9, ema21, ema50, rsi14, vwap, bb_* (5 cols), atr14,
    macd, macd_signal, macd_hist, rvol.
    """
    out = df.copy()
    close = out["close"]
    out["ema9"] = ema(close, 9)
    out["ema21"] = ema(close, 21)
    out["ema50"] = ema(close, 50)
    out["rsi14"] = rsi(close, 14)
    out["vwap"] = vwap(out)
    out = out.join(bollinger(close))
    out["atr14"] = atr(out)
    out = out.join(macd(close))
    out["rvol"] = relative_volume(out["volume"])
    return out
