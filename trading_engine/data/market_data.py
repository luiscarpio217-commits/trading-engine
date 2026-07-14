"""OHLCV market data and earnings dates via yfinance, with TTL caching.

All network calls are wrapped: failures log and return empty/None so a
transient Yahoo outage degrades a scan instead of crashing the engine.
"""

from __future__ import annotations

import logging
import threading
import time as _time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns, keep OHLCV, drop rows without a close."""
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):  # yfinance multi-ticker layout
        out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        log.warning("OHLCV frame missing columns %s", missing)
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    out = out[OHLCV_COLUMNS].dropna(subset=["close"])
    return out


class _TTLCache:
    def __init__(self) -> None:
        self._data: dict = {}
        self._lock = threading.Lock()

    def get(self, key, ttl: float):
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                return None
            ts, value = hit
            if _time.monotonic() - ts > ttl:
                return None
            return value

    def put(self, key, value) -> None:
        with self._lock:
            self._data[key] = (_time.monotonic(), value)


class YFinanceMarketData:
    """Historical + near-real-time OHLCV from Yahoo Finance."""

    def __init__(self, intraday_cache_seconds: int = 30, daily_cache_seconds: int = 3600,
                 quote_cache_seconds: int = 10) -> None:
        self._cache = _TTLCache()
        self._intraday_ttl = intraday_cache_seconds
        self._daily_ttl = daily_cache_seconds
        self._quote_ttl = quote_cache_seconds

    # -- internals ---------------------------------------------------------

    def _history(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        import yfinance as yf

        try:
            raw = yf.Ticker(symbol).history(
                period=period, interval=interval, auto_adjust=False,
            )
            return normalize_ohlcv(raw)
        except Exception as exc:  # network / parsing issues must not kill a scan
            log.warning("yfinance history failed for %s (%s %s): %s",
                        symbol, period, interval, exc)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

    # -- public API --------------------------------------------------------

    def get_intraday(self, symbol: str, interval: str = "5m",
                     lookback_days: int = 5) -> pd.DataFrame:
        """Intraday bars including the developing bar (near-real-time)."""
        key = ("intraday", symbol, interval, lookback_days)
        cached = self._cache.get(key, self._intraday_ttl)
        if cached is not None:
            return cached
        df = self._history(symbol, period=f"{lookback_days}d", interval=interval)
        self._cache.put(key, df)
        return df

    def get_daily(self, symbol: str, lookback_days: int = 120) -> pd.DataFrame:
        key = ("daily", symbol, lookback_days)
        cached = self._cache.get(key, self._daily_ttl)
        if cached is not None:
            return cached
        df = self._history(symbol, period=f"{lookback_days}d", interval="1d")
        self._cache.put(key, df)
        return df

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Most recent traded price (fast_info, falling back to last 1m close)."""
        key = ("quote", symbol)
        cached = self._cache.get(key, self._quote_ttl)
        if cached is not None:
            return cached
        import yfinance as yf

        price: Optional[float] = None
        try:
            fi = yf.Ticker(symbol).fast_info
            raw = getattr(fi, "last_price", None) or fi.get("lastPrice")  # type: ignore[union-attr]
            if raw:
                price = float(raw)
        except Exception:
            price = None
        if price is None:
            bars = self._history(symbol, period="1d", interval="1m")
            if not bars.empty:
                price = float(bars["close"].iloc[-1])
        if price is not None:
            self._cache.put(key, price)
        return price

    def get_earnings_dates(self, symbol: str) -> list[date]:
        """Upcoming + recent earnings dates. Empty list when unknown."""
        key = ("earnings", symbol, date.today())
        cached = self._cache.get(key, 86_400)
        if cached is not None:
            return cached
        import yfinance as yf

        dates: list[date] = []
        try:
            df = yf.Ticker(symbol).get_earnings_dates(limit=8)
            if df is not None and not df.empty:
                dates = sorted({ts.date() for ts in df.index if ts is not None})
        except Exception as exc:
            log.debug("earnings dates unavailable for %s: %s", symbol, exc)
        self._cache.put(key, dates)
        return dates
