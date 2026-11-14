"""Market data from the Alpaca Market Data API, with Yahoo as a fallback.

Drop-in replacement for YFinanceMarketData (same four methods), selected with
`data.provider: alpaca`. Uses the same ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY
environment variables as the Alpaca broker; free paper-account keys work.

Feeds on the free plan, and the defaults chosen here:
  * daily bars    -> "sip" (every US exchange). The free plan only blocks the
                     latest 15 minutes of SIP data, so requests end 16 minutes
                     ago. Full volume matters: the dollar-volume filter and the
                     52-week high need the consolidated numbers.
  * intraday bars -> "iex" (real time, but one exchange, a few % of volume).
                     Prices track the market closely for liquid names; volume
                     is IEX-only, so RVOL is relative to IEX volume.
  * latest price  -> "iex" latest trade.
With the paid plan (all-exchange real time) set every feed to "sip" and
alpaca_sip_delayed: false.

Every call is rate limited (default 180/min, under the free plan's 200) and
cached with the same TTLs as the Yahoo provider. If Alpaca errors or returns
nothing, the call falls back to Yahoo so a scan degrades instead of stopping.
IEX can also be stale for thinly traded small caps (no IEX print for a
while), so during market hours an IEX last trade older than
max_quote_age_seconds, or intraday bars whose last bar is older than
max_intraday_staleness_minutes, fall back to Yahoo too.
Earnings dates always come from Yahoo (Alpaca has no earnings calendar).
"""

from __future__ import annotations

import logging
import threading
import time as _time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from .market_data import OHLCV_COLUMNS, _TTLCache

log = logging.getLogger(__name__)

DATA_URL = "https://data.alpaca.markets/v2/stocks"
TZ = "America/New_York"
_TIMEFRAMES = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min", "1h": "1Hour",
               "60m": "1Hour"}


class _RateLimiter:
    def __init__(self, per_minute: int) -> None:
        self._per_minute = max(1, per_minute)
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = _time.monotonic()
                while self._calls and now - self._calls[0] >= 60.0:
                    self._calls.popleft()
                if len(self._calls) < self._per_minute:
                    self._calls.append(now)
                    return
                wait = 60.0 - (now - self._calls[0])
            _time.sleep(max(wait, 0.05))


def bars_to_frame(bars: list[dict], intraday: bool) -> pd.DataFrame:
    """Alpaca bars -> lowercase OHLCV frame.

    Intraday: tz-aware New York index, regular session (09:30-16:00) only, to
    match yfinance's default and keep VWAP/breakouts off pre/after-hours prints.
    Daily: naive dates (Alpaca stamps daily bars at New York midnight).
    """
    if not bars:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    df = pd.DataFrame(bars)
    idx = pd.DatetimeIndex(pd.to_datetime(df["t"], utc=True)).tz_convert(TZ)
    out = pd.DataFrame({
        "open": df["o"].astype(float).values,
        "high": df["h"].astype(float).values,
        "low": df["l"].astype(float).values,
        "close": df["c"].astype(float).values,
        "volume": df["v"].astype(float).values,
    }, index=idx)
    if intraday:
        minutes = out.index.hour * 60 + out.index.minute
        out = out[(minutes >= 9 * 60 + 30) & (minutes < 16 * 60)]
    else:
        out.index = out.index.normalize().tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(subset=["close"])


class AlpacaMarketData:
    def __init__(self, key_id: str, secret_key: str,
                 intraday_feed: str = "iex", daily_feed: str = "sip", quote_feed: str = "iex",
                 sip_delayed: bool = True, fallback=None,
                 intraday_cache_seconds: int = 30, daily_cache_seconds: int = 3600,
                 quote_cache_seconds: int = 10, max_calls_per_minute: int = 180,
                 max_quote_age_seconds: int = 300, max_intraday_staleness_minutes: int = 20,
                 session=None, clock=None) -> None:
        import requests

        self._session = session or requests.Session()
        self._session.headers.update({"APCA-API-KEY-ID": key_id,
                                      "APCA-API-SECRET-KEY": secret_key})
        self.intraday_feed = intraday_feed
        self.daily_feed = daily_feed
        self.quote_feed = quote_feed
        self.sip_delayed = sip_delayed
        self.fallback = fallback
        self._cache = _TTLCache()
        self._intraday_ttl = intraday_cache_seconds
        self._daily_ttl = daily_cache_seconds
        self._quote_ttl = quote_cache_seconds
        self._limiter = _RateLimiter(max_calls_per_minute)
        self.max_quote_age = timedelta(seconds=max_quote_age_seconds)
        self.max_intraday_staleness = timedelta(minutes=max_intraday_staleness_minutes)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def _intraday_stale(self, df: pd.DataFrame) -> bool:
        """During the session, is the newest bar too old to trade on?"""
        if df.empty or not self._market_open_now():
            return False
        now = pd.Timestamp(self._clock()).tz_convert(TZ)
        last = df.index[-1]
        limit_min = self.max_intraday_staleness.total_seconds() / 60
        if last.date() < now.date():
            # nothing yet today: fine in the first minutes after the open
            return (now.hour * 60 + now.minute) >= 9 * 60 + 30 + limit_min
        return now - last > self.max_intraday_staleness

    def _market_open_now(self) -> bool:
        local = pd.Timestamp(self._clock()).tz_convert(TZ)
        if local.weekday() >= 5:
            return False
        minutes = local.hour * 60 + local.minute
        return 9 * 60 + 30 <= minutes < 16 * 60

    # -- internals ---------------------------------------------------------

    def _get(self, path: str, params: dict) -> Optional[dict]:
        self._limiter.acquire()
        try:
            resp = self._session.get(f"{DATA_URL}/{path}", params=params, timeout=20)
        except Exception as exc:
            log.warning("alpaca data %s failed: %s", path, exc)
            return None
        if resp.status_code != 200:
            log.warning("alpaca data %s -> HTTP %s: %s", path, resp.status_code,
                        str(getattr(resp, "text", ""))[:200])
            return None
        try:
            return resp.json()
        except ValueError:
            return None

    def _end_for(self, feed: str) -> Optional[str]:
        if feed == "sip" and self.sip_delayed:
            end = datetime.now(timezone.utc) - timedelta(minutes=16)
            return end.strftime("%Y-%m-%dT%H:%M:%SZ")
        return None

    def _bars(self, symbol: str, timeframe: str, start: datetime, feed: str,
              intraday: bool) -> pd.DataFrame:
        params = {"timeframe": timeframe, "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "adjustment": "split", "feed": feed, "limit": 10000, "sort": "asc"}
        end = self._end_for(feed)
        if end:
            params["end"] = end
        bars: list[dict] = []
        for _ in range(20):  # page cap; a 5-day 5-minute pull is one page
            payload = self._get(f"{symbol}/bars", params)
            if payload is None:
                return pd.DataFrame(columns=OHLCV_COLUMNS)
            bars.extend(payload.get("bars") or [])
            token = payload.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        return bars_to_frame(bars, intraday)

    def _fallback(self, method: str, *args):
        if self.fallback is None:
            return None
        try:
            return getattr(self.fallback, method)(*args)
        except Exception as exc:
            log.debug("fallback %s failed: %s", method, exc)
            return None

    # -- public API (same as YFinanceMarketData) ------------------------------

    def get_intraday(self, symbol: str, interval: str = "5m",
                     lookback_days: int = 5) -> pd.DataFrame:
        key = ("intraday", symbol, interval, lookback_days)
        cached = self._cache.get(key, self._intraday_ttl)
        if cached is not None:
            return cached
        timeframe = _TIMEFRAMES.get(interval, "5Min")
        # lookback_days are trading days (yfinance period semantics); pad for weekends
        start = datetime.now(timezone.utc) - timedelta(days=lookback_days + 4)
        df = self._bars(symbol, timeframe, start, self.intraday_feed, intraday=True)
        if self._intraday_stale(df):
            log.debug("alpaca %s intraday (%s) stale; using fallback", symbol, self.intraday_feed)
            stale = self._fallback("get_intraday", symbol, interval, lookback_days)
            if stale is not None and not stale.empty:
                df = stale
        if df.empty:
            df = self._fallback("get_intraday", symbol, interval, lookback_days)
            if df is None:
                df = pd.DataFrame(columns=OHLCV_COLUMNS)
        self._cache.put(key, df)
        return df

    def get_daily(self, symbol: str, lookback_days: int = 120) -> pd.DataFrame:
        key = ("daily", symbol, lookback_days)
        cached = self._cache.get(key, self._daily_ttl)
        if cached is not None:
            return cached
        start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        df = self._bars(symbol, "1Day", start, self.daily_feed, intraday=False)
        if df.empty:
            df = self._fallback("get_daily", symbol, lookback_days)
            if df is None:
                df = pd.DataFrame(columns=OHLCV_COLUMNS)
        self._cache.put(key, df)
        return df

    def get_latest_price(self, symbol: str) -> Optional[float]:
        key = ("quote", symbol)
        cached = self._cache.get(key, self._quote_ttl)
        if cached is not None:
            return cached
        price = None
        payload = self._get(f"{symbol}/trades/latest", {"feed": self.quote_feed})
        if payload:
            trade = payload.get("trade") or {}
            try:
                price = float(trade.get("p") or 0) or None
            except (TypeError, ValueError):
                price = None
            if price is not None and self._market_open_now() and trade.get("t"):
                age = pd.Timestamp(self._clock()) - pd.Timestamp(trade["t"])
                if age > self.max_quote_age:
                    log.debug("alpaca %s last %s trade is %s old; using fallback", symbol,
                              self.quote_feed, age)
                    price = None
        if price is None:
            price = self._fallback("get_latest_price", symbol)
        if price is not None:
            self._cache.put(key, price)
        return price

    def get_earnings_dates(self, symbol: str):
        dates = self._fallback("get_earnings_dates", symbol)
        return dates or []
