"""Pre-trade filters: market hours, liquidity, and earnings blackout."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from .config import FilterSettings

log = logging.getLogger(__name__)


class MarketHoursFilter:
    """Regular US equity session: 9:30-16:00 ET, Monday-Friday.

    All comparisons convert timezone-aware "now" into the exchange timezone,
    so the host clock can be UTC (typical VPS) or anything else — session
    boundaries and session dates always key off US/Eastern, including DST.
    Naive datetimes are interpreted as UTC.

    Exchange holidays are not modeled; on a holiday the data layer simply
    produces no fresh bars, so no signals fire anyway.
    """

    def __init__(self, settings: FilterSettings) -> None:
        self._settings = settings
        try:
            self._tz = ZoneInfo(settings.timezone)
        except Exception as exc:  # ZoneInfoNotFoundError on hosts without tzdata
            raise RuntimeError(
                f"timezone database entry '{settings.timezone}' not found - "
                f"install the OS tzdata package (Ubuntu/Debian: "
                f"'apt install tzdata') so market hours resolve correctly"
            ) from exc

    def now_local(self, now: Optional[datetime] = None) -> datetime:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now.astimezone(self._tz)

    def is_open(self, now: Optional[datetime] = None) -> bool:
        local = self.now_local(now)
        if local.weekday() >= 5:  # Saturday/Sunday
            return False
        t = local.time()
        return self._settings.start_time() <= t < self._settings.end_time()

    def is_flatten_window(self, now: Optional[datetime] = None) -> bool:
        """True between the EOD flatten time and the close."""
        local = self.now_local(now)
        if local.weekday() >= 5:
            return False
        t = local.time()
        return self._settings.flatten_time() <= t < self._settings.end_time()

    def session_date(self, now: Optional[datetime] = None) -> date:
        return self.now_local(now).date()


class VolumeFilter:
    """Reject illiquid underlyings by 20-day average daily share volume."""

    def __init__(self, min_avg_daily_volume: float, lookback: int = 20) -> None:
        self._min = min_avg_daily_volume
        self._lookback = lookback

    def passes(self, daily_df: pd.DataFrame) -> bool:
        if daily_df is None or daily_df.empty or "volume" not in daily_df:
            return False
        avg = daily_df["volume"].tail(self._lookback).mean()
        return bool(avg >= self._min)


class EarningsFilter:
    """Block new trades within +/- `blackout_days` of an earnings date.

    `earnings_lookup(symbol) -> list[date]` is injected (yfinance in
    production, a dict in tests). Unknown earnings == no blackout.
    """

    def __init__(self, blackout_days: int,
                 earnings_lookup: Callable[[str], list[date]]) -> None:
        self._days = blackout_days
        self._lookup = earnings_lookup

    def passes(self, symbol: str, today: Optional[date] = None) -> bool:
        today = today or date.today()
        try:
            dates = self._lookup(symbol) or []
        except Exception as exc:
            log.debug("earnings lookup failed for %s: %s", symbol, exc)
            return True
        window = timedelta(days=self._days)
        return not any(abs(d - today) <= window for d in dates)


class SignalFilters:
    """All filters combined; returns (ok, reason) so rejections are loggable."""

    def __init__(self, settings: FilterSettings,
                 earnings_lookup: Callable[[str], list[date]]) -> None:
        self.market_hours = MarketHoursFilter(settings)
        self.volume = VolumeFilter(settings.min_avg_daily_volume)
        self.earnings = EarningsFilter(settings.earnings_blackout_days, earnings_lookup)
        self._settings = settings

    def check_session(self, now: Optional[datetime] = None) -> tuple[bool, str]:
        if self._settings.market_hours_only and not self.market_hours.is_open(now):
            return False, "outside market hours (9:30-16:00 ET)"
        if self._settings.eod_flatten and self.market_hours.is_flatten_window(now):
            return False, "inside end-of-day flatten window"
        return True, ""

    def check_symbol(self, symbol: str, daily_df: pd.DataFrame,
                     today: Optional[date] = None) -> tuple[bool, str]:
        if not self.volume.passes(daily_df):
            return False, f"below minimum average daily volume ({self._settings.min_avg_daily_volume:,.0f})"
        if not self.earnings.passes(symbol, today):
            return False, f"within {self._settings.earnings_blackout_days}d earnings blackout"
        return True, ""
