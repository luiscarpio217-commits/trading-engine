from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from trading_engine.config import FilterSettings
from trading_engine.filters import EarningsFilter, MarketHoursFilter, SignalFilters, VolumeFilter

ET = ZoneInfo("America/New_York")


def et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def test_market_hours_regular_session():
    f = MarketHoursFilter(FilterSettings())
    tuesday = date(2026, 7, 14)
    assert f.is_open(et(2026, 7, 14, 10, 0))
    assert f.is_open(et(2026, 7, 14, 9, 30))
    assert not f.is_open(et(2026, 7, 14, 9, 29))
    assert not f.is_open(et(2026, 7, 14, 16, 0))   # close is exclusive
    assert not f.is_open(et(2026, 7, 18, 12, 0))   # Saturday
    assert f.session_date(et(2026, 7, 14, 10, 0)) == tuesday


def test_market_hours_utc_input():
    f = MarketHoursFilter(FilterSettings())
    # 14:30 UTC == 10:30 ET during daylight saving
    assert f.is_open(datetime(2026, 7, 14, 14, 30, tzinfo=ZoneInfo("UTC")))
    # 13:00 UTC == 9:00 ET -> closed
    assert not f.is_open(datetime(2026, 7, 14, 13, 0, tzinfo=ZoneInfo("UTC")))


def test_flatten_window():
    f = MarketHoursFilter(FilterSettings())
    assert not f.is_flatten_window(et(2026, 7, 14, 15, 54))
    assert f.is_flatten_window(et(2026, 7, 14, 15, 55))
    assert f.is_flatten_window(et(2026, 7, 14, 15, 59))
    assert not f.is_flatten_window(et(2026, 7, 14, 16, 0))


def test_volume_filter():
    f = VolumeFilter(min_avg_daily_volume=1_000_000)
    liquid = pd.DataFrame({"volume": [2_000_000] * 30})
    illiquid = pd.DataFrame({"volume": [200_000] * 30})
    assert f.passes(liquid)
    assert not f.passes(illiquid)
    assert not f.passes(pd.DataFrame())


def test_earnings_blackout():
    earnings = {"AAPL": [date(2026, 7, 15)], "MSFT": [date(2026, 8, 30)]}
    f = EarningsFilter(blackout_days=2, earnings_lookup=lambda s: earnings.get(s, []))
    today = date(2026, 7, 14)
    assert not f.passes("AAPL", today)        # earnings tomorrow -> blocked
    assert f.passes("MSFT", today)            # far away -> fine
    assert f.passes("UNKNOWN", today)         # no data -> no blackout


def test_market_hours_on_utc_server(monkeypatch):
    """A VPS whose local clock is UTC must still key sessions off US/Eastern."""
    import time as _time

    monkeypatch.setenv("TZ", "UTC")
    _time.tzset()
    try:
        f = MarketHoursFilter(FilterSettings())
        utc = ZoneInfo("UTC")
        # summer (EDT, UTC-4): 13:30 UTC == 09:30 ET open, 20:00 UTC == 16:00 ET closed
        assert f.is_open(datetime(2026, 7, 14, 13, 30, tzinfo=utc))
        assert not f.is_open(datetime(2026, 7, 14, 13, 29, tzinfo=utc))
        assert not f.is_open(datetime(2026, 7, 14, 20, 0, tzinfo=utc))
        # winter (EST, UTC-5): open shifts to 14:30 UTC — DST handled
        assert f.is_open(datetime(2026, 1, 13, 14, 30, tzinfo=utc))
        assert not f.is_open(datetime(2026, 1, 13, 14, 29, tzinfo=utc))
        # naive datetimes are interpreted as UTC, not server-local
        assert f.is_open(datetime(2026, 7, 14, 14, 0))
        # session date rolls at ET midnight, not UTC midnight:
        # 01:00 UTC on the 15th is still 21:00 ET on the 14th
        assert f.session_date(datetime(2026, 7, 15, 1, 0, tzinfo=utc)) == date(2026, 7, 14)
        # EOD flatten window in UTC terms: 19:56 UTC == 15:56 ET
        assert f.is_flatten_window(datetime(2026, 7, 14, 19, 56, tzinfo=utc))
    finally:
        monkeypatch.delenv("TZ", raising=False)
        _time.tzset()


def test_missing_tzdata_gives_actionable_error():
    settings = FilterSettings(timezone="Not/AZone")
    with pytest.raises(RuntimeError, match="tzdata"):
        MarketHoursFilter(settings)


def test_signal_filters_combined():
    settings = FilterSettings()
    filters = SignalFilters(settings, earnings_lookup=lambda s: [])
    ok, _ = filters.check_session(et(2026, 7, 14, 10, 0))
    assert ok
    ok, reason = filters.check_session(et(2026, 7, 14, 8, 0))
    assert not ok and "market hours" in reason
    ok, reason = filters.check_session(et(2026, 7, 14, 15, 57))
    assert not ok and "flatten" in reason

    daily = pd.DataFrame({"volume": [5_000_000] * 30})
    ok, _ = filters.check_symbol("SPY", daily, date(2026, 7, 14))
    assert ok
    thin = pd.DataFrame({"volume": [1_000] * 30})
    ok, reason = filters.check_symbol("THIN", thin, date(2026, 7, 14))
    assert not ok and "volume" in reason
