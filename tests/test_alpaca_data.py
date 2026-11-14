"""Alpaca market data provider: parsing, feeds, fallbacks, engine wiring."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from trading_engine.config import Config
from trading_engine.data.alpaca_data import AlpacaMarketData, bars_to_frame
from trading_engine.data.market_data import YFinanceMarketData
from trading_engine.engine import build_market_data

TUE_1030 = datetime(2026, 7, 14, 14, 30, tzinfo=timezone.utc)   # 10:30 ET
SAT = datetime(2026, 7, 18, 14, 30, tzinfo=timezone.utc)


class FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {}
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, routes):
        self.routes = routes          # path suffix -> list of responses (popped in order)
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        for suffix, responses in self.routes.items():
            if url.endswith(suffix):
                return responses.pop(0) if len(responses) > 1 else responses[0]
        return FakeResponse(404, {"message": "not found"})


class FakeYahoo:
    def __init__(self):
        self.calls = []

    def get_intraday(self, *a):
        self.calls.append("intraday")
        return pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [9.99],
                             "volume": [1.0]},
                            index=pd.DatetimeIndex([pd.Timestamp("2026-07-14 10:25",
                                                                 tz="America/New_York")]))

    def get_daily(self, *a):
        self.calls.append("daily")
        return pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
                             "volume": [1.0]})

    def get_latest_price(self, *a):
        self.calls.append("quote")
        return 42.0

    def get_earnings_dates(self, *a):
        return [date(2026, 7, 30)]


def intraday_bars(times_utc: list[str], px=10.0):
    return {"bars": [{"t": t, "o": px, "h": px + 0.1, "l": px - 0.1, "c": px, "v": 100}
                     for t in times_utc], "symbol": "XYZ", "next_page_token": None}


def make(routes, fallback=None, now=TUE_1030, **kw):
    session = FakeSession(routes)
    md = AlpacaMarketData("k", "s", fallback=fallback, session=session,
                          clock=lambda: now, **kw)
    return md, session


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_intraday_frame_is_new_york_regular_session_only():
    df = bars_to_frame(intraday_bars(["2026-07-14T12:00:00Z",       # 08:00 ET pre-market
                                      "2026-07-14T13:30:00Z",       # 09:30 ET
                                      "2026-07-14T19:55:00Z",       # 15:55 ET
                                      "2026-07-14T20:05:00Z"])["bars"], intraday=True)
    assert [t.strftime("%H:%M") for t in df.index] == ["09:30", "15:55"]
    assert str(df.index.tz) == "America/New_York"
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_daily_frame_uses_new_york_dates():
    df = bars_to_frame([{"t": "2026-07-13T04:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5,
                         "v": 10},
                        {"t": "2026-01-05T05:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5,
                         "v": 10}], intraday=False)
    assert [d.date() for d in df.index] == [date(2026, 1, 5), date(2026, 7, 13)]
    assert df.index.tz is None


# ---------------------------------------------------------------------------
# requests: feeds, headers, delayed SIP, pagination
# ---------------------------------------------------------------------------


def test_daily_uses_sip_with_delayed_end_and_split_adjustment():
    page1 = {"bars": [{"t": "2026-07-10T04:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 5}],
             "next_page_token": "abc"}
    page2 = {"bars": [{"t": "2026-07-13T04:00:00Z", "o": 2, "h": 2, "l": 2, "c": 2, "v": 5}],
             "next_page_token": None}
    md, session = make({"/XYZ/bars": [FakeResponse(200, page1), FakeResponse(200, page2)]})
    df = md.get_daily("XYZ", 120)
    assert len(df) == 2
    url, params = session.calls[0]
    assert url == "https://data.alpaca.markets/v2/stocks/XYZ/bars"
    assert params["feed"] == "sip" and params["timeframe"] == "1Day"
    assert params["adjustment"] == "split" and "end" in params
    assert session.calls[1][1]["page_token"] == "abc"
    assert session.headers["APCA-API-KEY-ID"] == "k"
    md.get_daily("XYZ", 120)
    assert len(session.calls) == 2                     # cached


def test_intraday_uses_iex_without_end():
    md, session = make({"/XYZ/bars": [FakeResponse(200, intraday_bars(
        ["2026-07-14T14:20:00Z", "2026-07-14T14:25:00Z"]))]})
    df = md.get_intraday("XYZ", "5m", 5)
    assert len(df) == 2
    params = session.calls[0][1]
    assert params["feed"] == "iex" and params["timeframe"] == "5Min" and "end" not in params


def test_latest_price_from_iex_trade():
    md, _ = make({"/XYZ/trades/latest": [FakeResponse(200, {
        "symbol": "XYZ", "trade": {"t": "2026-07-14T14:29:30Z", "p": 10.37}})]})
    assert md.get_latest_price("XYZ") == pytest.approx(10.37)


# ---------------------------------------------------------------------------
# fallbacks
# ---------------------------------------------------------------------------


def test_errors_fall_back_to_yahoo():
    yahoo = FakeYahoo()
    md, _ = make({"/XYZ/bars": [FakeResponse(403, {"message": "forbidden"})],
                  "/XYZ/trades/latest": [FakeResponse(500, {})]}, fallback=yahoo)
    assert md.get_daily("XYZ").shape[0] == 1
    assert md.get_latest_price("XYZ") == 42.0
    assert md.get_earnings_dates("XYZ") == [date(2026, 7, 30)]
    assert yahoo.calls == ["daily", "quote"]


def test_stale_iex_quote_during_market_hours_falls_back():
    yahoo = FakeYahoo()
    md, _ = make({"/XYZ/trades/latest": [FakeResponse(200, {
        "trade": {"t": "2026-07-14T13:45:00Z", "p": 9.0}})]}, fallback=yahoo)
    assert md.get_latest_price("XYZ") == 42.0          # 45 min old IEX print -> Yahoo


def test_stale_iex_quote_is_fine_when_market_closed():
    md, _ = make({"/XYZ/trades/latest": [FakeResponse(200, {
        "trade": {"t": "2026-07-17T19:59:00Z", "p": 9.0}})]}, fallback=FakeYahoo(), now=SAT)
    assert md.get_latest_price("XYZ") == 9.0


def test_stale_intraday_bars_fall_back():
    yahoo = FakeYahoo()
    md, _ = make({"/XYZ/bars": [FakeResponse(200, intraday_bars(["2026-07-14T13:35:00Z"]))]},
                 fallback=yahoo)
    df = md.get_intraday("XYZ")
    assert yahoo.calls == ["intraday"] and float(df["close"].iloc[-1]) == pytest.approx(9.99)


# ---------------------------------------------------------------------------
# engine wiring
# ---------------------------------------------------------------------------


def test_build_market_data_picks_provider(monkeypatch):
    cfg = Config()
    assert isinstance(build_market_data(cfg), YFinanceMarketData)
    cfg.data.provider = "alpaca"
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    assert isinstance(build_market_data(cfg), YFinanceMarketData)   # no keys -> Yahoo
    monkeypatch.setenv("ALPACA_API_KEY_ID", "k")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "s")
    md = build_market_data(cfg)
    assert isinstance(md, AlpacaMarketData)
    assert isinstance(md.fallback, YFinanceMarketData)


def test_dashboard_status_shows_apex_watchlist_and_hold_days(tmp_path):
    from tests.test_apex import TUE_1030 as T, make_engine

    engine, _, _ = make_engine(tmp_path)
    engine.scan_once(now=T)
    status = engine.get_status()
    assert status["apex"]["entries"] == 1          # file has one entry
    assert isinstance(status["apex"]["live"], list)  # (expired relative to the real clock)
    pos = status["positions"][0]
    assert pos["source"] == "apex" and pos["max_hold_days"] == 5
    assert pos["sessions_held"] >= 1
    engine.shutdown()
