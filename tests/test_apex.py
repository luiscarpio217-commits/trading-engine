"""APEX integration: watchlist reader, insider_breakout, multi-day holds,
state persistence, dollar-volume filter, weekly report."""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from trading_engine.apex.watchlist import ApexWatchlist, WatchlistEntry
from trading_engine.cli import format_weekly, weekly_summary
from trading_engine.config import ApexSettings, Config, FilterSettings
from trading_engine.engine import TradingEngine
from trading_engine.execution.order_manager import sessions_held
from trading_engine.execution.paper import PaperBroker
from trading_engine.filters import SignalFilters
from trading_engine.models import Direction, Instrument, Signal
from trading_engine.storage.trade_log import TradeLog
from trading_engine.strategies.base import MarketSnapshot
from trading_engine.strategies.insider_breakout import InsiderBreakoutStrategy
from tests.test_order_manager import make_manager

TODAY = date(2026, 7, 14)                                     # a Tuesday
TUE_1030 = datetime(2026, 7, 14, 14, 30, tzinfo=timezone.utc)  # 10:30 ET
TUE_1556 = datetime(2026, 7, 14, 19, 56, tzinfo=timezone.utc)  # 15:56 ET (flatten window)


def et(day: date, hh: int, mm: int) -> datetime:
    """day hh:mm New York time (EDT, UTC-4, in July) as UTC."""
    return datetime(day.year, day.month, day.day, hh + 4, mm, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# market data builders
# ---------------------------------------------------------------------------


def make_daily(last_completed: date = date(2026, 7, 13), bars: int = 80,
               top: float = 10.0, include_today: bool = True) -> pd.DataFrame:
    idx = pd.bdate_range(end=last_completed, periods=bars)
    closes = pd.Series([top * (0.9 + 0.1 * i / (bars - 1)) for i in range(bars)], index=idx)
    df = pd.DataFrame({"open": closes * 0.995, "high": closes * 1.01, "low": closes * 0.99,
                       "close": closes, "volume": 1_000_000.0})
    if include_today:
        today = pd.Timestamp(last_completed) + pd.offsets.BDay(1)
        df.loc[today] = [top, top * 1.03, top * 0.99, top * 1.02, 400_000.0]
    return df


def make_intraday(day: date = TODAY, last: float = 10.2, bars: int = 12) -> pd.DataFrame:
    start = pd.Timestamp(f"{day} 09:30", tz="America/New_York")
    idx = pd.DatetimeIndex([start + pd.Timedelta(minutes=5 * i) for i in range(bars)])
    closes = [10.0 + (last - 10.0) * i / (bars - 1) for i in range(bars)]
    return pd.DataFrame({"open": closes, "high": [c + 0.02 for c in closes],
                         "low": [c - 0.02 for c in closes], "close": closes,
                         "volume": 50_000.0}, index=idx)


def entry(ticker="XYZ", score=70.0, filed=date(2026, 7, 13), expires=date(2026, 7, 20)):
    return WatchlistEntry(ticker=ticker, score=score, filed_date=filed, expires_on=expires,
                          insider_class="opportunistic", cluster_size=2, role_tier="ceo_cfo",
                          total_value_usd=120_000, insiders=("DOE JANE (CEO)",))


def write_watchlist(path, entries: list[dict], generated_at: datetime | None = None):
    payload = {"version": 1, "generated_at": (generated_at or TUE_1030).isoformat(),
               "ttl_days": 7, "entries": entries}
    path.write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# watchlist reader
# ---------------------------------------------------------------------------


def test_watchlist_filters_score_expiry_and_future(tmp_path):
    path = tmp_path / "watchlist.json"
    write_watchlist(path, [
        {"ticker": "aaa", "score": 80, "filed_date": "2026-07-13", "expires_on": "2026-07-20"},
        {"ticker": "LOW", "score": 30, "filed_date": "2026-07-13", "expires_on": "2026-07-20"},
        {"ticker": "OLD", "score": 90, "filed_date": "2026-07-01", "expires_on": "2026-07-08"},
        {"ticker": "FUT", "score": 90, "filed_date": "2026-07-15", "expires_on": "2026-07-22"},
        {"ticker": "BBB", "score": 60, "filed_date": "2026-07-13", "expires_on": "2026-07-20"},
        {"bad": "row"},
    ])
    wl = ApexWatchlist(path, min_score=50, max_symbols=30)
    assert [e.ticker for e in wl.entries(TODAY, TUE_1030)] == ["AAA", "BBB"]
    assert ApexWatchlist(path, min_score=50, max_symbols=1).entries(TODAY, TUE_1030)[0].ticker == "AAA"


def test_watchlist_reloads_on_change_and_tolerates_missing_or_broken(tmp_path):
    path = tmp_path / "watchlist.json"
    wl = ApexWatchlist(path)
    assert wl.entries(TODAY) == []                       # missing file
    write_watchlist(path, [{"ticker": "AAA", "score": 80, "filed_date": "2026-07-13",
                            "expires_on": "2026-07-20"}])
    assert [e.ticker for e in wl.entries(TODAY, TUE_1030)] == ["AAA"]
    path.write_text("{not json")
    os.utime(path, (time.time() + 5, time.time() + 5))
    assert wl.entries(TODAY, TUE_1030) == []             # broken file -> nothing, no crash


# ---------------------------------------------------------------------------
# insider_breakout strategy
# ---------------------------------------------------------------------------


def snap(intraday=None, daily=None, symbol="XYZ"):
    return MarketSnapshot(symbol=symbol, intraday=make_intraday() if intraday is None else intraday,
                          daily=make_daily() if daily is None else daily, now=TUE_1030)


def test_breakout_signal_when_price_clears_prior_high():
    sig = InsiderBreakoutStrategy(ApexSettings()).evaluate(snap(), entry())
    assert sig is not None
    assert sig.source == "apex" and sig.strategy == "insider_breakout"
    assert sig.instrument is Instrument.EQUITY and sig.direction is Direction.LONG
    assert sig.max_hold_days == 5
    assert sig.entry_price == pytest.approx(10.2)
    assert sig.stop_loss == pytest.approx(10.2 * 0.94)
    assert sig.target_price == pytest.approx(10.2 * 1.12)
    assert any("cluster: 2 insiders" in r for r in sig.reasons)


def test_no_signal_without_entry_or_below_prior_high_or_chasing():
    strat = InsiderBreakoutStrategy(ApexSettings())
    assert strat.evaluate(snap(), None) is None
    assert strat.evaluate(snap(intraday=make_intraday(last=10.05)), entry()) is None  # < 10.10
    assert strat.evaluate(snap(intraday=make_intraday(last=10.6)), entry()) is None   # > +3%


def test_no_signal_when_far_from_52_week_high():
    daily = make_daily()
    daily.iloc[5, daily.columns.get_loc("high")] = 20.0   # old high 2x above today
    assert InsiderBreakoutStrategy(ApexSettings()).evaluate(snap(daily=daily), entry()) is None
    off = ApexSettings(near_high_pct=None)
    assert InsiderBreakoutStrategy(off).evaluate(snap(daily=daily), entry()) is not None


def test_no_signal_outside_price_band_or_short_history():
    assert InsiderBreakoutStrategy(ApexSettings(max_price=5.0)).evaluate(snap(), entry()) is None
    short = make_daily(bars=30)
    assert InsiderBreakoutStrategy(ApexSettings()).evaluate(snap(daily=short), entry()) is None


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------


def test_dollar_volume_override():
    filters = SignalFilters(FilterSettings(min_avg_daily_volume=5_000_000), lambda s: [])
    daily = make_daily()     # 1M shares x ~$10 = ~$10M/day
    assert not filters.check_symbol("XYZ", daily, TODAY)[0]          # share floor fails
    assert filters.check_symbol("XYZ", daily, TODAY, min_volume=0,
                                min_dollar_volume=200_000)[0]
    ok, reason = filters.check_symbol("XYZ", daily, TODAY, min_volume=0,
                                      min_dollar_volume=50_000_000)
    assert not ok and "dollar volume" in reason


# ---------------------------------------------------------------------------
# multi-day holds in the order manager
# ---------------------------------------------------------------------------


def test_sessions_held_counts_weekdays_inclusive():
    assert sessions_held(date(2026, 7, 14), date(2026, 7, 14)) == 1
    assert sessions_held(date(2026, 7, 17), date(2026, 7, 20)) == 2   # Fri -> Mon
    assert sessions_held(date(2026, 7, 14), date(2026, 7, 20)) == 5
    assert sessions_held(date(2026, 7, 14), date(2026, 7, 13)) == 0


def _sig(symbol, hold=None):
    return Signal(symbol=symbol, strategy="x", direction=Direction.LONG, entry_price=100.0,
                  stop_loss=90.0, target_price=120.0, instrument=Instrument.EQUITY,
                  max_hold_days=hold)


def test_eod_flatten_skips_swing_and_time_stop_closes_on_last_session():
    om, broker, closed = make_manager()
    for s in ("DAY", "SWING"):
        broker.set_mark(s, 100.0)
    om.execute_signal(_sig("DAY"), qty=10, session_date=TODAY)
    om.execute_signal(_sig("SWING", hold=3), qty=10, session_date=TODAY)

    om.flatten_all("eod_flatten", include_swing=False)
    assert [c.symbol for c in closed] == ["DAY"]
    assert [p.symbol for p in om.open_positions()] == ["SWING"]

    assert om.close_due_swings(date(2026, 7, 15)) == 0      # session 2 of 3
    assert om.close_due_swings(date(2026, 7, 16)) == 1      # session 3 -> close
    assert closed[-1].symbol == "SWING" and closed[-1].exit_reason == "time_stop"


def test_option_signals_never_become_swing_positions():
    from tests.test_order_manager import option_signal
    om, broker, _ = make_manager()
    broker.set_mark("TEST", 100.0)
    sig = option_signal()
    sig.max_hold_days = 5
    broker.set_mark(sig.contract.occ, 2.0)
    om.execute_signal(sig, qty=1, option_premium=2.0, session_date=TODAY)
    assert om.open_positions()[0].max_hold_days is None


# ---------------------------------------------------------------------------
# engine end to end
# ---------------------------------------------------------------------------


class FakeMarket:
    def __init__(self):
        self.daily = make_daily()
        self.intraday = make_intraday()
        self.price = 10.2

    def get_daily(self, symbol, lookback_days=120):
        return self.daily

    def get_intraday(self, symbol, interval="5m", lookback_days=5):
        return self.intraday

    def get_latest_price(self, symbol):
        return self.price

    def get_earnings_dates(self, symbol):
        return []


def make_engine(tmp_path, state=False, broker=None):
    wl = tmp_path / "watchlist.json"
    if not wl.exists():
        write_watchlist(wl, [{"ticker": "XYZ", "score": 72, "filed_date": "2026-07-13",
                              "expires_on": "2026-07-20", "insider_class": "opportunistic",
                              "insiders": ["DOE JANE (CEO)"]}])
    cfg = Config()
    cfg.engine.tickers = []
    cfg.engine.db_path = str(tmp_path / "trading.db")
    cfg.engine.csv_dir = str(tmp_path)
    if state:
        cfg.engine.state_path = str(tmp_path / "state.json")
    cfg.strategies.options_flow.enabled = False
    cfg.execution.trade_options = False
    cfg.risk.max_position_notional_pct = 1.0
    cfg.apex.enabled = True
    cfg.apex.watchlist_path = str(wl)
    market = FakeMarket()
    broker = broker or PaperBroker(starting_cash=100_000, slippage_bps=0)
    engine = TradingEngine(config=cfg, broker=broker, market_data=market, options_data=None,
                           earnings_lookup=lambda s: [],
                           trade_log=TradeLog(cfg.engine.db_path, cfg.engine.csv_dir))
    return engine, market, broker


def test_engine_trades_watchlist_stock_and_holds_it_for_five_sessions(tmp_path):
    engine, market, broker = make_engine(tmp_path)
    signals = engine.scan_once(now=TUE_1030)
    assert [s.source for s in signals] == ["apex"]
    pos = engine.orders.open_positions()[0]
    assert (pos.symbol, pos.source, pos.max_hold_days, pos.entry_session) == \
        ("XYZ", "apex", 5, TODAY)

    # cooldown: a second scan the same day doesn't re-signal
    assert engine.scan_once(now=TUE_1030 + timedelta(minutes=5)) == []

    engine.manage_once(now=TUE_1556)                     # day 1 flatten window: kept
    assert len(engine.orders.open_positions()) == 1
    engine.manage_once(now=et(date(2026, 7, 17), 15, 56))  # Fri = session 4: kept
    assert len(engine.orders.open_positions()) == 1
    engine.manage_once(now=et(date(2026, 7, 20), 15, 56))  # Mon = session 5: time stop
    assert engine.orders.open_positions() == []

    trades = engine.trade_log.recent_trades(5)
    assert trades[0]["source"] == "apex" and trades[0]["exit_reason"] == "time_stop"
    assert engine.trade_log.stats()["by_source"]["apex"]["trades"] == 1
    engine.shutdown()


def test_engine_does_not_trade_apex_on_live_broker(tmp_path):
    from trading_engine.execution.base import Broker
    from trading_engine.models import AccountInfo

    class LiveBroker(Broker):
        name = "alpaca"

        def __init__(self):
            self.submitted = []

        def get_account(self):
            return AccountInfo(equity=100_000, cash=100_000, buying_power=100_000)

        def submit_order(self, order):
            self.submitted.append(order)
            return order

        def refresh_order(self, order):
            return order

        def cancel_order(self, order):
            pass

        def list_positions(self):
            return []

    live = LiveBroker()
    engine, _, _ = make_engine(tmp_path, broker=live)
    signals = engine.scan_once(now=TUE_1030)
    assert len(signals) == 1                     # signal logged...
    assert live.submitted == []                  # ...but never sent to a real broker
    engine.shutdown()


def test_state_survives_restart(tmp_path):
    engine, market, broker = make_engine(tmp_path, state=True)
    engine.scan_once(now=TUE_1030)
    engine.manage_once(now=TUE_1556)            # saves state
    qty = engine.orders.open_positions()[0].qty
    cash = broker.get_account().cash
    engine.shutdown()

    fresh = PaperBroker(starting_cash=100_000, slippage_bps=0)
    engine2, _, _ = make_engine(tmp_path, state=True, broker=fresh)
    restored = engine2.orders.open_positions()
    assert len(restored) == 1
    pos = restored[0]
    assert (pos.symbol, pos.qty, pos.source, pos.max_hold_days, pos.entry_session) == \
        ("XYZ", qty, "apex", 5, TODAY)
    assert fresh.get_account().cash == pytest.approx(cash)
    assert [o.note for o in engine2.orders.open_orders()] == ["stop_loss"]  # stop re-placed

    engine2.manage_once(now=et(date(2026, 7, 20), 15, 56))
    assert engine2.orders.open_positions() == []
    engine2.shutdown()


# ---------------------------------------------------------------------------
# weekly report
# ---------------------------------------------------------------------------


def test_weekly_summary_groups_by_source():
    now = datetime(2026, 7, 17, 21, 0, tzinfo=timezone.utc)
    trades = [
        {"exit_time": "2026-07-16T15:00:00+00:00", "source": "apex", "pnl": 12.5,
         "underlying": "XYZ"},
        {"exit_time": "2026-07-15T15:00:00+00:00", "source": "apex", "pnl": -4.0,
         "underlying": "ABC"},
        {"exit_time": "2026-07-15T15:00:00+00:00", "source": "edge2", "pnl": 3.0,
         "underlying": "QQQ"},
        {"exit_time": "2026-07-01T15:00:00+00:00", "source": "apex", "pnl": 100.0,
         "underlying": "OLD"},
    ]
    s = weekly_summary(trades, days=7, now=now)
    assert s["total_pnl"] == 11.5
    assert s["by_source"]["apex"]["trades"] == 2 and s["by_source"]["apex"]["wins"] == 1
    text = format_weekly(s)
    assert "apex: 2 trades, 50% winners, +8.50" in text
    assert "edge2: 1 trades" in text
