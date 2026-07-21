"""Offline integration test: fake market data -> signal -> order -> stop-out."""

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from trading_engine.config import Config
from trading_engine.engine import TradingEngine
from trading_engine.execution.paper import PaperBroker
from trading_engine.storage.trade_log import TradeLog
from tests.conftest import make_breakout_df, make_chain

TUESDAY_1030_ET = datetime(2026, 7, 14, 14, 30, tzinfo=timezone.utc)


class FakeMarketData:
    def __init__(self, intraday: pd.DataFrame):
        self.intraday = intraday
        self.latest = float(intraday["close"].iloc[-1])
        self.daily = pd.DataFrame({
            "open": [100.0] * 30, "high": [101.0] * 30, "low": [99.0] * 30,
            "close": [100.0] * 30, "volume": [5_000_000] * 30,
        })

    def get_intraday(self, symbol, interval="5m", lookback_days=5):
        return self.intraday

    def get_daily(self, symbol, lookback_days=120):
        return self.daily

    def get_latest_price(self, symbol):
        return self.latest

    def get_earnings_dates(self, symbol):
        return []


class FakeOptionsData:
    def __init__(self, chain):
        self.chain = chain

    def get_chain(self, symbol):
        return self.chain


def make_engine(tmp_path, trade_options=False, chain=None, momentum=True,
                options_flow=False):
    cfg = Config()
    cfg.engine.tickers = ["TEST"]
    cfg.engine.db_path = str(tmp_path / "trading.db")
    cfg.engine.csv_dir = str(tmp_path)
    cfg.strategies.momentum_breakout.enabled = momentum
    cfg.strategies.options_flow.enabled = options_flow
    cfg.execution.trade_options = trade_options
    cfg.risk.max_position_notional_pct = 1.0

    market = FakeMarketData(make_breakout_df())
    broker = PaperBroker(starting_cash=100_000, slippage_bps=0)
    engine = TradingEngine(
        config=cfg,
        broker=broker,
        market_data=market,
        options_data=FakeOptionsData(chain) if chain is not None else None,
        earnings_lookup=lambda s: [],
        trade_log=TradeLog(cfg.engine.db_path, cfg.engine.csv_dir),
    )
    return engine, market, broker


def test_scan_generates_and_executes_equity_signal(tmp_path):
    engine, market, broker = make_engine(tmp_path)
    signals = engine.scan_once(now=TUESDAY_1030_ET)
    assert len(signals) == 1
    sig = signals[0]
    assert sig.strategy == "momentum_breakout"

    positions = engine.orders.open_positions()
    assert len(positions) == 1
    pos = positions[0]
    # 1% of 100k = $1000 budget at (1.5 * ATR) per-share risk, capped by
    # notional (100% of equity / entry price)
    risk_qty = 1000 // sig.risk_per_share
    notional_qty = 100_000 // sig.entry_price
    assert pos.qty == pytest.approx(min(risk_qty, notional_qty))
    assert pos.qty > 0

    logged = engine.trade_log.recent_signals(5)
    assert logged[0]["status"].startswith("executed")

    # cooldown: an identical second scan stays quiet
    assert engine.scan_once(now=TUESDAY_1030_ET) == []


def test_stop_out_records_trade_and_risk(tmp_path):
    engine, market, broker = make_engine(tmp_path)
    [sig] = engine.scan_once(now=TUESDAY_1030_ET)
    # gap the underlying below the stop and run a manage cycle
    market.latest = sig.stop_loss - 1.0
    engine.manage_once(now=TUESDAY_1030_ET)
    assert engine.orders.open_positions() == []
    trades = engine.trade_log.recent_trades(5)
    assert len(trades) == 1
    assert trades[0]["exit_reason"] in ("stop_loss", "premium_stop")
    assert trades[0]["pnl"] < 0
    assert engine.risk.realized_pnl_today() == pytest.approx(trades[0]["pnl"])


def test_option_execution_path(tmp_path):
    spot = float(make_breakout_df()["close"].iloc[-1])
    chain = make_chain(spot=spot, asof=date.today())
    engine, market, broker = make_engine(tmp_path, trade_options=True, chain=chain)
    [sig] = engine.scan_once(now=TUESDAY_1030_ET)
    assert sig.contract is not None
    positions = engine.orders.open_positions()
    assert len(positions) == 1
    assert positions[0].multiplier == 100
    assert positions[0].symbol == sig.contract.occ

    # underlying rallies through target -> close at estimated premium, profit
    market.latest = sig.target_price + 0.5
    engine.manage_once(now=TUESDAY_1030_ET)
    trades = engine.trade_log.recent_trades(5)
    assert len(trades) == 1
    assert trades[0]["exit_reason"] == "target"
    assert trades[0]["pnl"] > 0


def test_options_flow_signal_through_engine(tmp_path):
    spot = float(make_breakout_df()["close"].iloc[-1])
    chain = make_chain(spot=spot, asof=date.today())
    engine, market, broker = make_engine(tmp_path, trade_options=True, chain=chain,
                                         momentum=False, options_flow=True)
    signals = engine.scan_once(now=TUESDAY_1030_ET)
    assert len(signals) == 1
    assert signals[0].strategy == "options_flow"
    assert engine.orders.open_positions()


def test_outside_market_hours_no_execution(tmp_path):
    engine, market, broker = make_engine(tmp_path)
    sunday = datetime(2026, 7, 12, 15, 0, tzinfo=timezone.utc)
    assert engine.scan_once(now=sunday) == []
    assert engine.orders.open_positions() == []


def test_daily_loss_halt_blocks_new_signals(tmp_path):
    engine, market, broker = make_engine(tmp_path)
    engine._ensure_session(TUESDAY_1030_ET)
    engine.risk.on_trade_closed(-5000)  # > 3% of 100k
    assert engine.risk.halted
    signals = engine.scan_once(now=TUESDAY_1030_ET)
    assert len(signals) == 1  # signal still generated and logged...
    assert engine.orders.open_positions() == []  # ...but nothing executed
    status = engine.trade_log.recent_signals(1)[0]["status"]
    assert status.startswith("blocked")


def test_status_snapshot_shape(tmp_path):
    engine, market, broker = make_engine(tmp_path)
    engine.scan_once(now=TUESDAY_1030_ET)
    status = engine.get_status()
    assert status["broker"] == "paper"
    assert "equity" in status["account"]
    assert isinstance(status["positions"], list) and status["positions"]
    assert status["day"]["max_daily_loss"] == pytest.approx(3000.0)
    assert "win_rate" in status["stats"]


def test_web_dashboard_endpoints(tmp_path):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from trading_engine.dashboard.web import create_app

    engine, market, broker = make_engine(tmp_path)
    engine.scan_once(now=TUESDAY_1030_ET)
    client = TestClient(create_app(engine.get_status))
    assert "Day Trading Engine" in client.get("/").text
    status = client.get("/api/status").json()
    assert status["broker"] == "paper"
    assert client.get("/api/positions").json()
    assert client.get("/api/stats").status_code == 200
