"""EDGE2 port: seam behavior, source tagging end-to-end, per-source stats.

The scanner logic itself is EDGE2's (ported verbatim); these tests cover the
integration contract: the first-flag seam, the td -> Signal mapping, the
paper-only guard, the source='edge2' tag surviving signal -> order ->
position -> trade, the schema migration, and keyword-only NLP mode.
"""

import sqlite3
import sys
from datetime import datetime, timezone

import pytest

from tests.conftest import make_breakout_df
from tests.test_engine import FakeMarketData
from trading_engine.config import Config
from trading_engine.edge2 import database as edge2_db
from trading_engine.edge2 import scanner as edge2_scanner
from trading_engine.edge2.bridge import Edge2Bridge, td_to_signal
from trading_engine.edge2.database import init_db, save_flagged_stock, set_db_path, was_flagged_today
from trading_engine.engine import TradingEngine
from trading_engine.execution.paper import PaperBroker
from trading_engine.models import Direction, Instrument, TradeRecord
from trading_engine.storage.trade_log import TradeLog

# A realistic td dict exactly as scan_universe() builds it (values follow
# calculate_setup(): entry zone price +/-1%, stop below, 1.5R/3R/5R targets).
SAMPLE_TD = {
    "ticker": "ABCD",
    "price": 4.52,
    "prev_close": 3.80,
    "gap_percent": 18.95,
    "volume": 8_400_000,
    "avg_volume": 1_900_000,
    "volume_ratio": 4.42,
    "session": "intraday",
    "setup_tag": "Gap-and-Go (High Vol)",
    "entry_low": 4.47,
    "entry_high": 4.57,
    "stop_loss": 3.80,
    "target_1": 5.48,
    "target_2": 6.48,
    "target_3": 7.82,
    "risk_reward": 3.0,
    "news": "[]",
    "catalyst_score": 0.72,
    "catalyst_category": "fda_regulatory",
    "catalyst_velocity": "high_velocity",
}


@pytest.fixture(autouse=True)
def reset_scanner_hook():
    """The seam hook is a module global; never leak it between tests."""
    yield
    edge2_scanner.on_first_flag = None


def make_edge2_engine(tmp_path, broker=None):
    cfg = Config()
    cfg.engine.tickers = ["TEST"]
    cfg.engine.db_path = str(tmp_path / "trading.db")
    cfg.engine.csv_dir = str(tmp_path)
    cfg.strategies.momentum_breakout.enabled = False
    cfg.strategies.options_flow.enabled = False
    cfg.execution.trade_options = False
    cfg.risk.max_position_notional_pct = 1.0
    cfg.edge2.enabled = True
    market = FakeMarketData(make_breakout_df())
    engine = TradingEngine(
        config=cfg,
        broker=broker or PaperBroker(starting_cash=100_000, slippage_bps=0),
        market_data=market,
        earnings_lookup=lambda s: [],
        trade_log=TradeLog(cfg.engine.db_path, cfg.engine.csv_dir),
    )
    return engine, market


class TestTdMapping:
    def test_maps_straight_off_td(self):
        sig = td_to_signal(SAMPLE_TD)
        assert sig.symbol == "ABCD"
        assert sig.direction is Direction.LONG
        assert sig.instrument is Instrument.EQUITY
        assert sig.entry_price == pytest.approx(4.52)     # td['price']
        assert sig.stop_loss == pytest.approx(3.80)       # td['stop_loss']
        assert sig.target_price == pytest.approx(6.48)    # td['target_2']
        assert sig.source == "edge2"
        assert sig.strategy == "edge2_scanner"
        joined = " | ".join(sig.reasons)
        assert "Gap-and-Go (High Vol)" in joined
        assert "5.48" in joined and "7.82" in joined      # t1/t3 preserved
        assert "fda_regulatory" in joined


class TestSeam:
    def test_first_flag_fires_hook_once_per_day(self, tmp_path):
        set_db_path(tmp_path / "edge2.db")
        init_db()
        calls = []
        edge2_scanner.on_first_flag = calls.append

        for _ in range(3):  # three scan cycles flag the same ticker
            first = not was_flagged_today("ABCD")
            save_flagged_stock(SAMPLE_TD)
            if first:
                edge2_scanner._open_ironfrost_paper_trade(SAMPLE_TD)
        assert len(calls) == 1
        assert calls[0]["ticker"] == "ABCD"

    def test_hook_failure_never_raises(self):
        def boom(td):
            raise RuntimeError("bridge exploded")
        edge2_scanner.on_first_flag = boom
        edge2_scanner._open_ironfrost_paper_trade(SAMPLE_TD)  # must not raise

    def test_no_hook_is_noop(self):
        edge2_scanner.on_first_flag = None
        edge2_scanner._open_ironfrost_paper_trade(SAMPLE_TD)


class TestBridgeEndToEnd:
    def test_flag_becomes_paper_trade_tagged_edge2(self, tmp_path):
        engine, market = make_edge2_engine(tmp_path)
        engine.edge2_bridge.open_paper_trade(SAMPLE_TD)

        positions = engine.orders.open_positions()
        assert len(positions) == 1
        pos = positions[0]
        assert pos.source == "edge2"
        assert pos.instrument is Instrument.EQUITY
        assert pos.stop_loss == pytest.approx(3.80)
        # sized by IronFrost's own risk math: $1000 budget / $0.72 per-share risk
        assert pos.qty == 1000 // (4.52 - 3.80)

        sig_row = engine.trade_log.recent_signals(1)[0]
        assert sig_row["source"] == "edge2"
        assert sig_row["status"].startswith("executed")

    def test_source_survives_through_stop_out(self, tmp_path):
        engine, market = make_edge2_engine(tmp_path)
        engine.edge2_bridge.open_paper_trade(SAMPLE_TD)
        market.latest = 3.50                       # gap through the stop
        engine.manage_once(now=datetime.now(timezone.utc))
        trades = engine.trade_log.recent_trades(5)
        assert len(trades) == 1
        assert trades[0]["source"] == "edge2"
        assert trades[0]["pnl"] < 0
        assert engine.risk.realized_pnl_today() == pytest.approx(trades[0]["pnl"])

    def test_paper_only_guard(self, tmp_path, caplog):
        import logging

        engine, market = make_edge2_engine(tmp_path)

        class NotPaper:                            # stands in for a live adapter
            name = "alpaca"
        engine.broker = NotPaper()
        with caplog.at_level(logging.WARNING, logger="trading_engine.edge2.bridge"):
            result = engine.edge2_bridge.open_paper_trade(SAMPLE_TD)
        assert result is None
        # the refusal must be logged, not silent
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING
                    and "refusing to trade" in r.getMessage()]
        assert len(warnings) == 1
        assert "paper-only" in warnings[0].getMessage()
        # ...and absolutely nothing must have happened
        assert engine.orders.open_positions() == []
        assert engine.orders.open_orders() == []
        assert engine.trade_log.recent_signals(5) == []   # not even journaled
        assert engine.trade_log.recent_trades(5) == []

    def test_bad_stop_skipped(self, tmp_path):
        engine, market = make_edge2_engine(tmp_path)
        td = dict(SAMPLE_TD, stop_loss=4.60)       # stop above entry: nonsense
        assert engine.edge2_bridge.open_paper_trade(td) is None
        assert engine.orders.open_positions() == []


class TestEngineWiring:
    def test_enabled_engine_wires_hook_and_one_db(self, tmp_path):
        engine, market = make_edge2_engine(tmp_path)
        assert engine.edge2_bridge is not None
        assert edge2_scanner.on_first_flag == engine.edge2_bridge.open_paper_trade
        assert edge2_db.DB_PATH == engine.config.engine.db_path
        # EDGE2 tables created inside IronFrost's trading.db (one SQLite DB)
        conn = sqlite3.connect(engine.config.engine.db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        conn.close()
        assert {"flagged_stocks", "flag_outcomes", "trades", "signals"} <= tables

    def test_disabled_by_default(self, tmp_path):
        cfg = Config()
        assert cfg.edge2.enabled is False

    def test_yaml_enable(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("edge2:\n  enabled: true\n  scan_interval_seconds: 90\n")
        cfg = Config.load(p)
        assert cfg.edge2.enabled is True
        assert cfg.edge2.scan_interval_seconds == 90


class TestKeywordModeOnly:
    def test_nlp_stays_keyword_mode_without_torch(self):
        assert edge2_scanner._nlp._enable_model is False
        assert "torch" not in sys.modules
        assert "transformers" not in sys.modules


class TestSourceStats:
    @staticmethod
    def trade(pnl, source):
        now = datetime.now(timezone.utc)
        return TradeRecord(symbol="X", underlying="X", strategy="s",
                           direction=Direction.LONG, instrument=Instrument.EQUITY,
                           qty=1, multiplier=1, entry_price=100.0,
                           exit_price=100.0 + pnl, entry_time=now, exit_time=now,
                           pnl=pnl, exit_reason="target", source=source)

    def test_separate_pnl_per_source(self, tmp_path):
        log = TradeLog(tmp_path / "t.db")
        for pnl, source in ((200.0, "ironfrost"), (-100.0, "ironfrost"),
                            (300.0, "edge2"), (-150.0, "edge2"), (-50.0, "edge2")):
            log.log_trade(self.trade(pnl, source))
        by_source = log.stats()["by_source"]
        iron, edge = by_source["ironfrost"], by_source["edge2"]
        assert iron["trades"] == 2 and edge["trades"] == 3
        assert iron["win_rate"] == pytest.approx(0.5)
        assert edge["win_rate"] == pytest.approx(1 / 3)
        assert iron["total_pnl"] == pytest.approx(100.0)
        assert edge["total_pnl"] == pytest.approx(100.0)
        assert iron["profit_factor"] == pytest.approx(2.0)
        assert edge["profit_factor"] == pytest.approx(1.5)
        log.close()

    def test_migration_backfills_ironfrost(self, tmp_path):
        """A pre-EDGE2 trading.db gains the source column, old rows tagged ironfrost."""
        db = tmp_path / "legacy.db"
        conn = sqlite3.connect(db)
        conn.execute("""CREATE TABLE trades (
            id TEXT PRIMARY KEY, symbol TEXT NOT NULL, underlying TEXT,
            strategy TEXT, direction TEXT, instrument TEXT,
            qty REAL, multiplier INTEGER, entry_price REAL, exit_price REAL,
            entry_time TEXT, exit_time TEXT, pnl REAL, pnl_pct REAL,
            exit_reason TEXT, signal_id TEXT)""")
        conn.execute(
            "INSERT INTO trades VALUES ('t1','SPY','SPY','momentum_breakout','long',"
            "'equity',10,1,100,101,'2026-07-20T14:00:00+00:00',"
            "'2026-07-20T15:00:00+00:00',10.0,1.0,'target','s1')")
        conn.commit()
        conn.close()

        log = TradeLog(db)
        rows = log.recent_trades(5)
        assert rows[0]["source"] == "ironfrost"
        assert log.stats()["by_source"]["ironfrost"]["trades"] == 1
        log.close()
