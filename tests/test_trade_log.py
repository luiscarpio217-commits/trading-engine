from datetime import datetime, timezone

import pytest

from trading_engine.models import (Direction, Instrument, Order, OrderSide, Signal,
                                   TradeRecord)
from trading_engine.storage.trade_log import TradeLog


def make_trade(pnl, strategy="momentum_breakout"):
    now = datetime.now(timezone.utc)
    return TradeRecord(symbol="TEST", underlying="TEST", strategy=strategy,
                       direction=Direction.LONG, instrument=Instrument.EQUITY,
                       qty=10, multiplier=1, entry_price=100.0,
                       exit_price=100.0 + pnl / 10, entry_time=now, exit_time=now,
                       pnl=pnl, exit_reason="target")


def test_round_trip_and_stats(tmp_path):
    log = TradeLog(tmp_path / "t.db", tmp_path)
    sig = Signal(symbol="TEST", strategy="momentum_breakout", direction=Direction.LONG,
                 entry_price=100, stop_loss=98, target_price=104,
                 reasons=["breakout above 20-bar level"])
    log.log_signal(sig)
    log.set_signal_status(sig.id, "executed")
    log.log_order(Order(symbol="TEST", side=OrderSide.BUY, qty=10, signal_id=sig.id))

    for pnl in (200.0, -100.0, 300.0, -100.0):
        log.log_trade(make_trade(pnl))
    log.log_equity(100_300, 90_000, 300.0, False)

    signals = log.recent_signals()
    assert signals[0]["id"] == sig.id
    assert signals[0]["status"] == "executed"

    stats = log.stats()
    assert stats["total_trades"] == 4
    assert stats["wins"] == 2
    assert stats["win_rate"] == pytest.approx(0.5)
    assert stats["total_pnl"] == pytest.approx(300.0)
    assert stats["avg_win"] == pytest.approx(250.0)
    assert stats["avg_loss"] == pytest.approx(100.0)
    assert stats["profit_factor"] == pytest.approx(2.5)
    assert stats["by_strategy"]["momentum_breakout"]["trades"] == 4

    assert (tmp_path / "trades.csv").exists()
    assert (tmp_path / "signals.csv").exists()
    trades_csv = (tmp_path / "trades.csv").read_text().strip().splitlines()
    assert len(trades_csv) == 5  # header + 4 trades

    assert log.equity_curve()[-1]["equity"] == pytest.approx(100_300)
    log.close()


def test_stats_empty(tmp_path):
    log = TradeLog(tmp_path / "t.db")
    stats = log.stats()
    assert stats["total_trades"] == 0
    assert stats["win_rate"] == 0.0
    log.close()
