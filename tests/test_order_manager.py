from datetime import date, timedelta

import pytest

from trading_engine.config import ExecutionSettings, RiskSettings
from trading_engine.execution.order_manager import OrderManager
from trading_engine.execution.paper import PaperBroker
from trading_engine.models import (Direction, Instrument, OptionContract, OptionType,
                                   OrderSide, OrderStatus, OrderType, Signal)


def equity_signal(entry=100.0, stop=98.0, target=104.0, direction=Direction.LONG):
    return Signal(symbol="TEST", strategy="momentum_breakout", direction=direction,
                  entry_price=entry, stop_loss=stop, target_price=target,
                  instrument=Instrument.EQUITY)


def option_signal(entry=100.0, stop=99.0, target=105.0):
    contract = OptionContract(underlying="TEST", option_type=OptionType.CALL,
                              strike=105.0, expiry=date(2026, 7, 17),
                              bid=1.9, ask=2.1, volume=5000, open_interest=1000, iv=0.9)
    return Signal(symbol="TEST", strategy="options_flow", direction=Direction.LONG,
                  entry_price=entry, stop_loss=stop, target_price=target,
                  instrument=Instrument.OPTION, contract=contract)


def make_manager(broker=None, **kwargs):
    broker = broker or PaperBroker(slippage_bps=0)
    closed = []
    om = OrderManager(broker, ExecutionSettings(), RiskSettings(),
                      on_trade_closed=closed.append, **kwargs)
    return om, broker, closed


class TestEquityFlow:
    def test_entry_fill_places_protective_stop(self):
        om, broker, _ = make_manager()
        broker.set_mark("TEST", 100.0)
        order = om.execute_signal(equity_signal(), qty=100)
        assert order.status is OrderStatus.FILLED
        positions = om.open_positions()
        assert len(positions) == 1 and positions[0].qty == 100
        stops = [o for o in om.open_orders() if o.note == "stop_loss"]
        assert len(stops) == 1
        assert stops[0].order_type is OrderType.STOP
        assert stops[0].qty == 100
        assert stops[0].stop_price == pytest.approx(98.0)
        assert stops[0].side is OrderSide.SELL

    def test_broker_stop_fill_closes_trade(self):
        om, broker, closed = make_manager()
        broker.set_mark("TEST", 100.0)
        om.execute_signal(equity_signal(), qty=100)
        broker.set_mark("TEST", 97.5)   # through the stop
        broker.process()                # broker stop order fills
        om.poll()
        assert om.open_positions() == []
        assert len(closed) == 1
        record = closed[0]
        assert record.exit_reason == "stop_loss"
        assert record.pnl == pytest.approx((97.5 - 100.0) * 100)

    def test_engine_side_target_close(self):
        om, broker, closed = make_manager()
        broker.set_mark("TEST", 100.0)
        om.execute_signal(equity_signal(), qty=50)
        broker.set_mark("TEST", 104.5)  # engine refreshes marks before manage()
        om.manage({"TEST": 104.5})      # above target
        om.poll()
        assert len(closed) == 1
        assert closed[0].exit_reason == "target"
        assert closed[0].pnl == pytest.approx((104.5 - 100.0) * 50)

    def test_short_thesis_uses_sell_short_and_buy_stop(self):
        om, broker, closed = make_manager()
        broker.set_mark("TEST", 100.0)
        order = om.execute_signal(
            equity_signal(entry=100.0, stop=102.0, target=96.0,
                          direction=Direction.SHORT), qty=10)
        assert order.side is OrderSide.SELL_SHORT
        stops = [o for o in om.open_orders() if o.note == "stop_loss"]
        assert stops[0].side is OrderSide.BUY_TO_COVER
        broker.set_mark("TEST", 95.5)
        om.manage({"TEST": 95.5})       # target hit for a short
        assert closed and closed[0].pnl == pytest.approx((100.0 - 95.5) * 10)

    def test_partial_fills_resize_stop(self):
        broker = PaperBroker(partial_fill_qty=40, slippage_bps=0)
        om, broker, _ = make_manager(broker)
        broker.set_mark("TEST", 100.0)
        om.execute_signal(equity_signal(), qty=100)
        pos = om.open_positions()[0]
        assert pos.qty == 40
        stops = [o for o in om.open_orders() if o.note == "stop_loss"]
        assert stops and stops[0].qty == 40
        broker.process()   # +40 = 80
        om.poll()
        broker.process()   # +20 = 100
        om.poll()
        pos = om.open_positions()[0]
        assert pos.qty == 100
        live_stops = [o for o in om.open_orders()
                      if o.note == "stop_loss" and not o.status.is_terminal]
        assert len(live_stops) == 1
        assert live_stops[0].qty == 100

    def test_flatten_all(self):
        om, broker, closed = make_manager()
        broker.set_mark("TEST", 100.0)
        broker.set_mark("OTHER", 50.0)
        om.execute_signal(equity_signal(), qty=10)
        sig2 = equity_signal(entry=50.0, stop=49.0, target=52.0)
        sig2.symbol = "OTHER"
        om.execute_signal(sig2, qty=20)
        assert len(om.open_positions()) == 2
        om.flatten_all("eod_flatten")
        om.poll()
        assert om.open_positions() == []
        assert {r.exit_reason for r in closed} == {"eod_flatten"}

    def test_has_open_exposure(self):
        om, broker, _ = make_manager()
        broker.set_mark("TEST", 100.0)
        assert not om.has_open_exposure("TEST")
        om.execute_signal(equity_signal(), qty=10)
        assert om.has_open_exposure("TEST")


class TestOptionFlow:
    def test_option_entry_and_premium_stop(self):
        om, broker, closed = make_manager()
        sig = option_signal()
        occ = sig.contract.occ
        broker.set_mark("TEST", 100.0)
        broker.set_mark(occ, 2.0)
        order = om.execute_signal(sig, qty=5, option_premium=2.0)
        assert order.side is OrderSide.BUY_TO_OPEN
        assert order.multiplier == 100
        pos = om.open_positions()[0]
        assert pos.premium_stop == pytest.approx(1.0)  # 50% premium stop
        # equity stops are not placed for options; engine-side management only
        assert [o for o in om.open_orders() if o.note == "stop_loss"] == []
        # premium collapses -> close
        broker.set_mark(occ, 0.9)
        om.manage({"TEST": 99.5}, {occ: 0.9})
        om.poll()
        assert closed and closed[0].exit_reason == "premium_stop"
        assert closed[0].pnl == pytest.approx((0.9 - 2.0) * 5 * 100)

    def test_option_underlying_stop(self):
        om, broker, closed = make_manager()
        sig = option_signal(stop=99.0)
        occ = sig.contract.occ
        broker.set_mark("TEST", 100.0)
        broker.set_mark(occ, 2.0)
        om.execute_signal(sig, qty=2, option_premium=2.0)
        broker.set_mark(occ, 1.6)
        om.manage({"TEST": 98.8}, {occ: 1.6})  # underlying through stop
        assert closed and closed[0].exit_reason == "stop_loss"

    def test_option_target_close_profit(self):
        om, broker, closed = make_manager()
        sig = option_signal(target=105.0)
        occ = sig.contract.occ
        broker.set_mark("TEST", 100.0)
        broker.set_mark(occ, 2.0)
        om.execute_signal(sig, qty=1, option_premium=2.0)
        broker.set_mark(occ, 4.5)
        om.manage({"TEST": 105.5}, {occ: 4.5})
        assert closed and closed[0].exit_reason == "target"
        assert closed[0].pnl == pytest.approx((4.5 - 2.0) * 100)
