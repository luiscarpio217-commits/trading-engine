import pytest

from trading_engine.execution.paper import PaperBroker
from trading_engine.models import Instrument, Order, OrderSide, OrderStatus, OrderType


def market_buy(symbol="TEST", qty=100):
    return Order(symbol=symbol, side=OrderSide.BUY, qty=qty)


class TestPaperBroker:
    def test_market_fill_with_slippage(self):
        b = PaperBroker(starting_cash=100_000, slippage_bps=5)
        b.set_mark("TEST", 50.0)
        order = b.submit_order(market_buy())
        assert order.status is OrderStatus.FILLED
        assert order.avg_fill_price == pytest.approx(50.025)  # buys pay up
        account = b.get_account()
        assert account.cash == pytest.approx(100_000 - 100 * 50.025)
        positions = b.list_positions()
        assert len(positions) == 1 and positions[0].qty == 100

    def test_market_order_waits_for_mark(self):
        b = PaperBroker()
        order = b.submit_order(market_buy())
        assert order.status is OrderStatus.NEW  # no mark yet
        b.set_mark("TEST", 20.0)
        b.process()
        assert order.status is OrderStatus.FILLED

    def test_limit_order_fills_on_cross(self):
        b = PaperBroker(slippage_bps=0)
        b.set_mark("TEST", 50.0)
        order = Order(symbol="TEST", side=OrderSide.BUY, qty=10,
                      order_type=OrderType.LIMIT, limit_price=49.0)
        b.submit_order(order)
        assert order.status is OrderStatus.NEW
        b.set_mark("TEST", 48.9)
        b.process()
        assert order.status is OrderStatus.FILLED
        assert order.avg_fill_price == pytest.approx(48.9)

    def test_stop_order_triggers(self):
        b = PaperBroker(slippage_bps=0)
        b.set_mark("TEST", 50.0)
        b.submit_order(market_buy(qty=10))
        stop = Order(symbol="TEST", side=OrderSide.SELL, qty=10,
                     order_type=OrderType.STOP, stop_price=45.0)
        b.submit_order(stop)
        assert stop.status is OrderStatus.NEW
        b.set_mark("TEST", 44.0)
        b.process()
        assert stop.status is OrderStatus.FILLED
        assert stop.avg_fill_price == pytest.approx(44.0)
        assert b.list_positions() == []  # flat

    def test_partial_fills(self):
        b = PaperBroker(partial_fill_qty=30, slippage_bps=0)
        b.set_mark("TEST", 10.0)
        order = b.submit_order(market_buy(qty=100))
        assert order.status is OrderStatus.PARTIALLY_FILLED
        assert order.filled_qty == 30
        b.process()
        b.process()
        assert order.filled_qty == 90
        b.process()
        assert order.status is OrderStatus.FILLED
        assert order.filled_qty == 100

    def test_option_multiplier_cash_impact(self):
        b = PaperBroker(starting_cash=10_000, slippage_bps=0)
        b.set_mark("TEST260717C00105000", 2.0)
        order = Order(symbol="TEST260717C00105000", side=OrderSide.BUY_TO_OPEN,
                      qty=3, instrument=Instrument.OPTION, multiplier=100,
                      underlying="TEST")
        b.submit_order(order)
        assert order.status is OrderStatus.FILLED
        assert b.get_account().cash == pytest.approx(10_000 - 3 * 2.0 * 100)

    def test_short_position_and_cover(self):
        b = PaperBroker(starting_cash=10_000, slippage_bps=0)
        b.set_mark("TEST", 100.0)
        b.submit_order(Order(symbol="TEST", side=OrderSide.SELL_SHORT, qty=10))
        assert b.list_positions()[0].qty == -10
        assert b.get_account().cash == pytest.approx(11_000)
        b.set_mark("TEST", 90.0)
        b.submit_order(Order(symbol="TEST", side=OrderSide.BUY_TO_COVER, qty=10))
        assert b.list_positions() == []
        assert b.get_account().cash == pytest.approx(11_000 - 900)  # +$100 profit
