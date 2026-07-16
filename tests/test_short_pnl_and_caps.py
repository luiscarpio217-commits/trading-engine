"""Regressions from the first live paper session.

Covers: P&L signs for bearish (SHORT-thesis) positions — bought puts and
short equity — single-source P&L agreement between order manager, journal,
risk manager and status; equity == cash + positions; halt-reason coherence;
and sizing caps that stop tiny premiums from producing enormous quantities
(the 177-contract $0.06 0DTE incident).
"""

from datetime import date

import pytest

from tests.conftest import make_breakout_df, make_chain
from tests.test_engine import TUESDAY_1030_ET, FakeOptionsData, make_engine
from trading_engine.config import ExecutionSettings, OptionsFlowSettings, RiskSettings
from trading_engine.execution.order_manager import OrderManager
from trading_engine.execution.paper import PaperBroker
from trading_engine.models import (Direction, Instrument, OptionContract, OptionType,
                                   OrderSide, Signal, position_side_label)
from trading_engine.risk.position_sizing import PositionSizer
from trading_engine.strategies import MarketSnapshot, OptionsFlowStrategy


def put_signal(entry=100.0, stop=101.0, target=95.0):
    """Bearish thesis expressed as a bought put (how the engine trades it)."""
    contract = OptionContract(underlying="TEST", option_type=OptionType.PUT,
                              strike=97.0, expiry=date(2026, 7, 17),
                              bid=1.9, ask=2.1, volume=5000, open_interest=1000, iv=0.9)
    return Signal(symbol="TEST", strategy="options_flow", direction=Direction.SHORT,
                  entry_price=entry, stop_loss=stop, target_price=target,
                  instrument=Instrument.OPTION, contract=contract)


def make_om():
    broker = PaperBroker(slippage_bps=0)
    closed = []
    om = OrderManager(broker, ExecutionSettings(), RiskSettings(),
                      on_trade_closed=closed.append)
    return om, broker, closed


class TestShortThesisPnlSigns:
    def test_bought_put_premium_rise_is_profit(self):
        """direction=short + instrument=option means LONG PUTS: premium up = gain."""
        om, broker, closed = make_om()
        sig = put_signal()
        occ = sig.contract.occ
        broker.set_mark("TEST", 100.0)
        broker.set_mark(occ, 2.0)
        order = om.execute_signal(sig, qty=3, option_premium=2.0)
        assert order.side is OrderSide.BUY_TO_OPEN     # bought, never sold short
        broker.set_mark(occ, 3.5)
        om.manage({"TEST": 94.5}, {occ: 3.5})          # underlying through target
        assert closed and closed[0].exit_reason == "target"
        assert closed[0].pnl == pytest.approx((3.5 - 2.0) * 3 * 100)  # +450 profit

    def test_bought_put_premium_collapse_is_loss(self):
        om, broker, closed = make_om()
        sig = put_signal()
        occ = sig.contract.occ
        broker.set_mark("TEST", 100.0)
        broker.set_mark(occ, 2.0)
        om.execute_signal(sig, qty=3, option_premium=2.0)
        broker.set_mark(occ, 0.8)
        om.manage({"TEST": 100.5}, {occ: 0.8})         # premium stop (<= 1.0)
        assert closed and closed[0].exit_reason == "premium_stop"
        assert closed[0].pnl == pytest.approx((0.8 - 2.0) * 3 * 100)  # -360 loss

    def test_short_equity_signs_both_ways(self):
        for exit_mark in (90.0, 110.0):  # short 10 @ 100: 90 -> +100, 110 -> -100
            om, broker, closed = make_om()
            broker.set_mark("TEST", 100.0)
            sig = Signal(symbol="TEST", strategy="momentum_breakout",
                         direction=Direction.SHORT, entry_price=100.0,
                         stop_loss=104.0, target_price=90.0,
                         instrument=Instrument.EQUITY)
            om.execute_signal(sig, qty=10)
            broker.set_mark("TEST", exit_mark)
            om.manage({"TEST": exit_mark})
            om.poll()
            assert closed, f"no close at mark {exit_mark}"
            assert closed[0].pnl == pytest.approx((100.0 - exit_mark) * 10)

    def test_engine_never_shorts_options(self):
        """No sell_to_open path exists: every option order is a buy or a close."""
        om, broker, _ = make_om()
        sig = put_signal()
        broker.set_mark("TEST", 100.0)
        broker.set_mark(sig.contract.occ, 2.0)
        om.execute_signal(sig, qty=2, option_premium=2.0)
        om.close_position(om.open_positions()[0], "manual")
        sides = {o.side for o in om._orders.values()
                 if o.instrument is Instrument.OPTION}
        assert sides <= {OrderSide.BUY_TO_OPEN, OrderSide.SELL_TO_CLOSE}

    def test_side_labels_do_not_say_short_for_options(self):
        assert position_side_label("option", "short") == "long put"
        assert position_side_label("option", "long") == "long call"
        assert position_side_label("equity", "short") == "short"
        assert position_side_label(Instrument.OPTION, Direction.SHORT) == "long put"


class TestSingleSourceOfTruth:
    def test_journal_risk_and_status_agree_after_stop_out(self, tmp_path):
        engine, market, broker = make_engine(tmp_path)
        [sig] = engine.scan_once(now=TUESDAY_1030_ET)
        market.latest = sig.stop_loss - 1.0
        engine.manage_once(now=TUESDAY_1030_ET)

        trades = engine.trade_log.recent_trades(10)
        journal_pnl = sum(t["pnl"] for t in trades)
        status = engine.get_status()
        assert journal_pnl < 0
        assert engine.risk.realized_pnl_today() == pytest.approx(journal_pnl)
        assert status["day"]["realized_pnl"] == pytest.approx(round(journal_pnl, 2))

    def test_equity_equals_cash_plus_positions(self, tmp_path):
        engine, market, broker = make_engine(tmp_path)
        engine.scan_once(now=TUESDAY_1030_ET)
        account = broker.get_account()
        positions_value = sum(p.market_value for p in broker.list_positions())
        assert account.equity == pytest.approx(account.cash + positions_value)
        # and day total P&L is derived from that same equity
        status = engine.get_status()
        assert status["day"]["total_pnl"] == pytest.approx(
            round(account.equity - status["day"]["start_equity"], 2), abs=0.01)

    def test_halt_reason_is_a_timestamped_snapshot(self, tmp_path):
        engine, market, broker = make_engine(tmp_path)
        engine._ensure_session(TUESDAY_1030_ET)
        engine.risk.on_trade_closed(-5000.0)
        assert engine.risk.halted
        frozen = engine.risk.halt_reason
        assert "daily loss limit" in frozen and "tripped" in frozen
        # realized keeps accruing after the halt (shutoff flatten closes trades),
        # but the trigger snapshot must not silently change underneath it
        engine.risk.on_trade_closed(+40_000.0)
        assert engine.risk.realized_pnl_today() == pytest.approx(35_000.0)
        assert engine.risk.halt_reason == frozen
        status = engine.get_status()
        assert status["halted"] is True
        assert status["halt_reason"] == frozen


class TestSizingCaps:
    def test_min_premium_blocks_lottery_tickets(self):
        """The 177-contract $0.06 incident must size to zero."""
        sizer = PositionSizer(RiskSettings())
        sig = put_signal()
        result = sizer.size(sig, equity=106_000, option_premium=0.06)
        assert not result.viable
        assert any("below minimum" in n for n in result.notes)

    def test_max_contracts_cap(self):
        sizer = PositionSizer(RiskSettings(max_position_notional_pct=1.0))
        sig = put_signal()
        # $10k budget at $15/contract would be 666 contracts -> capped at 25
        result = sizer.size(sig, equity=1_000_000, option_premium=0.15)
        assert result.qty == 25
        assert any("max_option_contracts" in n for n in result.notes)

    def test_engine_falls_back_to_equity_on_cheap_contract(self, tmp_path):
        spot = float(make_breakout_df()["close"].iloc[-1])
        chain = make_chain(spot=spot, asof=date(2026, 7, 14))
        frame = chain.contracts
        # make the contract the strategy will pick (0.42 delta unusual call) cheap
        cheap = (frame["volume"] == 5000)
        frame.loc[cheap, ["bid", "ask", "last", "mid"]] = [0.05, 0.07, 0.06, 0.06]
        engine, market, broker = make_engine(tmp_path, trade_options=True, chain=chain)
        [sig] = engine.scan_once(now=TUESDAY_1030_ET)
        positions = engine.orders.open_positions()
        assert len(positions) == 1
        assert positions[0].instrument is Instrument.EQUITY  # not 177 lottery tickets

    def test_options_flow_ignores_sub_dime_contracts(self):
        chain = make_chain(spot=100.0)
        frame = chain.contracts
        cheap = (frame["volume"] == 5000)
        frame.loc[cheap, ["bid", "ask", "last", "mid"]] = [0.05, 0.07, 0.06, 0.06]
        strat = OptionsFlowStrategy(OptionsFlowSettings())
        snap = MarketSnapshot(symbol="TEST", intraday=make_breakout_df(), chain=chain)
        assert strat.evaluate(snap) is None
