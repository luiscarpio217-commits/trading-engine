"""Premium-based profit protection: take-profit and trailing lock.

Both operate on option positions in the order manager's engine-side
monitoring, alongside the existing stop / target / premium-stop checks.
"""

from datetime import date

import pytest

from tests.test_engine import TUESDAY_1030_ET, FakeOptionsData, make_engine
from tests.conftest import make_breakout_df, make_chain
from trading_engine.config import (Config, ExecutionSettings,
                                   ProfitProtectionSettings, RiskSettings,
                                   TrailingSettings)
from trading_engine.execution.order_manager import OrderManager
from trading_engine.execution.paper import PaperBroker
from trading_engine.models import (Direction, Instrument, OptionContract,
                                   OptionType, Signal)


def call_signal(entry=100.0, stop=99.0, target=110.0):
    contract = OptionContract(underlying="TEST", option_type=OptionType.CALL,
                              strike=105.0, expiry=date(2026, 7, 24),
                              bid=1.9, ask=2.1, volume=5000,
                              open_interest=1000, iv=0.6)
    return Signal(symbol="TEST", strategy="momentum_breakout",
                  direction=Direction.LONG, entry_price=entry, stop_loss=stop,
                  target_price=target, instrument=Instrument.OPTION,
                  contract=contract)


def pp(take_profit=None, arm=None, giveback=None) -> ProfitProtectionSettings:
    return ProfitProtectionSettings(
        take_profit_pct=take_profit,
        trailing=TrailingSettings(arm_pct=arm, giveback_pct=giveback))


def make_om_with_option(protection, entry_premium=2.0, qty=3):
    broker = PaperBroker(slippage_bps=0)
    closed = []
    om = OrderManager(broker, ExecutionSettings(), RiskSettings(),
                      on_trade_closed=closed.append)
    sig = call_signal()
    occ = sig.contract.occ
    broker.set_mark("TEST", 100.0)
    broker.set_mark(occ, entry_premium)
    om.execute_signal(sig, qty=qty, option_premium=entry_premium,
                      profit_protection=protection)
    return om, broker, closed, occ


def tick(om, broker, occ, premium, underlying=100.0):
    """One manage cycle at the given marks (underlying far from stop/target)."""
    broker.set_mark(occ, premium)
    om.manage({"TEST": underlying}, {occ: premium})
    om.poll()


class TestTakeProfit:
    def test_triggers_at_threshold(self):
        om, broker, closed, occ = make_om_with_option(pp(take_profit=75))
        tick(om, broker, occ, 3.49)          # +74.5% — just below
        assert closed == []
        tick(om, broker, occ, 3.50)          # exactly +75%
        assert len(closed) == 1
        assert closed[0].exit_reason == "take_profit"
        assert closed[0].pnl == pytest.approx((3.5 - 2.0) * 3 * 100)

    def test_never_fires_when_disabled(self):
        om, broker, closed, occ = make_om_with_option(pp())  # everything off
        for premium in (3.0, 5.0, 9.0):
            tick(om, broker, occ, premium)
        assert closed == []                   # only target/stop rules apply

    def test_no_fire_below_threshold(self):
        om, broker, closed, occ = make_om_with_option(pp(take_profit=200))
        for premium in (2.5, 3.5, 5.0, 5.9):
            tick(om, broker, occ, premium)
        assert closed == []


class TestTrailingLock:
    def test_arms_tracks_peak_and_fires_on_retrace(self):
        om, broker, closed, occ = make_om_with_option(pp(arm=50, giveback=30))
        tick(om, broker, occ, 2.5)            # +25%: not armed yet
        pos = om.open_positions()[0]
        assert pos.peak_mark == 0.0
        tick(om, broker, occ, 3.0)            # +50%: arms at 3.0
        assert pos.peak_mark == pytest.approx(3.0)
        tick(om, broker, occ, 4.0)            # new high: peak ratchets
        assert pos.peak_mark == pytest.approx(4.0)
        tick(om, broker, occ, 3.2)            # -20% off peak: holds
        assert closed == []
        assert pos.peak_mark == pytest.approx(4.0)   # persists across cycles
        tick(om, broker, occ, 2.8)            # -30% off peak: locks in
        assert len(closed) == 1
        assert closed[0].exit_reason == "trailing_lock"
        assert closed[0].pnl == pytest.approx((2.8 - 2.0) * 3 * 100)  # still a win

    def test_peak_only_ratchets_up(self):
        om, broker, closed, occ = make_om_with_option(pp(arm=50, giveback=50))
        tick(om, broker, occ, 3.0)            # armed, peak 3.0
        tick(om, broker, occ, 2.0)            # -33% off peak < 50%: holds
        pos = om.open_positions()[0]
        assert pos.peak_mark == pytest.approx(3.0)  # did not follow mark down
        assert closed == []

    def test_not_armed_below_arm_threshold(self):
        om, broker, closed, occ = make_om_with_option(pp(arm=100, giveback=10))
        tick(om, broker, occ, 3.9)            # +95%: below arm
        tick(om, broker, occ, 2.1)            # huge retrace, but never armed
        assert closed == []
        assert om.open_positions()[0].peak_mark == 0.0

    def test_partial_trailing_config_is_inert(self):
        # arm without giveback (or vice versa) must not enable anything
        om, broker, closed, occ = make_om_with_option(pp(arm=50))
        tick(om, broker, occ, 4.0)
        tick(om, broker, occ, 2.1)
        assert closed == []
        assert om.open_positions()[0].trail_arm_pct is None


class TestInteraction:
    def test_take_profit_wins_when_hit_first(self):
        om, broker, closed, occ = make_om_with_option(pp(take_profit=75, arm=50, giveback=30))
        tick(om, broker, occ, 3.0)            # trailing arms
        tick(om, broker, occ, 3.5)            # +75%: take-profit level
        assert len(closed) == 1
        assert closed[0].exit_reason == "take_profit"

    def test_trailing_wins_when_take_profit_never_reached(self):
        om, broker, closed, occ = make_om_with_option(pp(take_profit=200, arm=50, giveback=30))
        tick(om, broker, occ, 3.4)            # +70%: armed, below +200% TP
        tick(om, broker, occ, 2.3)            # -32% off 3.4 peak
        assert len(closed) == 1
        assert closed[0].exit_reason == "trailing_lock"

    def test_protective_stop_beats_profit_exits(self):
        """Underlying stop still rules even while trailing is armed."""
        om, broker, closed, occ = make_om_with_option(pp(arm=50, giveback=30))
        tick(om, broker, occ, 4.0)            # armed, peak 4.0
        # underlying gaps through the stop while the mark also retraces hard
        tick(om, broker, occ, 2.0, underlying=98.5)
        assert len(closed) == 1
        assert closed[0].exit_reason == "stop_loss"

    def test_premium_stop_unaffected(self):
        om, broker, closed, occ = make_om_with_option(pp(take_profit=75))
        tick(om, broker, occ, 0.9)            # <= 50% premium stop
        assert closed and closed[0].exit_reason == "premium_stop"


class TestConfigAndWiring:
    def test_yaml_round_trip(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text(
            "strategies:\n"
            "  momentum_breakout:\n"
            "    profit_protection:\n"
            "      take_profit_pct: 75\n"
            "      trailing:\n"
            "        arm_pct: 50\n"
            "        giveback_pct: 30\n")
        cfg = Config.load(p)
        mom = cfg.strategies.momentum_breakout.profit_protection
        assert mom.take_profit_pct == 75
        assert mom.trailing.enabled and mom.trailing.giveback_pct == 30
        # other strategy untouched, defaults stay off
        flow = cfg.strategies.options_flow.profit_protection
        assert flow.take_profit_pct is None and not flow.trailing.enabled

    def test_defaults_are_off(self):
        cfg = Config()
        for s in (cfg.strategies.momentum_breakout, cfg.strategies.options_flow):
            assert s.profit_protection.take_profit_pct is None
            assert not s.profit_protection.trailing.enabled

    def test_engine_passes_strategy_settings_to_position(self, tmp_path):
        spot = float(make_breakout_df()["close"].iloc[-1])
        chain = make_chain(spot=spot, asof=date(2026, 7, 14))

        engine, market, broker = make_engine(tmp_path, trade_options=True, chain=chain)
        engine.config.strategies.momentum_breakout.profit_protection = pp(
            take_profit=75, arm=50, giveback=30)
        [sig] = engine.scan_once(now=TUESDAY_1030_ET)
        pos = engine.orders.open_positions()[0]
        assert pos.instrument is Instrument.OPTION
        assert pos.take_profit_pct == 75
        assert pos.trail_arm_pct == 50
        assert pos.trail_giveback_pct == 30
        assert pos.peak_mark == 0.0
