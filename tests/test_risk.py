from datetime import date

import pytest

from trading_engine.config import RiskSettings
from trading_engine.models import Direction, Instrument, Signal
from trading_engine.risk.manager import RiskManager
from trading_engine.risk.position_sizing import PositionSizer, kelly_fraction


def equity_signal(entry=100.0, stop=98.0, target=104.0):
    return Signal(symbol="TEST", strategy="momentum_breakout", direction=Direction.LONG,
                  entry_price=entry, stop_loss=stop, target_price=target,
                  instrument=Instrument.EQUITY)


class TestPositionSizing:
    def test_fixed_fractional_with_notional_cap(self):
        sizer = PositionSizer(RiskSettings())  # 1% risk, 20% notional cap
        result = sizer.size(equity_signal(), equity=100_000)
        # risk budget $1000 / $2 per-share risk = 500 shares,
        # but notional cap $20k / $100 = 200 shares binds first
        assert result.qty == 200
        assert result.risk_dollars == pytest.approx(1000.0)
        assert result.method == "fixed_fractional"

    def test_uncapped_when_notional_allows(self):
        s = RiskSettings(max_position_notional_pct=1.0)
        result = PositionSizer(s).size(equity_signal(), equity=100_000)
        assert result.qty == 500

    def test_option_contract_sizing(self):
        sizer = PositionSizer(RiskSettings(max_position_notional_pct=1.0))
        sig = equity_signal()
        sig.instrument = Instrument.OPTION
        result = sizer.size(sig, equity=100_000, option_premium=2.0)
        # $1000 budget / ($2.00 * 100) full-premium risk = 5 contracts
        assert result.qty == 5
        assert result.notional == pytest.approx(1000.0)

    def test_option_without_premium_not_viable(self):
        sig = equity_signal()
        sig.instrument = Instrument.OPTION
        result = PositionSizer(RiskSettings()).size(sig, equity=100_000)
        assert not result.viable

    def test_kelly_fraction_formula(self):
        # W=0.6, R=1.5 -> f* = 0.6 - 0.4/1.5 = 1/3
        assert kelly_fraction(0.6, 300.0, 200.0) == pytest.approx(1 / 3)
        assert kelly_fraction(0.3, 100.0, 200.0) == 0.0  # negative edge -> 0
        assert kelly_fraction(1.0, 100.0, 100.0) == 0.0  # degenerate input

    def test_kelly_budget_with_history(self):
        s = RiskSettings(sizing_method="kelly", kelly_min_trades=10)
        sizer = PositionSizer(s)
        stats = {"total_trades": 40, "win_rate": 0.6, "avg_win": 300.0, "avg_loss": 200.0}
        dollars, pct, method = sizer.risk_budget(100_000, stats)
        assert method == "kelly"
        assert pct == pytest.approx((1 / 3) * 0.5)  # half-Kelly
        assert dollars == pytest.approx(100_000 * (1 / 3) * 0.5)

    def test_kelly_falls_back_without_history(self):
        s = RiskSettings(sizing_method="kelly")
        dollars, pct, method = PositionSizer(s).risk_budget(
            100_000, {"total_trades": 3, "win_rate": 1.0, "avg_win": 100, "avg_loss": 0})
        assert method == "fixed_fractional"
        assert pct == pytest.approx(0.01)

    def test_kelly_cap(self):
        s = RiskSettings(sizing_method="kelly", kelly_min_trades=1,
                         kelly_multiplier=1.0, kelly_cap=0.05)
        stats = {"total_trades": 50, "win_rate": 0.9, "avg_win": 500.0, "avg_loss": 100.0}
        _, pct, method = PositionSizer(s).risk_budget(100_000, stats)
        assert method == "kelly"
        assert pct == pytest.approx(0.05)  # capped


class TestRiskManager:
    def test_daily_loss_shutoff_on_realized(self):
        rm = RiskManager(RiskSettings())  # 3% daily loss limit
        rm.start_day(date(2026, 7, 14), 100_000)
        rm.on_trade_closed(-2000)
        assert not rm.halted
        rm.on_trade_closed(-1500)  # cumulative -3500 <= -3000
        assert rm.halted
        ok, reason = rm.can_open(0)
        assert not ok and "halted" in reason

    def test_daily_loss_shutoff_on_equity_mark(self):
        rm = RiskManager(RiskSettings())
        rm.start_day(date(2026, 7, 14), 100_000)
        rm.mark_equity(97_500)
        assert not rm.halted
        rm.mark_equity(96_900)  # drawdown 3100 > 3000
        assert rm.halted
        assert rm.should_flatten_on_halt

    def test_max_open_positions_gate(self):
        rm = RiskManager(RiskSettings(max_open_positions=2))
        rm.start_day(date(2026, 7, 14), 100_000)
        assert rm.can_open(1)[0]
        ok, reason = rm.can_open(2)
        assert not ok and "max open positions" in reason

    def test_new_session_resets_halt(self):
        rm = RiskManager(RiskSettings())
        rm.start_day(date(2026, 7, 14), 100_000)
        rm.on_trade_closed(-5000)
        assert rm.halted
        rm.start_day(date(2026, 7, 15), 95_000)
        assert not rm.halted
        assert rm.can_open(0)[0]

    def test_same_day_start_is_idempotent(self):
        rm = RiskManager(RiskSettings())
        rm.start_day(date(2026, 7, 14), 100_000)
        rm.on_trade_closed(-1000)
        rm.start_day(date(2026, 7, 14), 99_000)  # no-op
        assert rm.realized_pnl_today() == pytest.approx(-1000)
