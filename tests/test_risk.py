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
        # $1000 budget / ($2.00 * 100 * 50% premium stop) = 10 contracts;
        # worst engine-managed loss 10 * $100 = $1000 == the budget
        assert result.qty == 10
        assert result.notional == pytest.approx(2000.0)

    def test_option_without_premium_not_viable(self):
        sig = equity_signal()
        sig.instrument = Instrument.OPTION
        result = PositionSizer(RiskSettings()).size(sig, equity=100_000)
        assert not result.viable
        assert result.reason == "no_premium"

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


def option_put_signal(entry: float, stop: float) -> Signal:
    sig = Signal(symbol="X", strategy="options_flow", direction=Direction.SHORT,
                 entry_price=entry, stop_loss=stop,
                 target_price=entry - 2 * (stop - entry))
    sig.instrument = Instrument.OPTION
    return sig


class TestDeployedSizedZeroRegression:
    """Live report: every option signal died 'sized_zero' on defaults.

    Cause: risk-per-contract used the FULL premium, so any contract over
    $10 premium exceeded the whole 1% budget on $100k and floor()'d to 0 —
    e.g. QQQ put entry 706.66/stop 713.73 with a ~$12 near-the-money
    premium. Sizing now budgets the loss at the engine-enforced premium
    stop (default 50%), so normal contracts size small-but-nonzero while
    every cap stays in force.
    """

    def test_qqq_put_reported_numbers(self):
        sig = option_put_signal(706.66, 713.73)
        result = PositionSizer(RiskSettings()).size(sig, 100_000, option_premium=12.0)
        assert result.viable
        assert result.qty == 1          # floor(1000 / (12 * 100 * 0.5))
        # engine-managed worst case stays inside the budget
        assert result.qty * 12.0 * 100 * 0.5 <= result.risk_dollars

    def test_aapl_put_reported_numbers(self):
        sig = option_put_signal(332.30, 335.62)
        result = PositionSizer(RiskSettings()).size(sig, 100_000, option_premium=11.0)
        assert result.viable and result.qty == 1

    def test_moderate_premium_sizes_more(self):
        sig = option_put_signal(706.66, 713.73)
        result = PositionSizer(RiskSettings()).size(sig, 100_000, option_premium=4.0)
        assert result.qty == 5          # floor(1000 / 200)

    def test_at_stop_risk_never_exceeds_budget(self):
        sizer = PositionSizer(RiskSettings())
        sig = option_put_signal(706.66, 713.73)
        for premium in (0.10, 0.5, 1.0, 2.5, 5.0, 8.0, 12.0, 19.0):
            result = sizer.size(sig, 100_000, option_premium=premium)
            assert result.viable, f"premium {premium} should size nonzero"
            assert result.qty * premium * 100 * 0.5 <= result.risk_dollars + 1e-9

    def test_reason_risk_budget_lt_one_contract(self):
        """A genuinely over-budget contract still zeroes - with the cause."""
        sig = option_put_signal(706.66, 713.73)
        result = PositionSizer(RiskSettings()).size(sig, 100_000, option_premium=25.0)
        assert not result.viable
        assert result.reason == "risk_budget_lt_one_contract"

    def test_reason_premium_below_min(self):
        sig = option_put_signal(706.66, 713.73)
        result = PositionSizer(RiskSettings()).size(sig, 100_000, option_premium=0.06)
        assert result.reason == "premium_below_min"

    def test_reason_notional_cap_lt_one_contract(self):
        sig = option_put_signal(706.66, 713.73)
        settings = RiskSettings(max_position_notional_pct=0.001)  # $100 cap
        result = PositionSizer(settings).size(sig, 100_000, option_premium=5.0)
        assert result.reason == "notional_cap_lt_one_contract"

    def test_caps_still_enforced(self):
        # huge budget + cheap-ish premium -> hard contract cap still binds
        settings = RiskSettings(max_position_notional_pct=1.0)
        sig = option_put_signal(706.66, 713.73)
        result = PositionSizer(settings).size(sig, 1_000_000, option_premium=0.15)
        assert result.qty == settings.max_option_contracts

    def test_equity_reason_codes(self):
        sizer = PositionSizer(RiskSettings())
        too_expensive = equity_signal(entry=5000.0, stop=1.0)
        assert sizer.size(too_expensive, 100_000).reason == "risk_budget_lt_one_share"
        bad_stop = equity_signal(entry=100.0, stop=100.0)
        assert sizer.size(bad_stop, 100_000).reason == "invalid_stop"


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
