from datetime import date

from trading_engine.config import MomentumSettings, OptionsFlowSettings
from trading_engine.data.indicators import compute_indicators
from trading_engine.models import Direction, Instrument, OptionType
from trading_engine.strategies import MarketSnapshot, MomentumBreakoutStrategy, OptionsFlowStrategy
from tests.conftest import make_breakout_df, make_chain


def snapshot_from(df, symbol="TEST", chain=None):
    return MarketSnapshot(symbol=symbol, intraday=compute_indicators(df), chain=chain)


class TestMomentumBreakout:
    def test_long_breakout_fires(self, breakout_df):
        strat = MomentumBreakoutStrategy(MomentumSettings())
        signal = strat.evaluate(snapshot_from(breakout_df))
        assert signal is not None
        assert signal.direction is Direction.LONG
        assert signal.option_type is OptionType.CALL
        assert signal.stop_loss < signal.entry_price < signal.target_price
        assert signal.reward_risk > 1.9
        assert any("breakout" in r for r in signal.reasons)
        assert any("RVOL" in r for r in signal.reasons)

    def test_no_signal_without_volume(self, breakout_df):
        df = breakout_df.copy()
        df.iloc[-1, df.columns.get_indexer(["volume"])[0]] = 100_000.0  # RVOL ~1
        strat = MomentumBreakoutStrategy(MomentumSettings())
        assert strat.evaluate(snapshot_from(df)) is None

    def test_no_signal_when_already_broken_out(self, breakout_df):
        """A bar consolidating above the old level (no new high) must not re-fire."""
        import pandas as pd

        df = breakout_df.copy()
        last = df.iloc[-1].copy()
        follow = last.copy()
        follow["open"] = last["close"]
        follow["close"] = last["close"] - 0.02   # holds gains, but under the
        follow["high"] = last["close"] + 0.01    # breakout bar's high
        follow["low"] = last["close"] - 0.10
        follow.name = df.index[-1] + (df.index[-1] - df.index[-2])
        df = pd.concat([df, follow.to_frame().T]).astype(float)
        strat = MomentumBreakoutStrategy(MomentumSettings())
        assert strat.evaluate(snapshot_from(df)) is None

    def test_contract_recommendation_from_chain(self, breakout_df):
        chain = make_chain(spot=float(breakout_df["close"].iloc[-1]))
        strat = MomentumBreakoutStrategy(MomentumSettings())
        signal = strat.evaluate(snapshot_from(breakout_df, chain=chain))
        assert signal is not None
        assert signal.instrument is Instrument.OPTION
        assert signal.contract is not None
        assert signal.contract.option_type is OptionType.CALL
        assert "DTE" in signal.expiry_recommendation

    def test_insufficient_data(self):
        df = make_breakout_df().tail(10)
        strat = MomentumBreakoutStrategy(MomentumSettings())
        assert strat.evaluate(snapshot_from(df)) is None


class TestOptionsFlow:
    def test_call_flow_long_signal(self, call_flow_chain):
        strat = OptionsFlowStrategy(OptionsFlowSettings())
        snap = MarketSnapshot(symbol="TEST", intraday=make_breakout_df(),
                              chain=call_flow_chain)
        signal = strat.evaluate(snap)
        assert signal is not None
        assert signal.direction is Direction.LONG
        assert signal.instrument is Instrument.OPTION
        assert signal.contract is not None
        assert signal.contract.strike == 105
        assert signal.stop_loss < signal.entry_price
        assert signal.target_price >= 105  # strike magnet
        assert any("unusual call flow" in r for r in signal.reasons)

    def test_put_flow_short_signal(self):
        chain = make_chain(unusual_side="put")
        strat = OptionsFlowStrategy(OptionsFlowSettings())
        snap = MarketSnapshot(symbol="TEST", intraday=make_breakout_df(), chain=chain)
        signal = strat.evaluate(snap)
        assert signal is not None
        assert signal.direction is Direction.SHORT
        assert signal.option_type is OptionType.PUT
        assert signal.stop_loss > signal.entry_price > signal.target_price

    def test_balanced_flow_no_signal(self):
        chain = make_chain(unusual_side="call", dominance=False)
        strat = OptionsFlowStrategy(OptionsFlowSettings())
        snap = MarketSnapshot(symbol="TEST", intraday=make_breakout_df(), chain=chain)
        assert strat.evaluate(snap) is None

    def test_no_chain_no_signal(self):
        strat = OptionsFlowStrategy(OptionsFlowSettings())
        snap = MarketSnapshot(symbol="TEST", intraday=make_breakout_df(), chain=None)
        assert strat.evaluate(snap) is None

    def test_find_unusual_respects_filters(self, call_flow_chain):
        strat = OptionsFlowStrategy(OptionsFlowSettings())
        unusual = strat.find_unusual(call_flow_chain)
        assert len(unusual) == 1
        row = unusual.iloc[0]
        assert row["volume"] >= 2.0 * row["open_interest"]
        assert row["dte"] <= 14
