"""Shared synthetic fixtures: no network, fully deterministic."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from trading_engine.data.options_data import OptionsChain, build_chain_frame


def make_intraday_df(bars_per_day: int = 78, days: int = 2, start_price: float = 100.0,
                     tz: str = "America/New_York") -> pd.DataFrame:
    """A gently trending 5m OHLCV frame across full sessions."""
    idx = []
    for d in range(days):
        day_start = pd.Timestamp("2026-07-13 09:30", tz=tz) + pd.Timedelta(days=d)
        idx.extend(day_start + pd.Timedelta(minutes=5 * i) for i in range(bars_per_day))
    n = len(idx)
    closes = start_price + np.arange(n) * 0.03
    opens = closes - 0.02
    highs = closes + 0.05
    lows = opens - 0.05
    volume = np.full(n, 100_000.0)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
        index=pd.DatetimeIndex(idx),
    )


def make_breakout_df() -> pd.DataFrame:
    """Uptrend, then a tight consolidation, then a high-volume breakout bar.

    Engineered so the momentum strategy's long conditions hold on the last bar:
    fresh close above 20-bar resistance, RVOL >= 1.5, EMA9>EMA21>EMA50,
    close above VWAP, RSI in [50, 80].
    """
    df = make_intraday_df(bars_per_day=78, days=2)
    n = len(df)
    # consolidation for the 21 bars before the last: oscillate under resistance
    base = df["close"].iloc[n - 23]
    for i, k in enumerate(range(n - 22, n - 1)):
        wiggle = 0.10 if i % 2 == 0 else -0.10
        close = base + wiggle
        df.iloc[k, df.columns.get_indexer(["open"])[0]] = base
        df.iloc[k, df.columns.get_indexer(["close"])[0]] = close
        df.iloc[k, df.columns.get_indexer(["high"])[0]] = close + 0.05
        df.iloc[k, df.columns.get_indexer(["low"])[0]] = min(base, close) - 0.05
    # breakout bar: clears consolidation high with 4x volume
    resistance = df["high"].iloc[n - 21:n - 1].max()
    entry = resistance + 0.60
    df.iloc[-1, df.columns.get_indexer(["open"])[0]] = base
    df.iloc[-1, df.columns.get_indexer(["close"])[0]] = entry
    df.iloc[-1, df.columns.get_indexer(["high"])[0]] = entry + 0.05
    df.iloc[-1, df.columns.get_indexer(["low"])[0]] = base - 0.05
    df.iloc[-1, df.columns.get_indexer(["volume"])[0]] = 400_000.0
    return df


def make_chain(spot: float = 100.0, asof: date = date(2026, 7, 13),
               unusual_side: str = "call", dominance: bool = True) -> OptionsChain:
    """A synthetic chain with one clearly unusual near-the-money contract."""
    expiry_near = asof + timedelta(days=4)
    expiry_far = asof + timedelta(days=32)
    rows = []

    def add(opt_type, strike, expiry, vol, oi, iv, bid, ask, delta=None):
        rows.append({
            "type": opt_type, "strike": strike, "expiry": expiry,
            "dte": (expiry - asof).days, "bid": bid, "ask": ask,
            "last": (bid + ask) / 2, "volume": vol, "open_interest": oi,
            "iv": iv, "delta": delta,
        })

    # ordinary background contracts (normal IV ~0.40, volume < OI)
    for strike in (90, 95, 100, 105, 110):
        add("call", strike, expiry_near, 200, 5_000, 0.40, 1.0, 1.2, 0.5 - (strike - 100) * 0.03)
        add("put", strike, expiry_near, 200, 5_000, 0.40, 1.0, 1.2, -0.5 - (strike - 100) * 0.03)
        add("call", strike, expiry_far, 150, 8_000, 0.38, 2.0, 2.4)
        add("put", strike, expiry_far, 150, 8_000, 0.38, 2.0, 2.4)

    # the unusual contract: 5x volume/OI, elevated IV, near the money
    if unusual_side == "call":
        add("call", 105, expiry_near, 5_000, 1_000, 0.90, 1.9, 2.1, 0.42)
        if not dominance:
            add("put", 95, expiry_near, 5_000, 1_000, 0.90, 1.9, 2.1, -0.42)
    else:
        add("put", 95, expiry_near, 5_000, 1_000, 0.90, 1.9, 2.1, -0.42)
        if not dominance:
            add("call", 105, expiry_near, 5_000, 1_000, 0.90, 1.9, 2.1, 0.42)

    return OptionsChain(underlying="TEST", spot=spot, asof=asof,
                        contracts=build_chain_frame(rows, "TEST"))


@pytest.fixture
def breakout_df() -> pd.DataFrame:
    return make_breakout_df()


@pytest.fixture
def call_flow_chain() -> OptionsChain:
    return make_chain(unusual_side="call")
