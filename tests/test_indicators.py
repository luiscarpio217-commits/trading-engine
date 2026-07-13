import numpy as np
import pandas as pd
import pytest

from trading_engine.data import indicators as ind
from tests.conftest import make_intraday_df


def test_ema_matches_pandas():
    s = pd.Series(np.linspace(10, 20, 60))
    expected = s.ewm(span=9, adjust=False, min_periods=9).mean()
    pd.testing.assert_series_equal(ind.ema(s, 9), expected)


def test_rsi_wilder_reference():
    """Compare against a loop implementation of the same recursion pandas uses.

    ewm(alpha, adjust=False) seeds the state with the first observation
    (the first diff, at index 1), then applies s = a*x + (1-a)*s.
    """
    rng = np.random.default_rng(7)
    s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
    period = 14
    alpha = 1.0 / period
    delta = s.diff().to_numpy()
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full(len(s), np.nan)
    avg_loss = np.full(len(s), np.nan)
    g, l = gains[1], losses[1]
    for i in range(2, len(s)):
        g = gains[i] * alpha + g * (1 - alpha)
        l = losses[i] * alpha + l * (1 - alpha)
        if i >= period:
            avg_gain[i], avg_loss[i] = g, l
    expected = 100 - 100 / (1 + avg_gain / avg_loss)
    got = ind.rsi(s, period).to_numpy()
    np.testing.assert_allclose(got[period + 1:], expected[period + 1:], rtol=1e-9)


def test_rsi_extremes():
    up = pd.Series(np.arange(1.0, 40.0))
    assert ind.rsi(up).iloc[-1] > 99.0
    flat = pd.Series(np.ones(40))
    assert ind.rsi(flat).iloc[-1] == pytest.approx(50.0)


def test_vwap_resets_per_session():
    df = make_intraday_df(bars_per_day=5, days=2)
    v = ind.vwap(df)
    # first bar of each session: vwap == that bar's typical price
    for k in (0, 5):
        typical = (df["high"].iloc[k] + df["low"].iloc[k] + df["close"].iloc[k]) / 3
        assert v.iloc[k] == pytest.approx(typical)
    # within a session vwap is cumulative, so bar 6's vwap uses only bars 5-6
    t5 = (df["high"].iloc[5] + df["low"].iloc[5] + df["close"].iloc[5]) / 3
    t6 = (df["high"].iloc[6] + df["low"].iloc[6] + df["close"].iloc[6]) / 3
    assert v.iloc[6] == pytest.approx((t5 * 100_000 + t6 * 100_000) / 200_000)


def test_bollinger_constant_series():
    s = pd.Series(np.full(30, 50.0))
    bb = ind.bollinger(s)
    assert bb["bb_upper"].iloc[-1] == pytest.approx(50.0)
    assert bb["bb_lower"].iloc[-1] == pytest.approx(50.0)
    assert bb["bb_width"].iloc[-1] == pytest.approx(0.0)


def test_bollinger_bands_ordering():
    rng = np.random.default_rng(1)
    s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 100)))
    bb = ind.bollinger(s)
    tail = bb.dropna()
    assert (tail["bb_upper"] >= tail["bb_mid"]).all()
    assert (tail["bb_mid"] >= tail["bb_lower"]).all()


def test_atr_constant_range():
    n = 60
    close = np.full(n, 100.0)
    df = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0,
                       "close": close, "volume": np.ones(n)})
    # every bar: TR = 2.0 -> ATR converges to 2.0
    assert ind.atr(df).iloc[-1] == pytest.approx(2.0)


def test_macd_consistency():
    rng = np.random.default_rng(3)
    s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 120)))
    out = ind.macd(s)
    fast = s.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = s.ewm(span=26, adjust=False, min_periods=26).mean()
    np.testing.assert_allclose(out["macd"].iloc[30:], (fast - slow).iloc[30:], rtol=1e-12)
    np.testing.assert_allclose(
        out["macd_hist"].iloc[40:],
        (out["macd"] - out["macd_signal"]).iloc[40:], rtol=1e-12)


def test_relative_volume_spike():
    vol = pd.Series([100.0] * 30 + [400.0])
    assert ind.relative_volume(vol).iloc[-1] == pytest.approx(4.0)


def test_rolling_resistance_excludes_current_bar():
    high = pd.Series([1, 2, 3, 4, 100.0])
    res = ind.rolling_resistance(high, lookback=4)
    assert res.iloc[-1] == pytest.approx(4.0)  # not 100


def test_volume_profile_poc_and_value_area():
    n = 300
    rng = np.random.default_rng(5)
    # cluster most trading around 105
    prices = np.concatenate([np.full(200, 105.0) + rng.normal(0, 0.2, 200),
                             np.linspace(100, 110, 100)])
    df = pd.DataFrame({
        "open": prices, "high": prices + 0.1, "low": prices - 0.1,
        "close": prices,
        "volume": np.concatenate([np.full(200, 50_000.0), np.full(100, 5_000.0)]),
    })
    vp = ind.volume_profile(df, bins=20)
    assert 104 <= vp.poc_price <= 106
    assert vp.value_area_low <= vp.poc_price <= vp.value_area_high
    assert vp.total_volume == pytest.approx(df["volume"].sum())
    covered = vp.profile[(vp.profile["price_mid"] >= vp.value_area_low)
                         & (vp.profile["price_mid"] <= vp.value_area_high)]["volume"].sum()
    assert covered >= 0.7 * vp.total_volume


def test_compute_indicators_columns(breakout_df):
    out = ind.compute_indicators(breakout_df)
    for col in ("ema9", "ema21", "ema50", "rsi14", "vwap", "bb_upper", "bb_lower",
                "atr14", "macd", "macd_signal", "macd_hist", "rvol"):
        assert col in out.columns
        assert out[col].notna().iloc[-1], f"{col} is NaN on the last bar"
