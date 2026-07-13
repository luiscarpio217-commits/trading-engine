"""Momentum breakout strategy.

Long setup (short is the mirror image when enabled):
  * current bar closes above the highest high of the prior N bars
    and the previous bar had not (fresh breakout only),
  * volume confirmation: bar volume >= `volume_multiple` x 20-bar average,
  * trend alignment: EMA9 > EMA21 > EMA50 and price above session VWAP,
  * RSI(14) between `rsi_min` and `rsi_max` (momentum, but not exhausted).

Stops are ATR-based, targets are a configured reward:risk multiple, and the
signal carries an options recommendation (nearest viable expiry, ~0.40 delta)
when a chain is available.
"""

from __future__ import annotations

import math
from typing import Optional

from ..config import MomentumSettings
from ..data.indicators import rolling_resistance, rolling_support, volume_profile
from ..models import Direction, Instrument, Signal
from .base import MarketSnapshot, Strategy, describe_expiry

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume",
                    "ema9", "ema21", "ema50", "rsi14", "vwap", "atr14",
                    "macd_hist", "rvol")


class MomentumBreakoutStrategy(Strategy):
    name = "momentum_breakout"

    def __init__(self, settings: Optional[MomentumSettings] = None) -> None:
        self.s = settings or MomentumSettings()

    def evaluate(self, snapshot: MarketSnapshot) -> Optional[Signal]:
        df = snapshot.intraday
        needed = self.s.breakout_lookback + 2
        if df is None or len(df) < needed:
            return None
        if any(col not in df.columns for col in REQUIRED_COLUMNS):
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]
        resistance = float(rolling_resistance(df["high"], self.s.breakout_lookback).iloc[-1])
        support = float(rolling_support(df["low"], self.s.breakout_lookback).iloc[-1])
        rvol = float(last["rvol"]) if last["rvol"] == last["rvol"] else 0.0
        atr = float(last["atr14"]) if last["atr14"] == last["atr14"] else 0.0
        if atr <= 0 or math.isnan(resistance) or math.isnan(support):
            return None
        if rvol < self.s.volume_multiple:
            return None

        direction = self._breakout_direction(last, prev, resistance, support)
        if direction is None:
            return None

        entry = float(last["close"])
        if direction is Direction.LONG:
            stop = entry - self.s.atr_stop_multiple * atr
            target = entry + self.s.reward_risk * (entry - stop)
        else:
            stop = entry + self.s.atr_stop_multiple * atr
            target = entry - self.s.reward_risk * (stop - entry)
        if stop <= 0 or target <= 0:
            return None

        level = resistance if direction is Direction.LONG else support
        reasons = [
            f"{'breakout above' if direction is Direction.LONG else 'breakdown below'} "
            f"{self.s.breakout_lookback}-bar level {level:.2f}",
            f"volume confirmation: RVOL {rvol:.1f}x (min {self.s.volume_multiple:.1f}x)",
            f"EMA alignment 9/21/50 {'bullish' if direction is Direction.LONG else 'bearish'}",
            f"price {'above' if direction is Direction.LONG else 'below'} VWAP {last['vwap']:.2f}",
            f"RSI14 {last['rsi14']:.0f}",
        ]

        confidence = 0.55
        confidence += min(0.15, 0.05 * (rvol - self.s.volume_multiple))
        macd_hist = float(last["macd_hist"]) if last["macd_hist"] == last["macd_hist"] else 0.0
        if (macd_hist > 0) == (direction is Direction.LONG) and macd_hist != 0:
            confidence += 0.10
            reasons.append("MACD histogram confirms")
        try:
            vp = volume_profile(df.tail(120))
            above_poc = entry > vp.poc_price
            if above_poc == (direction is Direction.LONG):
                confidence += 0.05
                reasons.append(f"entry {'above' if above_poc else 'below'} volume-profile POC {vp.poc_price:.2f}")
        except Exception:
            pass

        contract = None
        expiry_rec = "nearest weekly expiry, 1-7 DTE"
        if snapshot.chain is not None and not snapshot.chain.empty:
            contract = snapshot.chain.pick_contract(
                direction, min_dte=self.s.min_dte, target_delta=self.s.target_delta)
            if contract is not None:
                expiry_rec = describe_expiry(contract.expiry, snapshot.chain.asof)

        return Signal(
            symbol=snapshot.symbol,
            strategy=self.name,
            direction=direction,
            entry_price=round(entry, 4),
            stop_loss=round(stop, 4),
            target_price=round(target, 4),
            instrument=Instrument.OPTION if contract is not None else Instrument.EQUITY,
            confidence=round(min(confidence, 0.95), 2),
            expiry_recommendation=expiry_rec,
            contract=contract,
            reasons=reasons,
        )

    def _breakout_direction(self, last, prev, resistance: float,
                            support: float) -> Optional[Direction]:
        rsi = float(last["rsi14"])
        long_setup = (
            last["close"] > resistance
            and prev["close"] <= resistance
            and last["ema9"] > last["ema21"] > last["ema50"]
            and last["close"] > last["vwap"]
            and self.s.rsi_min <= rsi <= self.s.rsi_max
        )
        if long_setup:
            return Direction.LONG
        if not self.s.allow_short:
            return None
        rsi_max_short = 100.0 - self.s.rsi_min   # mirror: e.g. 50
        rsi_min_short = 100.0 - self.s.rsi_max   # mirror: e.g. 20
        short_setup = (
            last["close"] < support
            and prev["close"] >= support
            and last["ema9"] < last["ema21"] < last["ema50"]
            and last["close"] < last["vwap"]
            and rsi_min_short <= rsi <= rsi_max_short
        )
        return Direction.SHORT if short_setup else None
