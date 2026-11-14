"""Insider breakout: buy APEX watchlist stocks when price confirms.

Only runs on tickers from the APEX watchlist (a recent, high-scoring
open-market insider buy). Long setup:
  * price is above the prior session's high, but not more than
    max_chase_pct above it (don't chase a big gap),
  * the prior close is within near_high_pct of the 52-week high,
  * price is inside [min_price, max_price].
Dollar-volume liquidity is checked by the engine before this runs.

Exits are fixed percentages (stop_pct / target_pct) and the position is
held up to max_hold_days sessions - the same rules APEX's backtest measures
on daily bars, so paper results can be compared with the backtest directly.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from ..apex.watchlist import WatchlistEntry
from ..config import ApexSettings
from ..models import Direction, Instrument, Signal
from .base import MarketSnapshot, Strategy

TZ = "America/New_York"


def _session_dates(index: pd.Index) -> list[date]:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert(TZ)
    return list(idx.date)


def prior_session(daily: pd.DataFrame, today: date) -> Optional[pd.Series]:
    """The last completed daily bar before `today` (yfinance includes today's
    partial bar during the session)."""
    if daily is None or daily.empty:
        return None
    dates = _session_dates(daily.index)
    for i in range(len(daily) - 1, -1, -1):
        if dates[i] < today:
            return daily.iloc[i]
    return None


class InsiderBreakoutStrategy(Strategy):
    name = "insider_breakout"

    def __init__(self, settings: Optional[ApexSettings] = None) -> None:
        self.s = settings or ApexSettings()

    def evaluate(self, snapshot: MarketSnapshot,
                 entry: Optional[WatchlistEntry] = None) -> Optional[Signal]:
        if entry is None:
            return None
        intraday, daily = snapshot.intraday, snapshot.daily
        if intraday is None or intraday.empty or daily is None or daily.empty:
            return None
        today = _session_dates(intraday.index)[-1]
        prev = prior_session(daily, today)
        if prev is None:
            return None
        prev_high = float(prev["high"])
        price = float(intraday["close"].iloc[-1])
        if prev_high <= 0 or price <= prev_high:
            return None
        if price > prev_high * (1.0 + self.s.max_chase_pct):
            return None
        if not (self.s.min_price <= price <= self.s.max_price):
            return None

        completed = daily[[d < today for d in _session_dates(daily.index)]]
        if len(completed) < self.s.min_daily_bars:
            return None
        year_high = float(completed["high"].tail(self.s.high_lookback_days).max())
        prev_close = float(prev["close"])
        if (self.s.near_high_pct is not None
                and prev_close < year_high * (1.0 - self.s.near_high_pct)):
            return None

        stop = price * (1.0 - self.s.stop_pct)
        target = price * (1.0 + self.s.target_pct)
        who = ", ".join(entry.insiders[:3]) or "insider"
        reasons = [
            f"APEX score {entry.score:.0f}: {entry.insider_class or 'insider'} buy by {who}"
            f" (${entry.total_value_usd:,.0f}, filed {entry.filed_date})",
            f"price {price:.2f} broke prior-day high {prev_high:.2f} "
            f"(cap +{self.s.max_chase_pct:.0%})",
            f"prior close {prev_close:.2f} vs 52-week high {year_high:.2f}",
            f"hold up to {self.s.max_hold_days} sessions, stop -{self.s.stop_pct:.0%}, "
            f"target +{self.s.target_pct:.0%}",
        ]
        if entry.cluster_size > 1:
            reasons.insert(1, f"cluster: {entry.cluster_size} insiders buying")
        return Signal(
            symbol=snapshot.symbol,
            strategy=self.name,
            direction=Direction.LONG,
            entry_price=round(price, 4),
            stop_loss=round(stop, 4),
            target_price=round(target, 4),
            instrument=Instrument.EQUITY,
            confidence=round(min(0.5 + entry.score / 200.0, 0.95), 2),
            expiry_recommendation=f"shares, hold <= {self.s.max_hold_days} sessions",
            reasons=reasons,
            source="apex",
            max_hold_days=self.s.max_hold_days,
        )
