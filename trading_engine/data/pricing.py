"""Bounded option mark estimation for paper trading (Black-Scholes based).

The paper broker has no live option quotes, so open option positions are
marked with a Black-Scholes model **calibrated to the actual entry fill**:
implied volatility is backed out from the entry premium at the entry spot,
then held constant while spot and time-to-expiry evolve. Properties:

  * mark(entry_spot, entry_time) == entry_premium by construction — no jump
    at entry, one source of truth;
  * marks converge to intrinsic value at expiry (crucial for 0DTE);
  * extrinsic value is bounded by the at-the-money extrinsic at the
    calibrated vol — a far-OTM $0.06 contract can no longer be marked as if
    it had 0.50 delta (the linear-delta model marked such a put at 1.62
    after a $3 spot drop; this model keeps it in pennies until intrinsic).

If calibration fails (entry premium at/below intrinsic, degenerate inputs),
a conservative fallback marks the option at intrinsic(spot) plus the entry
extrinsic decayed linearly to expiry — extrinsic never grows, marks stay
bounded and floored at $0.01.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timezone
from typing import Optional

from ..models import OptionType

YEAR_SECONDS = 365.25 * 24 * 3600.0
MIN_T_YEARS = 600.0 / YEAR_SECONDS   # never price with under ~10 minutes left
MIN_MARK = 0.01


def expiry_datetime(expiry: date) -> datetime:
    """Expiry at the 16:00 ET close, approximated as 20:00 UTC year-round."""
    return datetime.combine(expiry, dtime(20, 0), tzinfo=timezone.utc)


def years_to_expiry(expiry: date, now: datetime) -> float:
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return max((expiry_datetime(expiry) - now).total_seconds(), 0.0) / YEAR_SECONDS


def intrinsic_value(option_type: OptionType, spot: float, strike: float) -> float:
    if option_type is OptionType.CALL:
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(option_type: OptionType, spot: float, strike: float,
             vol: float, t_years: float) -> float:
    """Black-Scholes price with r = q = 0 (adequate for intraday paper marks)."""
    if spot <= 0.0 or strike <= 0.0:
        return 0.0
    if t_years <= 0.0 or vol <= 0.0:
        return intrinsic_value(option_type, spot, strike)
    st = vol * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + 0.5 * vol * vol * t_years) / st
    d2 = d1 - st
    call = spot * _norm_cdf(d1) - strike * _norm_cdf(d2)
    if option_type is OptionType.CALL:
        return max(call, 0.0)
    return max(call - spot + strike, 0.0)  # put-call parity, r = 0


def implied_vol(option_type: OptionType, spot: float, strike: float,
                t_years: float, price: float,
                lo: float = 1e-3, hi: float = 8.0) -> Optional[float]:
    """Back out BS vol by bisection. None when no vol reproduces the price."""
    if spot <= 0.0 or strike <= 0.0 or t_years <= 0.0 or price <= 0.0:
        return None
    if price <= intrinsic_value(option_type, spot, strike) + 1e-9:
        return None
    if price >= bs_price(option_type, spot, strike, hi, t_years):
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bs_price(option_type, spot, strike, mid, t_years) < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class OptionMarkModel:
    """Per-position mark model, calibrated once at entry."""

    option_type: OptionType
    strike: float
    expiry: date
    entry_spot: float
    entry_premium: float
    entry_time: datetime
    iv: Optional[float] = None   # calibrated at entry; None -> decay fallback

    @classmethod
    def calibrate(cls, option_type: OptionType, strike: float, expiry: date,
                  entry_spot: float, entry_premium: float,
                  entry_time: Optional[datetime] = None) -> "OptionMarkModel":
        entry_time = entry_time or datetime.now(timezone.utc)
        t = max(years_to_expiry(expiry, entry_time), MIN_T_YEARS)
        iv = implied_vol(option_type, entry_spot, strike, t, entry_premium)
        return cls(option_type=option_type, strike=strike, expiry=expiry,
                   entry_spot=entry_spot, entry_premium=entry_premium,
                   entry_time=entry_time, iv=iv)

    def mark(self, spot: float, now: Optional[datetime] = None) -> float:
        now = now or datetime.now(timezone.utc)
        t = years_to_expiry(self.expiry, now)
        if self.iv is not None:
            if t <= 0.0:
                est = intrinsic_value(self.option_type, spot, self.strike)
            else:
                est = bs_price(self.option_type, spot, self.strike,
                               self.iv, max(t, MIN_T_YEARS))
        else:
            # Conservative fallback: intrinsic plus linearly decaying entry
            # extrinsic. Extrinsic never grows and dies at expiry.
            t_entry = max(years_to_expiry(self.expiry, self.entry_time), MIN_T_YEARS)
            entry_extrinsic = max(
                self.entry_premium
                - intrinsic_value(self.option_type, self.entry_spot, self.strike),
                0.0)
            frac = min(max(t / t_entry, 0.0), 1.0)
            est = intrinsic_value(self.option_type, spot, self.strike) \
                + entry_extrinsic * frac
        return round(max(est, MIN_MARK), 4)
