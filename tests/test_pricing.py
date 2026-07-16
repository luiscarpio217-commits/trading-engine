"""Regression tests for the bounded option mark model (data/pricing.py).

Motivated by a live paper session where a far-OTM $0.06 0DTE put was marked
at $1.62 by the old linear-delta model (default delta 0.5), fabricating a
+$27k profit on flatten. The model must be calibrated, conservative and
bounded.
"""

from datetime import date, datetime, timedelta, timezone

import pytest

from trading_engine.data.pricing import (MIN_MARK, OptionMarkModel, bs_price,
                                         implied_vol, intrinsic_value,
                                         years_to_expiry)
from trading_engine.models import OptionType

NOW = datetime(2026, 7, 14, 14 + 4, 30, tzinfo=timezone.utc)  # 14:30 ET
TODAY = date(2026, 7, 14)  # 0DTE relative to NOW


class TestBlackScholes:
    def test_price_bounds_and_monotonicity(self):
        # call >= intrinsic, increases with vol
        lo = bs_price(OptionType.CALL, 100, 95, 0.2, 0.05)
        hi = bs_price(OptionType.CALL, 100, 95, 0.6, 0.05)
        assert lo >= intrinsic_value(OptionType.CALL, 100, 95)
        assert hi > lo
        # put-call parity with r=0: C - P = S - K
        c = bs_price(OptionType.CALL, 100, 97, 0.4, 0.1)
        p = bs_price(OptionType.PUT, 100, 97, 0.4, 0.1)
        assert c - p == pytest.approx(100 - 97, abs=1e-9)

    def test_zero_time_is_intrinsic(self):
        assert bs_price(OptionType.PUT, 94, 97, 0.5, 0.0) == pytest.approx(3.0)
        assert bs_price(OptionType.CALL, 94, 97, 0.5, 0.0) == 0.0

    def test_implied_vol_round_trip(self):
        t = 0.02
        price = bs_price(OptionType.PUT, 100, 97, 0.45, t)
        iv = implied_vol(OptionType.PUT, 100, 97, t, price)
        assert iv == pytest.approx(0.45, abs=1e-4)

    def test_implied_vol_degenerate_inputs(self):
        assert implied_vol(OptionType.PUT, 100, 97, 0.02, 0.0) is None
        # price below intrinsic cannot be calibrated
        assert implied_vol(OptionType.PUT, 90, 97, 0.02, 5.0) is None


class TestOptionMarkModel:
    def make_otm_0dte_put(self, premium=0.06):
        """The live-session culprit: $0.06 far-OTM put expiring today."""
        return OptionMarkModel.calibrate(
            OptionType.PUT, strike=97.0, expiry=TODAY,
            entry_spot=100.0, entry_premium=premium, entry_time=NOW)

    def test_mark_at_entry_equals_entry_premium(self):
        model = self.make_otm_0dte_put()
        assert model.iv is not None  # calibration succeeded
        assert model.mark(100.0, NOW) == pytest.approx(0.06, abs=0.01)

    def test_no_linear_delta_blowup(self):
        """Spot -3% moved the old model 0.06 -> 1.62. Bounded model stays sane."""
        model = self.make_otm_0dte_put()
        mark = model.mark(97.0, NOW + timedelta(minutes=30))  # spot at the strike
        assert MIN_MARK <= mark < 0.9   # old model: 1.56 at this spot
        # still far below the phantom 1.62 even exactly at the strike

    def test_deep_itm_tracks_intrinsic(self):
        model = self.make_otm_0dte_put()
        mark = model.mark(94.0, NOW + timedelta(hours=1))
        intrinsic = 3.0
        assert mark >= intrinsic
        assert mark <= intrinsic + 1.0  # bounded extrinsic on top

    def test_converges_to_intrinsic_at_expiry(self):
        model = self.make_otm_0dte_put()
        just_before_close = datetime(2026, 7, 14, 19, 59, tzinfo=timezone.utc)
        assert model.mark(95.0, just_before_close) == pytest.approx(2.0, abs=0.15)
        after_expiry = datetime(2026, 7, 14, 21, 0, tzinfo=timezone.utc)
        assert model.mark(95.0, after_expiry) == pytest.approx(2.0, abs=1e-6)
        assert model.mark(99.0, after_expiry) == MIN_MARK  # OTM -> floor

    def test_mark_never_below_floor(self):
        model = self.make_otm_0dte_put()
        assert model.mark(120.0, NOW) >= MIN_MARK

    def test_fallback_extrinsic_never_grows(self):
        """Uncalibratable entries decay entry extrinsic; never mark up OTM."""
        model = OptionMarkModel(
            option_type=OptionType.PUT, strike=97.0, expiry=TODAY,
            entry_spot=100.0, entry_premium=0.06, entry_time=NOW, iv=None)
        later = NOW + timedelta(hours=2)
        otm_mark = model.mark(98.5, later)      # still OTM
        assert otm_mark <= 0.06 + 1e-9          # extrinsic only decays
        itm_mark = model.mark(95.0, later)
        assert 2.0 <= itm_mark <= 2.0 + 0.06    # intrinsic + decayed extrinsic

    def test_multi_day_option_reprices_reasonably(self):
        model = OptionMarkModel.calibrate(
            OptionType.CALL, strike=105.0, expiry=TODAY + timedelta(days=4),
            entry_spot=104.8, entry_premium=2.0, entry_time=NOW)
        assert model.iv is not None
        up = model.mark(106.5, NOW + timedelta(hours=1))
        down = model.mark(103.0, NOW + timedelta(hours=1))
        assert up > 2.0 > down
        assert up >= intrinsic_value(OptionType.CALL, 106.5, 105.0)


def test_years_to_expiry_never_negative():
    assert years_to_expiry(TODAY, NOW + timedelta(days=10)) == 0.0
    assert years_to_expiry(TODAY, NOW) > 0.0
