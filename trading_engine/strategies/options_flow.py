"""Options flow (unusual options activity) strategy.

Scans near-dated contracts for unusual activity:

  * contract volume >= `volume_oi_ratio` x open interest (fresh positioning,
    not existing inventory),
  * contract volume >= `min_volume`,
  * implied volatility >= `iv_multiple` x chain median IV (someone paying up),
  * strike within `moneyness_pct` of spot (near-the-money flow is the most
    directional).

Unusual premium (volume x mid x 100) is aggregated per side; when one side
out-weighs the other by `dominance_ratio`, that side sets the direction and
the highest-premium contract becomes both the recommended contract and the
price magnet for the target.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ..config import OptionsFlowSettings
from ..data.options_data import OptionsChain, row_to_contract
from ..models import Direction, Instrument, Signal
from .base import MarketSnapshot, Strategy, describe_expiry


class OptionsFlowStrategy(Strategy):
    name = "options_flow"

    def __init__(self, settings: Optional[OptionsFlowSettings] = None) -> None:
        self.s = settings or OptionsFlowSettings()

    def evaluate(self, snapshot: MarketSnapshot) -> Optional[Signal]:
        chain = snapshot.chain
        if chain is None or chain.empty or not chain.spot:
            return None
        unusual = self.find_unusual(chain)
        if unusual.empty:
            return None

        spot = float(chain.spot)
        premium = unusual["volume"] * unusual["mid"].fillna(0.0) * 100.0
        call_premium = float(premium[unusual["type"] == "call"].sum())
        put_premium = float(premium[unusual["type"] == "put"].sum())

        if call_premium > 0 and call_premium >= self.s.dominance_ratio * max(put_premium, 1.0):
            direction = Direction.LONG
            side, side_premium, other_premium = "call", call_premium, put_premium
        elif put_premium > 0 and put_premium >= self.s.dominance_ratio * max(call_premium, 1.0):
            direction = Direction.SHORT
            side, side_premium, other_premium = "put", put_premium, call_premium
        else:
            return None  # flow is two-sided; no directional edge

        side_rows = unusual[unusual["type"] == side]
        top = side_rows.loc[(side_rows["volume"] * side_rows["mid"].fillna(0.0)).idxmax()]
        contract = row_to_contract(top, chain.underlying)

        entry = spot
        if direction is Direction.LONG:
            stop = entry * (1.0 - self.s.stop_pct)
            min_target = entry + self.s.reward_risk * (entry - stop)
            target = max(float(top["strike"]), min_target)
        else:
            stop = entry * (1.0 + self.s.stop_pct)
            min_target = entry - self.s.reward_risk * (stop - entry)
            target = min(float(top["strike"]), min_target)

        vol_oi = float(top["volume"]) / max(float(top["open_interest"]), 1.0)
        confidence = 0.5
        confidence += min(0.2, 0.05 * (vol_oi - self.s.volume_oi_ratio))
        dominance = side_premium / max(other_premium, 1.0)
        confidence += min(0.15, 0.03 * (dominance - self.s.dominance_ratio))

        reasons = [
            f"unusual {side} flow: ${side_premium:,.0f} premium vs ${other_premium:,.0f} opposite",
            f"top contract {contract.describe()}: volume {int(top['volume']):,} "
            f"= {vol_oi:.1f}x open interest {int(top['open_interest']):,}",
            f"contract IV {float(top['iv']):.0%} vs chain median {chain.median_iv():.0%}",
            f"strike within {self.s.moneyness_pct:.0%} of spot {spot:.2f}",
        ]

        return Signal(
            symbol=snapshot.symbol,
            strategy=self.name,
            direction=direction,
            entry_price=round(entry, 4),
            stop_loss=round(stop, 4),
            target_price=round(target, 4),
            instrument=Instrument.OPTION,
            confidence=round(min(confidence, 0.95), 2),
            expiry_recommendation=describe_expiry(contract.expiry, chain.asof),
            contract=contract,
            reasons=reasons,
        )

    def find_unusual(self, chain: OptionsChain) -> pd.DataFrame:
        """Contracts passing all unusual-activity screens."""
        df = chain.contracts
        if df.empty:
            return df
        median_iv = chain.median_iv()
        spot = float(chain.spot)
        mask = (
            (df["dte"] >= 0)
            & (df["dte"] <= self.s.max_dte)
            & (df["volume"] >= self.s.min_volume)
            & (df["open_interest"] > 0)
            & (df["volume"] >= self.s.volume_oi_ratio * df["open_interest"])
            & ((df["strike"] / spot - 1.0).abs() <= self.s.moneyness_pct)
        )
        if median_iv == median_iv and median_iv > 0:  # NaN-safe
            mask &= df["iv"] >= self.s.iv_multiple * median_iv
        return df[mask]
