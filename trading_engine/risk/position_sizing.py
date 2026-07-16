"""Position sizing: fixed-fractional risk and (optionally) Kelly criterion.

The sizer answers one question: given this signal and account equity, how
many shares or contracts? The risk budget in dollars is

  * fixed fractional: equity * max_risk_per_trade_pct
  * Kelly:            equity * clamp(kelly_multiplier * f*, 0, kelly_cap)
                      where f* = W - (1-W)/R from realized trade stats;
                      falls back to fixed fractional until `kelly_min_trades`
                      trades exist or when the edge is non-positive.

Shares risk (entry - stop) per share. Long option contracts conservatively
risk the full premium per contract (premium * 100). Both are additionally
capped by `max_position_notional_pct` of equity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..config import RiskSettings
from ..models import Instrument, Signal


@dataclass
class SizingResult:
    qty: int
    risk_dollars: float          # budget allocated to the trade
    risk_pct: float              # fraction of equity that budget represents
    method: str                  # fixed_fractional | kelly
    notional: float
    notes: list[str] = field(default_factory=list)

    @property
    def viable(self) -> bool:
        return self.qty > 0


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Classic Kelly f* = W - (1-W)/R with R = avg_win/avg_loss.

    Returns 0 when inputs are degenerate or the edge is negative.
    """
    if not (0.0 < win_rate < 1.0) or avg_win <= 0 or avg_loss <= 0:
        return 0.0
    r = avg_win / avg_loss
    f = win_rate - (1.0 - win_rate) / r
    return max(f, 0.0)


class PositionSizer:
    def __init__(self, settings: Optional[RiskSettings] = None) -> None:
        self.s = settings or RiskSettings()

    def risk_budget(self, equity: float, stats: Optional[dict] = None) -> tuple[float, float, str]:
        """(risk_dollars, risk_pct, method) for one trade."""
        base_pct = self.s.max_risk_per_trade_pct
        if self.s.sizing_method != "kelly":
            return equity * base_pct, base_pct, "fixed_fractional"

        stats = stats or {}
        trades = int(stats.get("total_trades", 0))
        if trades < self.s.kelly_min_trades:
            return equity * base_pct, base_pct, "fixed_fractional"
        f = kelly_fraction(
            float(stats.get("win_rate", 0.0)),
            float(stats.get("avg_win", 0.0)),
            float(stats.get("avg_loss", 0.0)),
        )
        if f <= 0.0:
            return equity * base_pct, base_pct, "fixed_fractional"
        pct = min(f * self.s.kelly_multiplier, self.s.kelly_cap)
        return equity * pct, pct, "kelly"

    def size(self, signal: Signal, equity: float,
             option_premium: Optional[float] = None,
             stats: Optional[dict] = None) -> SizingResult:
        risk_dollars, risk_pct, method = self.risk_budget(equity, stats)
        max_notional = equity * self.s.max_position_notional_pct
        notes: list[str] = []

        if signal.instrument is Instrument.OPTION:
            premium = option_premium
            if (premium is None or premium <= 0) and signal.contract is not None:
                premium = signal.contract.mid
            if premium is None or premium <= 0:
                return SizingResult(0, risk_dollars, risk_pct, method, 0.0,
                                    ["no usable option premium; cannot size"])
            if premium < self.s.min_option_premium:
                return SizingResult(0, risk_dollars, risk_pct, method, 0.0,
                                    [f"premium ${premium:.2f} below minimum "
                                     f"${self.s.min_option_premium:.2f}; cheap contracts "
                                     f"produce oversized, untradeable quantities"])
            per_contract_risk = premium * 100.0  # long option: full premium at risk
            qty = math.floor(risk_dollars / per_contract_risk)
            qty = min(qty, math.floor(max_notional / per_contract_risk))
            if qty > self.s.max_option_contracts:
                notes.append(f"capped at max_option_contracts "
                             f"({self.s.max_option_contracts}, was {qty})")
                qty = self.s.max_option_contracts
            if qty <= 0:
                notes.append(f"premium ${premium:.2f} exceeds risk budget ${risk_dollars:.2f}")
            notional = max(qty, 0) * premium * 100.0
            return SizingResult(max(qty, 0), risk_dollars, risk_pct, method, notional, notes)

        per_share_risk = signal.risk_per_share
        if per_share_risk <= 0:
            return SizingResult(0, risk_dollars, risk_pct, method, 0.0,
                                ["invalid stop: zero risk per share"])
        qty = math.floor(risk_dollars / per_share_risk)
        if signal.entry_price > 0:
            qty = min(qty, math.floor(max_notional / signal.entry_price))
        if qty <= 0:
            notes.append("risk budget too small for one share within notional cap")
        notional = qty * signal.entry_price
        return SizingResult(max(qty, 0), risk_dollars, risk_pct, method, notional, notes)
