"""Position sizing: fixed-fractional risk and (optionally) Kelly criterion.

The sizer answers one question: given this signal and account equity, how
many shares or contracts? The risk budget in dollars is

  * fixed fractional: equity * max_risk_per_trade_pct
  * Kelly:            equity * clamp(kelly_multiplier * f*, 0, kelly_cap)
                      where f* = W - (1-W)/R from realized trade stats;
                      falls back to fixed fractional until `kelly_min_trades`
                      trades exist or when the edge is non-positive.

Shares risk (entry - stop) per share. Long option contracts risk the loss
at the engine-enforced premium stop: premium * 100 * premium_stop_pct
(default 50% - the order manager closes the position there), mirroring
equity risk-to-stop sizing. Sizing against the full premium instead
zeroed every contract whose premium exceeded budget/100 - on a ~$700
underlying that was every normal near-the-money contract. Caps stay:
min_option_premium, max_option_contracts, and max_position_notional_pct
(applied to full premium notional).

Every zero-size outcome carries a machine-readable `reason` so the signal
status can show the exact gate (e.g. "sized_zero:risk_budget_lt_one_contract").
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
    reason: str = ""             # machine-readable cause when qty == 0

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
                                    ["no usable option premium; cannot size"],
                                    reason="no_premium")
            if premium < self.s.min_option_premium:
                return SizingResult(0, risk_dollars, risk_pct, method, 0.0,
                                    [f"premium ${premium:.2f} below minimum "
                                     f"${self.s.min_option_premium:.2f}; cheap contracts "
                                     f"produce oversized, untradeable quantities"],
                                    reason="premium_below_min")
            # Risk per contract = loss at the engine-enforced premium stop
            # (the order manager closes there), analogous to equity
            # risk-to-stop. Clamped so a degenerate premium_stop_pct can
            # neither zero the divisor nor exceed the full premium.
            stop_fraction = min(max(self.s.premium_stop_pct, 0.10), 1.0)
            per_contract_risk = premium * 100.0 * stop_fraction
            per_contract_notional = premium * 100.0
            trace = (f"premium ${premium:.2f}, at-stop risk "
                     f"${per_contract_risk:.2f}/contract ({stop_fraction:.0%} premium stop), "
                     f"budget ${risk_dollars:.2f}")
            qty = math.floor(risk_dollars / per_contract_risk)
            if qty <= 0:
                return SizingResult(0, risk_dollars, risk_pct, method, 0.0,
                                    [f"one contract risks more than the whole budget: {trace}"],
                                    reason="risk_budget_lt_one_contract")
            notional_qty = math.floor(max_notional / per_contract_notional)
            if notional_qty <= 0:
                return SizingResult(0, risk_dollars, risk_pct, method, 0.0,
                                    [f"one contract exceeds the notional cap "
                                     f"${max_notional:.2f}: {trace}"],
                                    reason="notional_cap_lt_one_contract")
            if qty > notional_qty:
                notes.append(f"capped by notional (${max_notional:.0f}): {qty} -> {notional_qty}")
                qty = notional_qty
            if qty > self.s.max_option_contracts:
                notes.append(f"capped at max_option_contracts "
                             f"({self.s.max_option_contracts}, was {qty})")
                qty = self.s.max_option_contracts
            notes.append(trace)
            return SizingResult(qty, risk_dollars, risk_pct, method,
                                qty * per_contract_notional, notes)

        per_share_risk = signal.risk_per_share
        if per_share_risk <= 0:
            return SizingResult(0, risk_dollars, risk_pct, method, 0.0,
                                ["invalid stop: zero risk per share"],
                                reason="invalid_stop")
        qty = math.floor(risk_dollars / per_share_risk)
        if qty <= 0:
            return SizingResult(0, risk_dollars, risk_pct, method, 0.0,
                                [f"one share risks ${per_share_risk:.2f}, over the "
                                 f"${risk_dollars:.2f} budget"],
                                reason="risk_budget_lt_one_share")
        if signal.entry_price > 0:
            notional_qty = math.floor(max_notional / signal.entry_price)
            if notional_qty <= 0:
                return SizingResult(0, risk_dollars, risk_pct, method, 0.0,
                                    [f"one share exceeds the notional cap ${max_notional:.2f}"],
                                    reason="notional_cap_lt_one_share")
            qty = min(qty, notional_qty)
        return SizingResult(qty, risk_dollars, risk_pct, method,
                            qty * signal.entry_price, notes)
