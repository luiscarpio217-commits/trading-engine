"""Account-level risk controls: daily loss auto-shutoff, position limits.

The RiskManager owns the per-day state machine:

    start_day(equity)  -> snapshots start-of-day equity, clears the halt
    on_trade_closed()  -> accumulates realized P&L, checks the loss limit
    mark_equity()      -> checks the limit against mark-to-market equity
    can_open()         -> gate every new entry passes through

Once `halted` trips it stays tripped for the rest of the session; the engine
optionally flattens everything (`flatten_on_shutoff`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

from ..config import RiskSettings

log = logging.getLogger(__name__)


@dataclass
class DayState:
    session: date
    start_equity: float
    realized_pnl: float = 0.0
    halted: bool = False
    halt_reason: str = ""


class RiskManager:
    def __init__(self, settings: Optional[RiskSettings] = None) -> None:
        self.s = settings or RiskSettings()
        self._day: Optional[DayState] = None

    # -- session lifecycle ---------------------------------------------------

    def start_day(self, session: date, equity: float) -> None:
        if self._day is not None and self._day.session == session:
            return
        self._day = DayState(session=session, start_equity=equity)
        log.info("risk: new session %s, start equity %.2f, daily loss limit %.2f",
                 session, equity, self.max_daily_loss())

    @property
    def day(self) -> Optional[DayState]:
        return self._day

    @property
    def halted(self) -> bool:
        return bool(self._day and self._day.halted)

    @property
    def halt_reason(self) -> str:
        return self._day.halt_reason if self._day else ""

    def max_daily_loss(self) -> float:
        if self._day is None:
            return 0.0
        return self._day.start_equity * self.s.max_daily_loss_pct

    def realized_pnl_today(self) -> float:
        return self._day.realized_pnl if self._day else 0.0

    # -- events ----------------------------------------------------------------

    def on_trade_closed(self, pnl: float) -> None:
        if self._day is None:
            return
        self._day.realized_pnl += pnl
        if self._day.realized_pnl <= -self.max_daily_loss():
            self._halt(f"daily loss limit hit: realized {self._day.realized_pnl:.2f} "
                       f"<= -{self.max_daily_loss():.2f}")

    def mark_equity(self, equity: float) -> None:
        """Check the limit against total (realized + unrealized) drawdown."""
        if self._day is None or self._day.halted:
            return
        drawdown = equity - self._day.start_equity
        if drawdown <= -self.max_daily_loss():
            self._halt(f"daily loss limit hit: equity drawdown {drawdown:.2f} "
                       f"<= -{self.max_daily_loss():.2f}")

    def _halt(self, reason: str) -> None:
        if self._day is None or self._day.halted:
            return
        self._day.halted = True
        self._day.halt_reason = reason
        log.warning("risk: TRADING HALTED - %s", reason)

    # -- gates -------------------------------------------------------------------

    def can_open(self, open_positions: int) -> tuple[bool, str]:
        if self._day is None:
            return False, "session not started"
        if self._day.halted:
            return False, f"halted: {self._day.halt_reason}"
        if open_positions >= self.s.max_open_positions:
            return False, f"max open positions ({self.s.max_open_positions}) reached"
        return True, ""

    @property
    def should_flatten_on_halt(self) -> bool:
        return self.halted and self.s.flatten_on_shutoff
