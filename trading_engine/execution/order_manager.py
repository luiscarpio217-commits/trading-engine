"""Order and position management.

Responsibilities:
  * turn sized signals into broker orders (equity or option),
  * track fills — including partial fills — into managed positions,
  * keep every equity position protected by a broker stop order sized to
    the currently filled quantity (re-placed as partials accrete),
  * monitor stops/targets engine-side (options have no reliable broker
    stops, and the engine-side check also backstops equity stops),
  * premium-based profit protection for option positions: a fixed
    take-profit (close at +N% premium gain) and/or a trailing lock (once
    up arm%, close after a giveback% retrace off the peak mark) — both
    optional, per strategy; whichever triggers first wins,
  * flatten everything on demand (end of day, daily-loss halt),
  * emit a TradeRecord via callback when a round trip completes.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from ..config import ExecutionSettings, ProfitProtectionSettings, RiskSettings
from ..models import (Direction, Instrument, Order, OrderSide, OrderStatus,
                      OrderType, Signal, TradeRecord, utcnow)
from .base import Broker

log = logging.getLogger(__name__)


@dataclass
class ManagedPosition:
    symbol: str                  # tradable symbol (ticker or OCC)
    underlying: str
    instrument: Instrument
    direction: Direction         # thesis: LONG -> long shares / long calls
    strategy: str
    signal_id: str
    multiplier: int
    stop_loss: float             # on the underlying price
    target: float                # on the underlying price
    qty: float = 0.0             # currently open quantity (positive)
    avg_entry: float = 0.0
    entry_time: datetime = field(default_factory=utcnow)
    premium_stop: Optional[float] = None   # options: close if mark <= this
    take_profit_pct: Optional[float] = None  # options: close at +N% premium gain
    trail_arm_pct: Optional[float] = None    # options: start trailing once up N%
    trail_giveback_pct: Optional[float] = None  # then close on N% retrace off peak
    peak_mark: float = 0.0                 # trailing high-water mark (0 = not armed)
    mark: float = 0.0                      # tradable mark
    underlying_mark: float = 0.0
    realized_pnl: float = 0.0
    closed_qty: float = 0.0
    closed_value: float = 0.0
    entry_order_id: str = ""
    stop_order_id: str = ""
    closing: bool = False
    close_reason: str = ""

    @property
    def unrealized_pnl(self) -> float:
        if self.qty <= 0 or self.mark <= 0:
            return 0.0
        per_unit = (self.mark - self.avg_entry if self._long_units
                    else self.avg_entry - self.mark)
        return per_unit * self.qty * self.multiplier

    @property
    def _long_units(self) -> bool:
        """Whether the held units are long (shares bought / options bought)."""
        return self.instrument is Instrument.OPTION or self.direction is Direction.LONG

    def fill_pnl(self, exit_price: float, qty: float) -> float:
        per_unit = (exit_price - self.avg_entry if self._long_units
                    else self.avg_entry - exit_price)
        return per_unit * qty * self.multiplier


class OrderManager:
    def __init__(self, broker: Broker, exec_settings: Optional[ExecutionSettings] = None,
                 risk_settings: Optional[RiskSettings] = None,
                 on_trade_closed: Optional[Callable[[TradeRecord], None]] = None,
                 on_order_update: Optional[Callable[[Order], None]] = None) -> None:
        self.broker = broker
        self.exec_s = exec_settings or ExecutionSettings()
        self.risk_s = risk_settings or RiskSettings()
        self.on_trade_closed = on_trade_closed
        self.on_order_update = on_order_update
        self._orders: dict[str, Order] = {}
        self._seen_fill: dict[str, float] = {}
        self._positions: dict[str, ManagedPosition] = {}
        self._lock = threading.RLock()

    # -- queries -----------------------------------------------------------

    def open_positions(self) -> list[ManagedPosition]:
        with self._lock:
            return [p for p in self._positions.values() if p.qty > 0 or not _entry_done(self._orders.get(p.entry_order_id))]

    def open_orders(self) -> list[Order]:
        with self._lock:
            return [o for o in self._orders.values() if not o.status.is_terminal]

    def has_open_exposure(self, underlying: str) -> bool:
        with self._lock:
            for p in self._positions.values():
                if p.underlying == underlying:
                    return True
            for o in self._orders.values():
                if o.underlying == underlying and not o.status.is_terminal:
                    return True
            return False

    # -- signal execution --------------------------------------------------

    def execute_signal(self, signal: Signal, qty: int,
                       option_premium: Optional[float] = None,
                       profit_protection: Optional[ProfitProtectionSettings] = None
                       ) -> Optional[Order]:
        """Build and submit the entry order for a sized signal."""
        if qty <= 0:
            return None
        use_option = (signal.instrument is Instrument.OPTION
                      and signal.contract is not None
                      and self.exec_s.trade_options)
        if use_option:
            contract = signal.contract
            premium = option_premium or contract.mid
            order = Order(
                symbol=contract.occ,
                underlying=signal.symbol,
                side=OrderSide.BUY_TO_OPEN,
                qty=qty,
                instrument=Instrument.OPTION,
                multiplier=100,
                order_type=self._entry_order_type(),
                limit_price=self._limit_for(premium, buying=True),
                signal_id=signal.id,
                note="entry",
            )
        else:
            side = OrderSide.BUY if signal.direction is Direction.LONG else OrderSide.SELL_SHORT
            order = Order(
                symbol=signal.symbol,
                side=side,
                qty=qty,
                instrument=Instrument.EQUITY,
                multiplier=1,
                order_type=self._entry_order_type(),
                limit_price=self._limit_for(signal.entry_price,
                                            buying=signal.direction is Direction.LONG),
                signal_id=signal.id,
                note="entry",
            )

        with self._lock:
            position = ManagedPosition(
                symbol=order.symbol,
                underlying=signal.symbol,
                instrument=order.instrument,
                direction=signal.direction,
                strategy=signal.strategy,
                signal_id=signal.id,
                multiplier=order.multiplier,
                stop_loss=signal.stop_loss,
                target=signal.target_price,
                entry_order_id=order.id,
            )
            if order.instrument is Instrument.OPTION:
                entry_premium = option_premium or (signal.contract.mid if signal.contract else 0.0)
                if entry_premium and entry_premium > 0:
                    position.premium_stop = entry_premium * (1.0 - self.risk_s.premium_stop_pct)
                if profit_protection is not None:
                    position.take_profit_pct = profit_protection.take_profit_pct
                    if profit_protection.trailing.enabled:
                        position.trail_arm_pct = profit_protection.trailing.arm_pct
                        position.trail_giveback_pct = profit_protection.trailing.giveback_pct
            self._orders[order.id] = order
            self._seen_fill[order.id] = 0.0
            self.broker.submit_order(order)
            self._notify_order(order)
            if order.status is OrderStatus.REJECTED:
                self._orders.pop(order.id, None)
                self._seen_fill.pop(order.id, None)
                return order
            self._positions[order.symbol] = position
            self._absorb_fills(order)
        log.info("entry order %s: %s %s x%d (%s) signal=%s", order.id,
                 order.side.value, order.symbol, qty, order.order_type.value, signal.id)
        return order

    def _entry_order_type(self) -> OrderType:
        return OrderType.LIMIT if self.exec_s.order_type == "limit" else OrderType.MARKET

    def _limit_for(self, reference: float, buying: bool) -> Optional[float]:
        if self.exec_s.order_type != "limit" or not reference:
            return None
        offset = self.exec_s.limit_offset_bps / 10_000.0
        return round(reference * (1 + offset) if buying else reference * (1 - offset), 4)

    # -- polling / fills ------------------------------------------------------

    def poll(self) -> None:
        """Refresh working orders from the broker and absorb fill deltas."""
        with self._lock:
            for order in list(self._orders.values()):
                if order.status.is_terminal and self._seen_fill.get(order.id, 0.0) >= order.filled_qty:
                    continue
                if not order.status.is_terminal:
                    try:
                        self.broker.refresh_order(order)
                    except Exception as exc:
                        log.warning("refresh_order failed for %s: %s", order.id, exc)
                        continue
                self._absorb_fills(order)

    def _absorb_fills(self, order: Order) -> None:
        seen = self._seen_fill.get(order.id, 0.0)
        delta = order.filled_qty - seen
        if delta > 1e-9:
            self._seen_fill[order.id] = order.filled_qty
            if order.note == "entry":
                self._on_entry_fill(order, delta)
            else:
                self._on_close_fill(order, delta)
            self._notify_order(order)
        elif order.status.is_terminal and order.note != "entry":
            self._finalize_if_flat(order)

    def _on_entry_fill(self, order: Order, delta: float) -> None:
        pos = self._positions.get(order.symbol)
        if pos is None:
            return
        total = pos.qty + delta
        pos.avg_entry = (pos.avg_entry * pos.qty + order.avg_fill_price * delta) / total
        pos.qty = total
        pos.mark = order.avg_fill_price
        log.info("entry fill: %s +%g @ %.4f (position %g)", order.symbol, delta,
                 order.avg_fill_price, pos.qty)
        if pos.instrument is Instrument.EQUITY and not pos.closing:
            self._ensure_stop_order(pos)

    def _ensure_stop_order(self, pos: ManagedPosition) -> None:
        """Place or resize the protective broker stop for an equity position."""
        existing = self._orders.get(pos.stop_order_id)
        if existing is not None and not existing.status.is_terminal:
            if abs(existing.qty - pos.qty) < 1e-9:
                return
            try:
                self.broker.cancel_order(existing)
            except Exception as exc:
                log.warning("stop replace: cancel failed for %s: %s", existing.id, exc)
        side = (OrderSide.SELL if pos.direction is Direction.LONG
                else OrderSide.BUY_TO_COVER)
        stop = Order(
            symbol=pos.symbol,
            side=side,
            qty=pos.qty,
            order_type=OrderType.STOP,
            stop_price=round(pos.stop_loss, 4),
            instrument=pos.instrument,
            multiplier=pos.multiplier,
            signal_id=pos.signal_id,
            note="stop_loss",
        )
        self._orders[stop.id] = stop
        self._seen_fill[stop.id] = 0.0
        try:
            self.broker.submit_order(stop)
        except Exception as exc:
            log.error("stop order submit failed for %s: %s (engine-side stop remains)",
                      pos.symbol, exc)
            self._orders.pop(stop.id, None)
            self._seen_fill.pop(stop.id, None)
            return
        self._notify_order(stop)
        if stop.status is OrderStatus.REJECTED:
            log.error("stop order rejected for %s (engine-side stop remains)", pos.symbol)
            self._orders.pop(stop.id, None)
            self._seen_fill.pop(stop.id, None)
            return
        pos.stop_order_id = stop.id
        self._absorb_fills(stop)

    def _on_close_fill(self, order: Order, delta: float) -> None:
        pos = self._positions.get(order.symbol)
        if pos is None:
            return
        delta = min(delta, pos.qty)
        if delta <= 0:
            return
        pnl = pos.fill_pnl(order.avg_fill_price, delta)
        pos.realized_pnl += pnl
        pos.qty -= delta
        pos.closed_qty += delta
        pos.closed_value += order.avg_fill_price * delta
        if not pos.close_reason:
            pos.close_reason = order.note or "close"
        log.info("close fill: %s -%g @ %.4f (%s, pnl %.2f, remaining %g)",
                 order.symbol, delta, order.avg_fill_price, pos.close_reason, pnl, pos.qty)
        self._finalize_if_flat(order)

    def _finalize_if_flat(self, close_order: Order) -> None:
        pos = self._positions.get(close_order.symbol)
        if pos is None or pos.qty > 1e-9 or pos.closed_qty <= 0:
            return
        entry_order = self._orders.get(pos.entry_order_id)
        if entry_order is not None and not entry_order.status.is_terminal:
            return  # entry still working; don't finalize yet
        exit_price = pos.closed_value / pos.closed_qty
        record = TradeRecord(
            symbol=pos.symbol,
            underlying=pos.underlying,
            strategy=pos.strategy,
            direction=pos.direction,
            instrument=pos.instrument,
            qty=pos.closed_qty,
            multiplier=pos.multiplier,
            entry_price=round(pos.avg_entry, 4),
            exit_price=round(exit_price, 4),
            entry_time=pos.entry_time,
            exit_time=utcnow(),
            pnl=round(pos.realized_pnl, 2),
            exit_reason=pos.close_reason or close_order.note or "close",
            signal_id=pos.signal_id,
        )
        self._cancel_if_working(pos.stop_order_id)
        del self._positions[pos.symbol]
        log.info("trade closed: %s %s pnl %.2f (%s)", record.underlying,
                 record.strategy, record.pnl, record.exit_reason)
        if self.on_trade_closed is not None:
            try:
                self.on_trade_closed(record)
            except Exception:
                log.exception("on_trade_closed callback failed")

    # -- engine-side stop/target monitoring -------------------------------------

    def manage(self, underlying_marks: dict[str, float],
               tradable_marks: Optional[dict[str, float]] = None) -> None:
        """Update marks and enforce stops/targets engine-side."""
        tradable_marks = tradable_marks or {}
        with self._lock:
            for pos in list(self._positions.values()):
                if pos.symbol in tradable_marks:
                    pos.mark = tradable_marks[pos.symbol]
                if pos.underlying in underlying_marks:
                    pos.underlying_mark = underlying_marks[pos.underlying]
                if pos.closing or pos.qty <= 0:
                    continue
                self._update_trailing_peak(pos)
                reason = self._exit_reason(pos)
                if reason:
                    self.close_position(pos, reason)

    def _update_trailing_peak(self, pos: ManagedPosition) -> None:
        """Arm and ratchet the trailing high-water mark. Persists on the
        position across manage cycles; a new high can never trigger its own
        retrace (peak is updated before the exit check reads it)."""
        if (pos.instrument is not Instrument.OPTION
                or pos.trail_arm_pct is None or pos.trail_giveback_pct is None
                or pos.avg_entry <= 0 or pos.mark <= 0):
            return
        if pos.peak_mark > 0:                       # armed: ratchet up only
            pos.peak_mark = max(pos.peak_mark, pos.mark)
        elif pos.mark >= pos.avg_entry * (1.0 + pos.trail_arm_pct / 100.0):
            pos.peak_mark = pos.mark
            log.info("trailing armed for %s: mark %.4f (entry %.4f, +%.0f%%)",
                     pos.symbol, pos.mark, pos.avg_entry, pos.trail_arm_pct)

    def _exit_reason(self, pos: ManagedPosition) -> str:
        u = pos.underlying_mark
        if u > 0:
            if pos.direction is Direction.LONG:
                if u <= pos.stop_loss:
                    return "stop_loss"
                if u >= pos.target:
                    return "target"
            else:
                if u >= pos.stop_loss:
                    return "stop_loss"
                if u <= pos.target:
                    return "target"
        if (pos.instrument is Instrument.OPTION and pos.premium_stop is not None
                and 0 < pos.mark <= pos.premium_stop):
            return "premium_stop"
        # premium-based profit protection (options only). Protective exits
        # above always win the cycle; between these two, take-profit is
        # checked first — if both are true in one snapshot the mark is at or
        # beyond the take-profit level, so that label is the accurate one.
        if pos.instrument is Instrument.OPTION and pos.mark > 0 and pos.avg_entry > 0:
            if (pos.take_profit_pct is not None
                    and pos.mark >= pos.avg_entry * (1.0 + pos.take_profit_pct / 100.0)):
                return "take_profit"
            if (pos.peak_mark > 0 and pos.trail_giveback_pct is not None
                    and pos.mark <= pos.peak_mark * (1.0 - pos.trail_giveback_pct / 100.0)):
                return "trailing_lock"
        return ""

    def close_position(self, pos: ManagedPosition, reason: str) -> Optional[Order]:
        with self._lock:
            if pos.closing:
                return None
            entry = self._orders.get(pos.entry_order_id)
            if entry is not None and not entry.status.is_terminal:
                try:
                    self.broker.cancel_order(entry)
                    self.broker.refresh_order(entry)
                    self._absorb_fills(entry)
                except Exception as exc:
                    log.warning("cancel entry %s failed: %s", entry.id, exc)
            self._cancel_if_working(pos.stop_order_id)
            if pos.qty <= 0:
                # nothing filled; drop the placeholder position
                self._positions.pop(pos.symbol, None)
                return None
            pos.closing = True
            pos.close_reason = reason
            if pos.instrument is Instrument.OPTION:
                side = OrderSide.SELL_TO_CLOSE
            else:
                side = (OrderSide.SELL if pos.direction is Direction.LONG
                        else OrderSide.BUY_TO_COVER)
            order = Order(
                symbol=pos.symbol,
                underlying=pos.underlying,
                side=side,
                qty=pos.qty,
                order_type=OrderType.MARKET,
                instrument=pos.instrument,
                multiplier=pos.multiplier,
                signal_id=pos.signal_id,
                note=reason,
            )
            self._orders[order.id] = order
            self._seen_fill[order.id] = 0.0
            try:
                self.broker.submit_order(order)
            except Exception as exc:
                log.error("close order submit failed for %s: %s", pos.symbol, exc)
                self._orders.pop(order.id, None)
                self._seen_fill.pop(order.id, None)
                pos.closing = False
                return None
            self._notify_order(order)
            if order.status is OrderStatus.REJECTED:
                pos.closing = False  # retry on next manage() pass
                return order
            self._absorb_fills(order)
            return order

    def flatten_all(self, reason: str) -> None:
        with self._lock:
            for order in list(self._orders.values()):
                if not order.status.is_terminal and order.note == "entry":
                    try:
                        self.broker.cancel_order(order)
                    except Exception:
                        pass
            for pos in list(self._positions.values()):
                if not pos.closing:
                    self.close_position(pos, reason)

    # -- helpers -------------------------------------------------------------------

    def _cancel_if_working(self, order_id: str) -> None:
        order = self._orders.get(order_id)
        if order is not None and not order.status.is_terminal:
            try:
                self.broker.cancel_order(order)
                order.status = OrderStatus.CANCELED if order.filled_qty == 0 else order.status
            except Exception as exc:
                log.warning("cancel %s failed: %s", order_id, exc)

    def _notify_order(self, order: Order) -> None:
        if self.on_order_update is not None:
            try:
                self.on_order_update(order)
            except Exception:
                log.exception("on_order_update callback failed")


def _entry_done(order: Optional[Order]) -> bool:
    return order is None or order.status.is_terminal
