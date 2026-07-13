"""In-process paper broker: deterministic fills against engine-supplied marks.

No credentials or network needed, so the whole engine runs out of the box.
Fill model:
  * market orders fill at the current mark +/- slippage (buys pay up),
  * limit orders fill when the mark crosses the limit,
  * stop orders trigger when the mark crosses the stop, then fill as market.

`partial_fill_qty` caps how much of an order fills per `process()` pass —
production leaves it None (full fills); tests use it to exercise the order
manager's partial-fill handling.
"""

from __future__ import annotations

import itertools
import logging
import threading
from typing import Optional

from ..models import (AccountInfo, BrokerPosition, Order, OrderSide, OrderStatus,
                      OrderType, utcnow)
from .base import Broker

log = logging.getLogger(__name__)


class PaperBroker(Broker):
    name = "paper"

    def __init__(self, starting_cash: float = 100_000.0, slippage_bps: float = 5.0,
                 partial_fill_qty: Optional[float] = None) -> None:
        self._cash = starting_cash
        self._slippage = slippage_bps / 10_000.0
        self._partial_fill_qty = partial_fill_qty
        self._orders: dict[str, Order] = {}
        self._positions: dict[str, BrokerPosition] = {}
        self._marks: dict[str, float] = {}
        self._lock = threading.RLock()
        self._ids = itertools.count(1)

    # -- marks -------------------------------------------------------------

    def set_mark(self, symbol: str, price: float) -> None:
        if price is None or price <= 0:
            return
        with self._lock:
            self._marks[symbol] = float(price)
            if symbol in self._positions:
                self._positions[symbol].mark = float(price)

    def get_quote(self, symbol: str) -> Optional[float]:
        with self._lock:
            return self._marks.get(symbol)

    # -- Broker API ----------------------------------------------------------

    def get_account(self) -> AccountInfo:
        with self._lock:
            positions_value = sum(p.market_value for p in self._positions.values())
            equity = self._cash + positions_value
            return AccountInfo(equity=equity, cash=self._cash,
                               buying_power=max(self._cash, 0.0))

    def submit_order(self, order: Order) -> Order:
        with self._lock:
            order.broker_order_id = f"paper-{next(self._ids)}"
            order.status = OrderStatus.NEW
            order.submitted_at = utcnow()
            self._orders[order.broker_order_id] = order
            self._try_fill(order)
            return order

    def refresh_order(self, order: Order) -> Order:
        with self._lock:
            self._try_fill(order)
            return order

    def cancel_order(self, order: Order) -> None:
        with self._lock:
            tracked = self._orders.get(order.broker_order_id, order)
            if not tracked.status.is_terminal:
                tracked.status = (OrderStatus.CANCELED if tracked.filled_qty == 0
                                  else OrderStatus.FILLED if tracked.remaining_qty == 0
                                  else OrderStatus.CANCELED)
                tracked.updated_at = utcnow()

    def list_positions(self) -> list[BrokerPosition]:
        with self._lock:
            return [BrokerPosition(**vars(p)) for p in self._positions.values()
                    if p.qty != 0]

    def process(self) -> None:
        """Evaluate resting orders against current marks."""
        with self._lock:
            for order in list(self._orders.values()):
                if not order.status.is_terminal:
                    self._try_fill(order)

    # -- fill engine ------------------------------------------------------------

    def _try_fill(self, order: Order) -> None:
        mark = self._marks.get(order.symbol)
        if mark is None or order.status.is_terminal:
            return

        fill_price: Optional[float] = None
        if order.order_type is OrderType.MARKET:
            slip = mark * self._slippage
            fill_price = mark + slip if order.side.is_buy else mark - slip
        elif order.order_type is OrderType.LIMIT and order.limit_price is not None:
            if order.side.is_buy and mark <= order.limit_price:
                fill_price = min(mark, order.limit_price)
            elif not order.side.is_buy and mark >= order.limit_price:
                fill_price = max(mark, order.limit_price)
        elif order.order_type is OrderType.STOP and order.stop_price is not None:
            triggered = (mark >= order.stop_price if order.side.is_buy
                         else mark <= order.stop_price)
            if triggered:
                slip = mark * self._slippage
                fill_price = mark + slip if order.side.is_buy else mark - slip
        if fill_price is None:
            return

        qty = order.remaining_qty
        if self._partial_fill_qty is not None:
            qty = min(qty, self._partial_fill_qty)
        if qty <= 0:
            return
        self._apply_fill(order, qty, round(fill_price, 4))

    def _apply_fill(self, order: Order, qty: float, price: float) -> None:
        prior = order.filled_qty
        order.avg_fill_price = ((order.avg_fill_price * prior + price * qty)
                                / (prior + qty))
        order.filled_qty = prior + qty
        order.status = (OrderStatus.FILLED if order.remaining_qty <= 1e-9
                        else OrderStatus.PARTIALLY_FILLED)
        order.updated_at = utcnow()

        signed = qty if order.side.is_buy else -qty
        self._cash -= signed * price * order.multiplier

        pos = self._positions.get(order.symbol)
        if pos is None:
            pos = BrokerPosition(symbol=order.symbol, qty=0.0, avg_entry_price=0.0,
                                 mark=price, multiplier=order.multiplier)
            self._positions[order.symbol] = pos
        new_qty = pos.qty + signed
        if pos.qty == 0 or (pos.qty > 0) == (signed > 0):  # opening / adding
            total = abs(pos.qty) + qty
            pos.avg_entry_price = ((pos.avg_entry_price * abs(pos.qty) + price * qty)
                                   / total)
        pos.qty = new_qty
        pos.mark = price
        if abs(pos.qty) < 1e-9:
            del self._positions[order.symbol]
        log.debug("paper fill: %s %s %.0f @ %.4f (%s)", order.side.value,
                  order.symbol, qty, price, order.status.value)
