"""Alpaca broker adapter (paper or live) via the v2 REST API.

Credentials come from ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY (see
`AlpacaSettings`). Equities use the plain ticker; options use the OCC
symbol (Alpaca supports single-leg options on v2 orders).
"""

from __future__ import annotations

import logging
from typing import Optional

import requests

from ..config import AlpacaSettings
from ..models import (AccountInfo, BrokerPosition, Order, OrderSide, OrderStatus,
                      OrderType, utcnow)
from .base import Broker

log = logging.getLogger(__name__)

_STATUS_MAP = {
    "new": OrderStatus.NEW,
    "accepted": OrderStatus.NEW,
    "pending_new": OrderStatus.NEW,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "expired": OrderStatus.CANCELED,
    "done_for_day": OrderStatus.CANCELED,
    "rejected": OrderStatus.REJECTED,
}

# Alpaca uses plain buy/sell; shorts are sells without a position.
_SIDE_MAP = {
    OrderSide.BUY: "buy",
    OrderSide.BUY_TO_COVER: "buy",
    OrderSide.BUY_TO_OPEN: "buy",
    OrderSide.SELL: "sell",
    OrderSide.SELL_SHORT: "sell",
    OrderSide.SELL_TO_CLOSE: "sell",
}


class AlpacaBroker(Broker):
    name = "alpaca"

    def __init__(self, settings: Optional[AlpacaSettings] = None,
                 session: Optional[requests.Session] = None) -> None:
        self.s = settings or AlpacaSettings()
        if not self.s.key_id or not self.s.secret:
            raise RuntimeError(
                "Alpaca credentials missing: set ALPACA_API_KEY_ID and "
                "ALPACA_API_SECRET_KEY environment variables")
        self._http = session or requests.Session()
        self._http.headers.update({
            "APCA-API-KEY-ID": self.s.key_id,
            "APCA-API-SECRET-KEY": self.s.secret,
            "Accept": "application/json",
        })

    def _url(self, path: str) -> str:
        return f"{self.s.base_url}{path}"

    def get_account(self) -> AccountInfo:
        r = self._http.get(self._url("/v2/account"), timeout=15)
        r.raise_for_status()
        a = r.json()
        return AccountInfo(
            equity=float(a.get("equity", 0.0)),
            cash=float(a.get("cash", 0.0)),
            buying_power=float(a.get("buying_power", 0.0)),
            currency=a.get("currency", "USD"),
        )

    def submit_order(self, order: Order) -> Order:
        payload: dict = {
            "symbol": order.symbol,
            "qty": str(int(order.qty)),
            "side": _SIDE_MAP[order.side],
            "type": order.order_type.value,
            "time_in_force": "day",
        }
        if order.order_type is OrderType.LIMIT:
            payload["limit_price"] = f"{order.limit_price:.2f}"
        if order.order_type is OrderType.STOP:
            payload["stop_price"] = f"{order.stop_price:.2f}"
        r = self._http.post(self._url("/v2/orders"), json=payload, timeout=15)
        order.submitted_at = utcnow()
        if r.status_code >= 400:
            order.status = OrderStatus.REJECTED
            order.note = (order.note + f" | alpaca reject: {r.text[:200]}").strip(" |")
            log.error("alpaca order rejected (%s %s): %s", order.side.value,
                      order.symbol, r.text[:300])
            return order
        body = r.json()
        order.broker_order_id = body.get("id", "")
        self._sync(order, body)
        return order

    def refresh_order(self, order: Order) -> Order:
        if not order.broker_order_id:
            return order
        r = self._http.get(self._url(f"/v2/orders/{order.broker_order_id}"), timeout=15)
        if r.status_code == 404:
            return order
        r.raise_for_status()
        self._sync(order, r.json())
        return order

    def cancel_order(self, order: Order) -> None:
        if not order.broker_order_id:
            return
        r = self._http.delete(self._url(f"/v2/orders/{order.broker_order_id}"), timeout=15)
        if r.status_code not in (204, 404, 422):
            log.warning("alpaca cancel %s -> %s %s", order.broker_order_id,
                        r.status_code, r.text[:200])

    def list_positions(self) -> list[BrokerPosition]:
        r = self._http.get(self._url("/v2/positions"), timeout=15)
        r.raise_for_status()
        out = []
        for p in r.json():
            qty = float(p.get("qty", 0.0))
            out.append(BrokerPosition(
                symbol=p.get("symbol", ""),
                qty=qty,
                avg_entry_price=float(p.get("avg_entry_price", 0.0)),
                mark=float(p.get("current_price") or 0.0),
                multiplier=int(float(p.get("qty_available_multiplier", 1) or 1))
                if "qty_available_multiplier" in p
                else (100 if p.get("asset_class") == "us_option" else 1),
            ))
        return out

    def get_quote(self, symbol: str) -> Optional[float]:
        # Data API host differs from the trading host.
        url = "https://data.alpaca.markets/v2/stocks/trades/latest"
        try:
            r = self._http.get(url, params={"symbols": symbol}, timeout=10)
            r.raise_for_status()
            trade = (r.json().get("trades") or {}).get(symbol)
            return float(trade["p"]) if trade else None
        except Exception:
            return None

    @staticmethod
    def _sync(order: Order, body: dict) -> None:
        order.status = _STATUS_MAP.get(str(body.get("status", "")).lower(),
                                       order.status)
        order.filled_qty = float(body.get("filled_qty") or 0.0)
        avg = body.get("filled_avg_price")
        if avg is not None:
            order.avg_fill_price = float(avg)
        order.updated_at = utcnow()
