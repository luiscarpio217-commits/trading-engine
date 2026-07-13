"""Tradier adapter: REST client (market data + trading) and Broker impl.

`TradierClient` wraps the plain REST endpoints and is shared by
`TradierBroker` (execution) and `TradierOptionsData` (chains with greeks).
Credentials come from TRADIER_ACCESS_TOKEN / TRADIER_ACCOUNT_ID (see
`TradierSettings`); `sandbox: true` targets sandbox.tradier.com.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import requests

from ..config import TradierSettings
from ..models import (AccountInfo, BrokerPosition, Instrument, Order, OrderSide,
                      OrderStatus, OrderType, utcnow)
from .base import Broker

log = logging.getLogger(__name__)

_STATUS_MAP = {
    "pending": OrderStatus.NEW,
    "open": OrderStatus.NEW,
    "submitted": OrderStatus.NEW,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "expired": OrderStatus.CANCELED,
    "rejected": OrderStatus.REJECTED,
    "error": OrderStatus.REJECTED,
}

_EQUITY_SIDES = {
    OrderSide.BUY: "buy",
    OrderSide.SELL: "sell",
    OrderSide.SELL_SHORT: "sell_short",
    OrderSide.BUY_TO_COVER: "buy_to_cover",
}
_OPTION_SIDES = {
    OrderSide.BUY_TO_OPEN: "buy_to_open",
    OrderSide.SELL_TO_CLOSE: "sell_to_close",
    OrderSide.BUY: "buy_to_open",
    OrderSide.SELL: "sell_to_close",
}


def _as_list(node: Any) -> list:
    """Tradier collapses single-element lists to a bare object."""
    if node is None or node == "null":
        return []
    return node if isinstance(node, list) else [node]


class TradierClient:
    def __init__(self, settings: Optional[TradierSettings] = None,
                 session: Optional[requests.Session] = None) -> None:
        self.s = settings or TradierSettings()
        if not self.s.token:
            raise RuntimeError("Tradier credentials missing: set TRADIER_ACCESS_TOKEN "
                               "(and TRADIER_ACCOUNT_ID for trading)")
        self._http = session or requests.Session()
        self._http.headers.update({
            "Authorization": f"Bearer {self.s.token}",
            "Accept": "application/json",
        })

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        r = self._http.get(f"{self.s.base_url}{path}", params=params or {}, timeout=15)
        r.raise_for_status()
        return r.json() or {}

    # -- market data -------------------------------------------------------

    def get_quote(self, symbol: str) -> Optional[float]:
        data = self._get("/markets/quotes", {"symbols": symbol})
        quotes = _as_list((data.get("quotes") or {}).get("quote"))
        if not quotes:
            return None
        q = quotes[0]
        last = q.get("last") or q.get("close") or q.get("prevclose")
        return float(last) if last is not None else None

    def get_option_expirations(self, symbol: str) -> list[str]:
        data = self._get("/markets/options/expirations",
                         {"symbol": symbol, "includeAllRoots": "true"})
        return [str(d) for d in _as_list((data.get("expirations") or {}).get("date"))]

    def get_option_chain(self, symbol: str, expiration: str) -> list[dict]:
        data = self._get("/markets/options/chains",
                         {"symbol": symbol, "expiration": expiration, "greeks": "true"})
        return _as_list((data.get("options") or {}).get("option"))

    # -- account / trading -------------------------------------------------

    def _account_path(self, suffix: str = "") -> str:
        if not self.s.account_id:
            raise RuntimeError("TRADIER_ACCOUNT_ID not set")
        return f"/accounts/{self.s.account_id}{suffix}"

    def get_balances(self) -> dict:
        return (self._get(self._account_path("/balances")).get("balances")) or {}

    def get_positions(self) -> list[dict]:
        data = self._get(self._account_path("/positions"))
        return _as_list((data.get("positions") or {}).get("position"))

    def get_order(self, order_id: str) -> dict:
        data = self._get(self._account_path(f"/orders/{order_id}"))
        return data.get("order") or {}

    def place_order(self, payload: dict) -> dict:
        r = self._http.post(f"{self.s.base_url}{self._account_path('/orders')}",
                            data=payload, timeout=15)
        body = {}
        try:
            body = r.json() or {}
        except ValueError:
            pass
        if r.status_code >= 400:
            return {"error": body.get("errors", {}).get("error") or r.text[:300],
                    "status_code": r.status_code}
        return body.get("order") or {}

    def cancel_order(self, order_id: str) -> None:
        r = self._http.delete(f"{self.s.base_url}{self._account_path(f'/orders/{order_id}')}",
                              timeout=15)
        if r.status_code >= 400 and r.status_code != 404:
            log.warning("tradier cancel %s -> %s %s", order_id, r.status_code, r.text[:200])


class TradierBroker(Broker):
    name = "tradier"

    def __init__(self, settings: Optional[TradierSettings] = None,
                 client: Optional[TradierClient] = None) -> None:
        self.client = client or TradierClient(settings)

    def get_account(self) -> AccountInfo:
        b = self.client.get_balances()
        equity = float(b.get("total_equity") or 0.0)
        cash = float(b.get("total_cash") or 0.0)
        margin = b.get("margin") or {}
        cash_acct = b.get("cash") or {}
        buying_power = float(margin.get("stock_buying_power")
                             or cash_acct.get("cash_available") or cash)
        return AccountInfo(equity=equity, cash=cash, buying_power=buying_power)

    def submit_order(self, order: Order) -> Order:
        if order.instrument is Instrument.OPTION:
            payload = {
                "class": "option",
                "symbol": order.underlying,
                "option_symbol": order.symbol,
                "side": _OPTION_SIDES[order.side],
            }
        else:
            payload = {
                "class": "equity",
                "symbol": order.symbol,
                "side": _EQUITY_SIDES[order.side],
            }
        payload.update({
            "quantity": str(int(order.qty)),
            "type": order.order_type.value,
            "duration": "day",
        })
        if order.order_type is OrderType.LIMIT:
            payload["price"] = f"{order.limit_price:.2f}"
        if order.order_type is OrderType.STOP:
            payload["stop"] = f"{order.stop_price:.2f}"

        result = self.client.place_order(payload)
        order.submitted_at = utcnow()
        if "error" in result or not result.get("id"):
            order.status = OrderStatus.REJECTED
            order.note = (order.note + f" | tradier reject: {result.get('error')}").strip(" |")
            log.error("tradier order rejected (%s %s): %s", order.side.value,
                      order.symbol, result.get("error"))
            return order
        order.broker_order_id = str(result["id"])
        order.status = _STATUS_MAP.get(str(result.get("status", "pending")).lower(),
                                       OrderStatus.NEW)
        return order

    def refresh_order(self, order: Order) -> Order:
        if not order.broker_order_id:
            return order
        body = self.client.get_order(order.broker_order_id)
        if not body:
            return order
        order.status = _STATUS_MAP.get(str(body.get("status", "")).lower(), order.status)
        order.filled_qty = float(body.get("exec_quantity") or 0.0)
        avg = body.get("avg_fill_price")
        if avg not in (None, "", 0, "0.00000"):
            order.avg_fill_price = float(avg)
        order.updated_at = utcnow()
        return order

    def cancel_order(self, order: Order) -> None:
        if order.broker_order_id:
            self.client.cancel_order(order.broker_order_id)

    def list_positions(self) -> list[BrokerPosition]:
        out = []
        for p in self.client.get_positions():
            symbol = p.get("symbol", "")
            qty = float(p.get("quantity", 0.0))
            cost = float(p.get("cost_basis", 0.0))
            multiplier = 100 if len(symbol) > 12 else 1  # OCC symbols are long
            per_unit = (cost / (abs(qty) * multiplier)) if qty else 0.0
            out.append(BrokerPosition(symbol=symbol, qty=qty,
                                      avg_entry_price=per_unit,
                                      multiplier=multiplier))
        return out

    def get_quote(self, symbol: str) -> Optional[float]:
        try:
            return self.client.get_quote(symbol)
        except Exception:
            return None
