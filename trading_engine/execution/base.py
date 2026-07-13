"""Broker interface. Adapters: PaperBroker, AlpacaBroker, TradierBroker."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..models import AccountInfo, BrokerPosition, Order


class Broker(ABC):
    name: str = "broker"

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """Current equity / cash / buying power."""

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        """Submit and return the order with broker_order_id + status set.

        Adapters must set status to REJECTED (not raise) on broker rejects
        so the order manager can handle it uniformly.
        """

    @abstractmethod
    def refresh_order(self, order: Order) -> Order:
        """Update status / filled_qty / avg_fill_price from the broker."""

    @abstractmethod
    def cancel_order(self, order: Order) -> None:
        """Best-effort cancel."""

    @abstractmethod
    def list_positions(self) -> list[BrokerPosition]:
        """Broker-reported open positions (reconciliation / dashboard)."""

    def get_quote(self, symbol: str) -> Optional[float]:
        """Latest trade/mark price if the broker can quote it."""
        return None
