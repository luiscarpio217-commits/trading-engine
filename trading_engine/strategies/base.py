"""Strategy interface and the per-symbol market snapshot strategies consume."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from ..data.options_data import OptionsChain
from ..models import Signal


@dataclass
class MarketSnapshot:
    """Everything a strategy may need about one symbol at scan time."""

    symbol: str
    intraday: pd.DataFrame                    # OHLCV enriched by compute_indicators()
    daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    chain: Optional[OptionsChain] = None
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def spot(self) -> float:
        if self.chain is not None and self.chain.spot:
            return self.chain.spot
        return float(self.intraday["close"].iloc[-1])


class Strategy(ABC):
    """A signal generator. Implementations must be pure: no I/O, no orders."""

    name: str = "strategy"

    @abstractmethod
    def evaluate(self, snapshot: MarketSnapshot) -> Optional[Signal]:
        """Return a Signal if the setup triggers on this snapshot, else None."""


def describe_expiry(expiry, asof) -> str:
    dte = (expiry - asof).days
    return f"{expiry.isoformat()} ({dte} DTE)"
