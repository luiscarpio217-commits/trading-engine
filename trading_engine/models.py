"""Core domain models shared across the engine."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class Direction(str, Enum):
    """Directional thesis of a signal (long = bullish, short = bearish).

    This is the *thesis*, not the sign of the units held. Option positions
    are always LONG PREMIUM (bought): a SHORT thesis buys puts; the engine
    never sells options short. Only equity positions can hold short units
    (sell_short/buy_to_cover). Use `position_side_label()` when displaying.
    """

    LONG = "long"
    SHORT = "short"

    @property
    def option_type(self) -> "OptionType":
        return OptionType.CALL if self is Direction.LONG else OptionType.PUT

    @property
    def sign(self) -> int:
        return 1 if self is Direction.LONG else -1


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class Instrument(str, Enum):
    EQUITY = "equity"
    OPTION = "option"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"
    SELL_SHORT = "sell_short"
    BUY_TO_COVER = "buy_to_cover"
    BUY_TO_OPEN = "buy_to_open"      # options
    SELL_TO_CLOSE = "sell_to_close"  # options

    @property
    def is_buy(self) -> bool:
        return self in (OrderSide.BUY, OrderSide.BUY_TO_COVER, OrderSide.BUY_TO_OPEN)


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(str, Enum):
    PENDING = "pending"                    # created locally, not yet acknowledged
    NEW = "new"                            # accepted by broker, resting
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"

    @property
    def is_terminal(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED)


def position_side_label(instrument: "Instrument | str", direction: "Direction | str") -> str:
    """Human-readable side for journals/dashboards.

    Options are always bought, so a bearish option position reads
    'long put' — never 'short', which misleads readers into premium-short
    P&L expectations (premium up would look like a loss when it is a gain).
    """
    inst = instrument.value if isinstance(instrument, Instrument) else str(instrument)
    dirn = direction.value if isinstance(direction, Direction) else str(direction)
    if inst == Instrument.OPTION.value:
        return "long call" if dirn == Direction.LONG.value else "long put"
    return "long" if dirn == Direction.LONG.value else "short"


def occ_symbol(underlying: str, expiry: date, option_type: OptionType, strike: float) -> str:
    """Build an OCC option symbol, e.g. AAPL260717C00195000."""
    strike_int = int(round(strike * 1000))
    letter = "C" if option_type is OptionType.CALL else "P"
    return f"{underlying.upper()}{expiry.strftime('%y%m%d')}{letter}{strike_int:08d}"


@dataclass
class OptionContract:
    """A single options contract, normalized across data providers."""

    underlying: str
    option_type: OptionType
    strike: float
    expiry: date
    occ: str = ""
    bid: float = float("nan")
    ask: float = float("nan")
    last: float = float("nan")
    volume: int = 0
    open_interest: int = 0
    iv: float = float("nan")
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.occ:
            self.occ = occ_symbol(self.underlying, self.expiry, self.option_type, self.strike)

    @property
    def mid(self) -> float:
        bid = self.bid if self.bid == self.bid else 0.0  # NaN check
        ask = self.ask if self.ask == self.ask else 0.0
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        if self.last == self.last and self.last > 0:
            return self.last
        return max(bid, ask)

    def describe(self) -> str:
        return (f"{self.underlying} {self.expiry.isoformat()} "
                f"{self.strike:g}{'C' if self.option_type is OptionType.CALL else 'P'}")


@dataclass
class Signal:
    """A trade signal produced by a strategy, before risk sizing."""

    symbol: str
    strategy: str
    direction: Direction
    entry_price: float
    stop_loss: float
    target_price: float
    instrument: Instrument = Instrument.OPTION
    confidence: float = 0.5
    expiry_recommendation: str = ""
    contract: Optional[OptionContract] = None
    reasons: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utcnow)
    id: str = field(default_factory=lambda: new_id("sig"))

    @property
    def option_type(self) -> OptionType:
        return self.direction.option_type

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_risk(self) -> float:
        risk = self.risk_per_share
        return abs(self.target_price - self.entry_price) / risk if risk > 0 else 0.0


@dataclass
class Order:
    """A broker order (equity or single-leg option)."""

    symbol: str                       # tradable symbol: equity ticker or OCC option symbol
    side: OrderSide
    qty: float
    order_type: OrderType = OrderType.MARKET
    instrument: Instrument = Instrument.EQUITY
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    underlying: str = ""              # for options; equals symbol for equities
    multiplier: int = 1               # 100 for options
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    broker_order_id: str = ""
    signal_id: str = ""
    note: str = ""                    # e.g. "entry", "stop_loss", "target", "flatten"
    submitted_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    id: str = field(default_factory=lambda: new_id("ord"))

    def __post_init__(self) -> None:
        if not self.underlying:
            self.underlying = self.symbol

    @property
    def remaining_qty(self) -> float:
        return max(self.qty - self.filled_qty, 0.0)


@dataclass
class Fill:
    order_id: str
    qty: float
    price: float
    timestamp: datetime = field(default_factory=utcnow)


@dataclass
class AccountInfo:
    equity: float
    cash: float
    buying_power: float
    currency: str = "USD"


@dataclass
class BrokerPosition:
    """Position as reported by the broker (reconciliation / dashboard)."""

    symbol: str
    qty: float                        # signed: negative = short
    avg_entry_price: float
    mark: float = 0.0
    multiplier: int = 1

    @property
    def market_value(self) -> float:
        return self.qty * self.mark * self.multiplier

    @property
    def unrealized_pnl(self) -> float:
        return (self.mark - self.avg_entry_price) * self.qty * self.multiplier


@dataclass
class TradeRecord:
    """A completed round-trip trade, for logging and statistics."""

    symbol: str
    underlying: str
    strategy: str
    direction: Direction
    instrument: Instrument
    qty: float
    multiplier: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    exit_reason: str
    signal_id: str = ""
    id: str = field(default_factory=lambda: new_id("trd"))

    @property
    def pnl_pct(self) -> float:
        basis = self.entry_price * self.qty * self.multiplier
        return (self.pnl / basis * 100.0) if basis else 0.0
