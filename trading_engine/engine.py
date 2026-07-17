"""Engine orchestration: data -> filters -> strategies -> risk -> execution.

Two scheduled jobs drive the system:
  * scan    (default 60s): evaluate strategies per ticker, size and place entries
  * manage  (default 10s): refresh marks, absorb fills, enforce stops/targets,
                           EOD flatten, daily-loss auto-shutoff

Both respect the RiskManager kill switch and the market-hours filter.
"""

from __future__ import annotations

import logging
import time as _time
from datetime import datetime, timedelta, timezone
from typing import Optional

from .config import Config
from .data.indicators import compute_indicators
from .data.market_data import YFinanceMarketData
from .data.options_data import TradierOptionsData, YFinanceOptionsData
from .data.pricing import OptionMarkModel
from .execution.base import Broker
from .execution.order_manager import OrderManager
from .execution.paper import PaperBroker
from .filters import SignalFilters
from .models import Instrument, Order, Signal, TradeRecord, position_side_label
from .risk.manager import RiskManager
from .risk.position_sizing import PositionSizer
from .storage.trade_log import TradeLog
from .strategies import MarketSnapshot, MomentumBreakoutStrategy, OptionsFlowStrategy, Strategy

log = logging.getLogger(__name__)


def build_broker(config: Config) -> Broker:
    kind = config.execution.broker.lower()
    if kind == "paper":
        return PaperBroker(starting_cash=config.execution.paper.starting_cash,
                           slippage_bps=config.execution.paper.slippage_bps)
    if kind == "alpaca":
        from .execution.alpaca import AlpacaBroker
        return AlpacaBroker(config.execution.alpaca)
    if kind == "tradier":
        from .execution.tradier import TradierBroker
        return TradierBroker(config.execution.tradier)
    raise ValueError(f"unknown broker '{config.execution.broker}'")


class TradingEngine:
    def __init__(self, config: Optional[Config] = None,
                 broker: Optional[Broker] = None,
                 market_data=None, options_data=None,
                 earnings_lookup=None, trade_log: Optional[TradeLog] = None) -> None:
        self.config = config or Config()
        cfg = self.config

        self.market_data = market_data or YFinanceMarketData(
            intraday_cache_seconds=cfg.data.intraday_cache_seconds,
            daily_cache_seconds=cfg.data.daily_cache_seconds,
            quote_cache_seconds=cfg.data.quote_cache_seconds,
        )
        self.broker = broker or build_broker(cfg)
        self.options_data = options_data or self._build_options_data()
        self.filters = SignalFilters(
            cfg.filters,
            earnings_lookup or self.market_data.get_earnings_dates,
        )
        self.trade_log = trade_log or TradeLog(cfg.engine.db_path, cfg.engine.csv_dir)
        self.risk = RiskManager(cfg.risk)
        self.sizer = PositionSizer(cfg.risk)
        self.orders = OrderManager(
            self.broker, cfg.execution, cfg.risk,
            on_trade_closed=self._on_trade_closed,
            on_order_update=self.trade_log.log_order,
        )

        self.strategies: list[Strategy] = []
        if cfg.strategies.momentum_breakout.enabled:
            self.strategies.append(MomentumBreakoutStrategy(cfg.strategies.momentum_breakout))
        if cfg.strategies.options_flow.enabled:
            self.strategies.append(OptionsFlowStrategy(cfg.strategies.options_flow))

        self._cooldowns: dict[tuple[str, str], datetime] = {}
        self._option_meta: dict[str, OptionMarkModel] = {}  # occ -> calibrated mark model
        self._flattened_session = None
        self._halt_flattened = False
        self._last_equity_log = 0.0
        self._scheduler = None

    # -- wiring helpers ------------------------------------------------------

    def _build_options_data(self):
        cfg = self.config
        needs_chain = (cfg.strategies.options_flow.enabled
                       or cfg.execution.trade_options)
        if not needs_chain:
            return None
        if cfg.data.options_provider == "tradier":
            try:
                from .execution.tradier import TradierClient
                client = TradierClient(cfg.execution.tradier)
                return TradierOptionsData(client, cfg.data.max_chain_expiries,
                                          cfg.data.chain_cache_seconds)
            except Exception as exc:
                log.warning("tradier options data unavailable (%s); using yfinance", exc)
        return YFinanceOptionsData(cfg.data.max_chain_expiries,
                                   cfg.data.chain_cache_seconds,
                                   market_data=self.market_data)

    def _on_trade_closed(self, record: TradeRecord) -> None:
        self.trade_log.log_trade(record)
        self.trade_log.set_signal_status(record.signal_id, "closed")
        self.risk.on_trade_closed(record.pnl)
        self._option_meta.pop(record.symbol, None)

    # -- scan cycle -----------------------------------------------------------

    def scan_once(self, now: Optional[datetime] = None,
                  execute: bool = True) -> list[Signal]:
        """One full strategy pass over the ticker list. Returns generated signals."""
        now = now or datetime.now(timezone.utc)
        self._ensure_session(now)

        session_ok, session_reason = self.filters.check_session(now)
        if execute and not session_ok:
            log.debug("scan skipped: %s", session_reason)
            return []

        signals: list[Signal] = []
        for symbol in self.config.engine.tickers:
            try:
                signals.extend(self._scan_symbol(symbol, now, execute))
            except Exception:
                log.exception("scan failed for %s", symbol)
        return signals

    def _scan_symbol(self, symbol: str, now: datetime, execute: bool) -> list[Signal]:
        cfg = self.config
        daily = self.market_data.get_daily(symbol, cfg.data.daily_lookback_days)
        ok, reason = self.filters.check_symbol(symbol, daily,
                                               self.filters.market_hours.session_date(now))
        if not ok:
            log.debug("%s filtered: %s", symbol, reason)
            return []

        intraday = self.market_data.get_intraday(
            symbol, cfg.data.intraday_interval, cfg.data.intraday_lookback_days)
        if intraday.empty:
            log.debug("%s: no intraday data", symbol)
            return []
        enriched = compute_indicators(intraday)
        chain = self.options_data.get_chain(symbol) if self.options_data else None
        snapshot = MarketSnapshot(symbol=symbol, intraday=enriched, daily=daily,
                                  chain=chain, now=now)

        out: list[Signal] = []
        for strategy in self.strategies:
            if self._in_cooldown(symbol, strategy.name, now):
                continue
            signal = strategy.evaluate(snapshot)
            if signal is None:
                continue
            self._set_cooldown(symbol, strategy.name, now)
            self.trade_log.log_signal(signal)
            log.info("signal: %s %s %s entry %.2f stop %.2f target %.2f (%s)",
                     signal.strategy, signal.direction.value, symbol,
                     signal.entry_price, signal.stop_loss, signal.target_price,
                     signal.expiry_recommendation)
            out.append(signal)
            if execute:
                self._execute_signal(signal)
        return out

    def _in_cooldown(self, symbol: str, strategy: str, now: datetime) -> bool:
        until = self._cooldowns.get((symbol, strategy))
        return until is not None and now < until

    def _set_cooldown(self, symbol: str, strategy: str, now: datetime) -> None:
        minutes = {
            "momentum_breakout": self.config.strategies.momentum_breakout.cooldown_minutes,
            "options_flow": self.config.strategies.options_flow.cooldown_minutes,
        }.get(strategy, 30)
        self._cooldowns[(symbol, strategy)] = now + timedelta(minutes=minutes)

    # -- execution path ------------------------------------------------------------

    def _execute_signal(self, signal: Signal) -> None:
        if self.orders.has_open_exposure(signal.symbol):
            self.trade_log.set_signal_status(signal.id, "skipped:existing_exposure")
            return
        ok, reason = self.risk.can_open(len(self.orders.open_positions()))
        if not ok:
            self.trade_log.set_signal_status(signal.id, f"blocked:{reason}")
            log.info("signal %s blocked: %s", signal.id, reason)
            return

        premium = None
        if signal.contract is not None:
            mid = signal.contract.mid
            premium = mid if mid == mid and mid > 0 else None
        min_premium = self.config.risk.min_option_premium
        if premium is not None and premium < min_premium:
            log.info("signal %s: contract premium $%.2f below minimum $%.2f, "
                     "expressing as equity instead", signal.id, premium, min_premium)
            premium = None
        if not (self.config.execution.trade_options and premium):
            signal.instrument = Instrument.EQUITY  # express as shares instead

        account = self.broker.get_account()
        stats = (self.trade_log.stats()
                 if self.config.risk.sizing_method == "kelly" else None)
        sizing = self.sizer.size(signal, account.equity, premium, stats)
        if not sizing.viable:
            self.trade_log.set_signal_status(
                signal.id, f"sized_zero:{sizing.reason or 'unknown'}")
            log.info("signal %s sized to zero [%s]: %s", signal.id,
                     sizing.reason or "unknown", "; ".join(sizing.notes))
            return

        self._prime_paper_marks(signal, premium)
        order = self.orders.execute_signal(signal, sizing.qty, premium)
        if order is None:
            self.trade_log.set_signal_status(signal.id, "not_submitted")
            return
        if order.status.value == "rejected":
            self.trade_log.set_signal_status(signal.id, "rejected_by_broker")
            return
        if signal.instrument is Instrument.OPTION and signal.contract is not None:
            contract = signal.contract
            self._option_meta[contract.occ] = OptionMarkModel.calibrate(
                option_type=signal.option_type,
                strike=contract.strike,
                expiry=contract.expiry,
                entry_spot=signal.entry_price,
                entry_premium=premium,
            )
        self.trade_log.set_signal_status(
            signal.id, f"executed:{sizing.qty}@{sizing.method}:{sizing.risk_pct:.3%}")
        log.info("executed signal %s: qty %d, risk $%.2f (%.2f%% %s)%s", signal.id,
                 sizing.qty, sizing.risk_dollars, sizing.risk_pct * 100, sizing.method,
                 f" [{'; '.join(sizing.notes)}]" if sizing.notes else "")

    def _prime_paper_marks(self, signal: Signal, premium: Optional[float]) -> None:
        if not isinstance(self.broker, PaperBroker):
            return
        self.broker.set_mark(signal.symbol, signal.entry_price)
        if signal.instrument is Instrument.OPTION and signal.contract is not None and premium:
            self.broker.set_mark(signal.contract.occ, premium)

    # -- manage cycle ------------------------------------------------------------

    def manage_once(self, now: Optional[datetime] = None) -> None:
        now = now or datetime.now(timezone.utc)
        self._ensure_session(now)

        underlying_marks: dict[str, float] = {}
        tradable_marks: dict[str, float] = {}
        for pos in self.orders.open_positions():
            spot = self._mark_underlying(pos.underlying, underlying_marks)
            if pos.instrument is Instrument.OPTION:
                mark = self._mark_option(pos.symbol, spot)
                if mark is not None:
                    tradable_marks[pos.symbol] = mark
            elif spot is not None:
                tradable_marks[pos.symbol] = spot
        for order in self.orders.open_orders():
            self._mark_underlying(order.underlying, underlying_marks)

        if isinstance(self.broker, PaperBroker):
            for sym, mark in {**underlying_marks, **tradable_marks}.items():
                self.broker.set_mark(sym, mark)
            self.broker.process()

        self.orders.poll()
        self.orders.manage(underlying_marks, tradable_marks)

        if (self.config.filters.eod_flatten
                and self.filters.market_hours.is_flatten_window(now)):
            session = self.filters.market_hours.session_date(now)
            if self._flattened_session != session and self.orders.open_positions():
                log.info("end-of-day flatten (%s)", self.config.filters.eod_flatten_time)
                self.orders.flatten_all("eod_flatten")
            self._flattened_session = session

        try:
            account = self.broker.get_account()
        except Exception as exc:
            log.warning("get_account failed: %s", exc)
            return
        self.risk.mark_equity(account.equity)
        if self.risk.should_flatten_on_halt and not self._halt_flattened:
            log.warning("daily loss auto-shutoff: flattening all positions")
            self.orders.flatten_all("daily_loss_halt")
            self._halt_flattened = True

        if _time.monotonic() - self._last_equity_log >= 60:
            self.trade_log.log_equity(account.equity, account.cash,
                                      self.risk.realized_pnl_today(), self.risk.halted)
            self._last_equity_log = _time.monotonic()

    def _ensure_session(self, now: datetime) -> None:
        session = self.filters.market_hours.session_date(now)
        if self.risk.day is None or self.risk.day.session != session:
            try:
                equity = self.broker.get_account().equity
            except Exception as exc:
                log.warning("cannot start session, get_account failed: %s", exc)
                return
            self.risk.start_day(session, equity)
            self._halt_flattened = False

    def _mark_underlying(self, symbol: str, cache: dict[str, float]) -> Optional[float]:
        if symbol in cache:
            return cache[symbol]
        price = None
        try:
            price = self.market_data.get_latest_price(symbol)
        except Exception:
            price = None
        if price is None:
            price = self.broker.get_quote(symbol)
        if price is not None:
            cache[symbol] = price
        return price

    def _mark_option(self, occ: str, spot: Optional[float]) -> Optional[float]:
        """Real broker quote when available, else the entry-calibrated model.

        Paper marks are self-set (get_quote would just echo our last mark),
        so paper mode always uses the model: Black-Scholes at the implied vol
        backed out from the entry fill, converging to intrinsic at expiry and
        bounded — see data/pricing.py.
        """
        if not isinstance(self.broker, PaperBroker):
            try:
                quote = self.broker.get_quote(occ)
            except Exception:
                quote = None
            if quote:
                return quote
        model = self._option_meta.get(occ)
        if model is not None and spot:
            return model.mark(spot)
        if isinstance(self.broker, PaperBroker):
            return self.broker.get_quote(occ)  # last known mark
        return None

    # -- status / loop -------------------------------------------------------------

    def get_status(self) -> dict:
        equity_val = None
        try:
            account = self.broker.get_account()
            equity_val = account.equity
            account_d = {"equity": round(account.equity, 2),
                         "cash": round(account.cash, 2),
                         "buying_power": round(account.buying_power, 2)}
        except Exception as exc:
            account_d = {"error": str(exc)}
        day = self.risk.day
        positions = []
        for p in self.orders.open_positions():
            positions.append({
                "symbol": p.symbol, "underlying": p.underlying,
                "instrument": p.instrument.value, "direction": p.direction.value,
                "side": position_side_label(p.instrument, p.direction),
                "strategy": p.strategy, "qty": p.qty,
                "avg_entry": round(p.avg_entry, 4), "mark": round(p.mark, 4),
                "underlying_mark": round(p.underlying_mark, 4),
                "stop_loss": p.stop_loss, "target": p.target,
                "unrealized_pnl": round(p.unrealized_pnl, 2),
                "entry_time": p.entry_time.isoformat(),
            })
        open_orders = [{
            "id": o.id, "symbol": o.symbol, "side": o.side.value,
            "qty": o.qty, "filled_qty": o.filled_qty, "type": o.order_type.value,
            "status": o.status.value, "note": o.note,
        } for o in self.orders.open_orders()]
        return {
            "time": datetime.now(timezone.utc).isoformat(),
            "broker": self.broker.name,
            "tickers": self.config.engine.tickers,
            "market_open": self.filters.market_hours.is_open(),
            "halted": self.risk.halted,
            "halt_reason": self.risk.halt_reason,
            "account": account_d,
            "day": {
                "session": day.session.isoformat() if day else None,
                "start_equity": round(day.start_equity, 2) if day else None,
                "realized_pnl": round(day.realized_pnl, 2) if day else 0.0,
                # total = realized + unrealized, from mark-to-market equity
                "total_pnl": (round(equity_val - day.start_equity, 2)
                              if day and equity_val is not None else None),
                "max_daily_loss": round(self.risk.max_daily_loss(), 2),
            },
            "positions": positions,
            "open_orders": open_orders,
            "signals": self.trade_log.recent_signals(10),
            "trades": self.trade_log.recent_trades(10),
            "stats": self.trade_log.stats(),
        }

    def run(self, blocking: bool = True):
        """Start the scheduler loop. Returns the scheduler when non-blocking."""
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.schedulers.blocking import BlockingScheduler

        cls = BlockingScheduler if blocking else BackgroundScheduler
        scheduler = cls(timezone="UTC")
        scheduler.add_job(self.scan_once, "interval",
                          seconds=self.config.engine.scan_interval_seconds,
                          id="scan", coalesce=True, max_instances=1,
                          misfire_grace_time=30)
        scheduler.add_job(self.manage_once, "interval",
                          seconds=self.config.engine.manage_interval_seconds,
                          id="manage", coalesce=True, max_instances=1,
                          misfire_grace_time=30)
        self._scheduler = scheduler
        log.info("engine starting: broker=%s tickers=%s scan=%ss manage=%ss",
                 self.broker.name, ",".join(self.config.engine.tickers),
                 self.config.engine.scan_interval_seconds,
                 self.config.engine.manage_interval_seconds)
        if blocking:
            try:
                scheduler.start()
            except (KeyboardInterrupt, SystemExit):
                log.info("engine stopping")
                scheduler.shutdown(wait=False)
        else:
            scheduler.start()
        return scheduler

    def shutdown(self) -> None:
        if self._scheduler is not None:
            try:
                self._scheduler.shutdown(wait=False)
            except Exception:
                pass
        self.trade_log.close()
