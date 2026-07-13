"""Persistent journal: signals, orders, closed trades, equity curve.

SQLite is the source of truth (good for backtesting review via SQL);
trades and signals are also mirrored to CSV files for spreadsheet use.
Writes are serialized with a lock so scheduler threads can share one
instance.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..models import Order, Signal, TradeRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    direction TEXT NOT NULL,
    instrument TEXT NOT NULL,
    entry_price REAL, stop_loss REAL, target_price REAL,
    confidence REAL,
    expiry_recommendation TEXT,
    contract TEXT,
    reasons TEXT,
    status TEXT DEFAULT 'generated'
);
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,
    updated_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    underlying TEXT,
    side TEXT, qty REAL, order_type TEXT,
    limit_price REAL, stop_price REAL,
    status TEXT, filled_qty REAL, avg_fill_price REAL,
    broker_order_id TEXT, signal_id TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    underlying TEXT,
    strategy TEXT,
    direction TEXT,
    instrument TEXT,
    qty REAL, multiplier INTEGER,
    entry_price REAL, exit_price REAL,
    entry_time TEXT, exit_time TEXT,
    pnl REAL, pnl_pct REAL,
    exit_reason TEXT, signal_id TEXT
);
CREATE TABLE IF NOT EXISTS equity_curve (
    ts TEXT PRIMARY KEY,
    equity REAL, cash REAL, realized_day_pnl REAL, halted INTEGER
);
"""

TRADE_CSV_FIELDS = ["id", "exit_time", "underlying", "symbol", "strategy", "direction",
                    "instrument", "qty", "multiplier", "entry_price", "exit_price",
                    "pnl", "pnl_pct", "exit_reason", "entry_time", "signal_id"]
SIGNAL_CSV_FIELDS = ["id", "created_at", "symbol", "strategy", "direction", "instrument",
                     "entry_price", "stop_loss", "target_price", "confidence",
                     "expiry_recommendation", "contract", "status", "reasons"]


def _iso(dt: Optional[datetime]) -> str:
    return (dt or datetime.now(timezone.utc)).isoformat()


class TradeLog:
    def __init__(self, db_path: str | Path, csv_dir: Optional[str | Path] = None) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_dir = Path(csv_dir) if csv_dir else None
        if self._csv_dir:
            self._csv_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- writes --------------------------------------------------------------

    def log_signal(self, signal: Signal, status: str = "generated") -> None:
        contract = signal.contract.describe() if signal.contract else ""
        row = (signal.id, _iso(signal.created_at), signal.symbol, signal.strategy,
               signal.direction.value, signal.instrument.value, signal.entry_price,
               signal.stop_loss, signal.target_price, signal.confidence,
               signal.expiry_recommendation, contract,
               json.dumps(signal.reasons), status)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO signals VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
            self._conn.commit()
        self._append_csv("signals.csv", SIGNAL_CSV_FIELDS, {
            "id": signal.id, "created_at": _iso(signal.created_at),
            "symbol": signal.symbol, "strategy": signal.strategy,
            "direction": signal.direction.value, "instrument": signal.instrument.value,
            "entry_price": signal.entry_price, "stop_loss": signal.stop_loss,
            "target_price": signal.target_price, "confidence": signal.confidence,
            "expiry_recommendation": signal.expiry_recommendation,
            "contract": contract, "status": status,
            "reasons": " | ".join(signal.reasons),
        })

    def set_signal_status(self, signal_id: str, status: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE signals SET status=? WHERE id=?", (status, signal_id))
            self._conn.commit()

    def log_order(self, order: Order) -> None:
        row = (order.id, _iso(order.updated_at or order.submitted_at), order.symbol,
               order.underlying, order.side.value, order.qty, order.order_type.value,
               order.limit_price, order.stop_price, order.status.value,
               order.filled_qty, order.avg_fill_price, order.broker_order_id,
               order.signal_id, order.note)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
            self._conn.commit()

    def log_trade(self, trade: TradeRecord) -> None:
        row = (trade.id, trade.symbol, trade.underlying, trade.strategy,
               trade.direction.value, trade.instrument.value, trade.qty,
               trade.multiplier, trade.entry_price, trade.exit_price,
               _iso(trade.entry_time), _iso(trade.exit_time), trade.pnl,
               round(trade.pnl_pct, 4), trade.exit_reason, trade.signal_id)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
            self._conn.commit()
        self._append_csv("trades.csv", TRADE_CSV_FIELDS, {
            "id": trade.id, "exit_time": _iso(trade.exit_time),
            "underlying": trade.underlying, "symbol": trade.symbol,
            "strategy": trade.strategy, "direction": trade.direction.value,
            "instrument": trade.instrument.value, "qty": trade.qty,
            "multiplier": trade.multiplier, "entry_price": trade.entry_price,
            "exit_price": trade.exit_price, "pnl": trade.pnl,
            "pnl_pct": round(trade.pnl_pct, 4), "exit_reason": trade.exit_reason,
            "entry_time": _iso(trade.entry_time), "signal_id": trade.signal_id,
        })

    def log_equity(self, equity: float, cash: float, realized_day_pnl: float,
                   halted: bool) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO equity_curve VALUES (?,?,?,?,?)",
                (_iso(None), equity, cash, realized_day_pnl, int(halted)))
            self._conn.commit()

    # -- reads ---------------------------------------------------------------------

    def recent_signals(self, limit: int = 20) -> list[dict]:
        return self._rows("SELECT * FROM signals ORDER BY created_at DESC LIMIT ?", (limit,))

    def recent_trades(self, limit: int = 50) -> list[dict]:
        return self._rows("SELECT * FROM trades ORDER BY exit_time DESC LIMIT ?", (limit,))

    def equity_curve(self, limit: int = 500) -> list[dict]:
        rows = self._rows("SELECT * FROM equity_curve ORDER BY ts DESC LIMIT ?", (limit,))
        return list(reversed(rows))

    def stats(self) -> dict:
        """Aggregate performance for the dashboard and Kelly sizing."""
        trades = self._rows("SELECT pnl, strategy FROM trades")
        total = len(trades)
        wins = [t["pnl"] for t in trades if t["pnl"] > 0]
        losses = [-t["pnl"] for t in trades if t["pnl"] < 0]
        gross_win = sum(wins)
        gross_loss = sum(losses)
        by_strategy: dict[str, dict] = {}
        for t in trades:
            s = by_strategy.setdefault(t["strategy"], {"trades": 0, "wins": 0, "pnl": 0.0})
            s["trades"] += 1
            s["wins"] += 1 if t["pnl"] > 0 else 0
            s["pnl"] += t["pnl"]
        return {
            "total_trades": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / total) if total else 0.0,
            "total_pnl": round(gross_win - gross_loss, 2),
            "avg_win": (gross_win / len(wins)) if wins else 0.0,
            "avg_loss": (gross_loss / len(losses)) if losses else 0.0,
            "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win else 0.0,
            "by_strategy": by_strategy,
        }

    def _rows(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    # -- csv mirror -------------------------------------------------------------

    def _append_csv(self, filename: str, fields: list[str], row: dict) -> None:
        if self._csv_dir is None:
            return
        path = self._csv_dir / filename
        new_file = not path.exists()
        with self._lock, path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
