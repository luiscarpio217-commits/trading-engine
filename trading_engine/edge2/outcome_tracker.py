"""
EDGE2 patch 2 — Post-flag outcome tracker
=========================================
Answers the question the site currently can't: "what happened AFTER the flag?"
Stores price 30 min and 60 min after each flag, plus the post-flag peak,
so the history page can show whether the scanner catches moves early.

Self-contained: creates its own table, doesn't touch the existing schema.

Wired in three places:
1. scanner.scan_universe() calls record_flag() the first time a ticker is
   flagged each day.
2. main.scanner_loop() calls update_open_outcomes() every ~5 minutes.
3. The /today route joins outcomes_for_history() into the history table and
   shows summary_stats() as a credibility line.
"""

import sqlite3
import time

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS flag_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    flag_ts INTEGER NOT NULL,           -- unix time of the flag
    flag_price REAL NOT NULL,
    price_30m REAL,
    price_60m REAL,
    peak_price REAL,                    -- highest price seen after flag
    last_checked INTEGER,
    UNIQUE(ticker, flag_ts)
);
"""


def _conn(db_path: str) -> sqlite3.Connection:
    c = sqlite3.connect(db_path)
    c.execute(TABLE_SQL)
    return c


def record_flag(db_path: str, ticker: str, flag_price: float) -> None:
    """Call once when the scanner creates a new flag."""
    with _conn(db_path) as c:
        c.execute(
            "INSERT OR IGNORE INTO flag_outcomes "
            "(ticker, flag_ts, flag_price, peak_price) VALUES (?,?,?,?)",
            (ticker.upper(), int(time.time()), flag_price, flag_price),
        )


def update_open_outcomes(db_path: str, get_price, max_age_hours: int = 7) -> int:
    """
    Fill in 30m/60m snapshots and track the post-flag peak for flags
    younger than max_age_hours. `get_price(ticker) -> float | None`
    is your existing price fetcher. Returns number of rows updated.

    The peak keeps updating for the full max_age_hours window; the 30m/60m
    snapshots are only stamped near their time windows so a late fetch
    (downtime, rate limits) never backfills stale data as a 30m/60m return.
    """
    now = int(time.time())
    cutoff = now - max_age_hours * 3600
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT id, ticker, flag_ts, flag_price, price_30m, price_60m, peak_price "
            "FROM flag_outcomes WHERE flag_ts > ?",
            (cutoff,),
        ).fetchall()

    # Fetch prices BEFORE opening a write transaction so slow/failed network
    # calls can never hold a SQLite write lock.
    prices = {}
    for _rid, ticker, *_rest in rows:
        if ticker not in prices:
            try:
                prices[ticker] = get_price(ticker)
            except Exception:
                prices[ticker] = None

    updated = 0
    with _conn(db_path) as c:
        for rid, ticker, flag_ts, flag_price, p30, p60, peak in rows:
            price = prices.get(ticker)
            if price is None:
                continue

            age = now - flag_ts
            # Stamp snapshots only within a 15-min grace window past their mark
            new_p30 = p30 if p30 is not None else (price if 1800 <= age <= 2700 else None)
            new_p60 = p60 if p60 is not None else (price if 3600 <= age <= 4500 else None)
            new_peak = max(peak or flag_price, price)
            c.execute(
                "UPDATE flag_outcomes SET price_30m=?, price_60m=?, "
                "peak_price=?, last_checked=? WHERE id=?",
                (new_p30, new_p60, new_peak, now, rid),
            )
            updated += 1
    return updated


def outcomes_for_history(db_path: str) -> list[dict]:
    """Rows ready to render on the history page, newest first."""
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT ticker, flag_ts, flag_price, price_30m, price_60m, peak_price "
            "FROM flag_outcomes ORDER BY flag_ts DESC"
        ).fetchall()
    out = []
    for ticker, ts, fp, p30, p60, peak in rows:
        def ret(p):
            return round((p - fp) / fp * 100, 1) if (p and fp) else None
        out.append({
            "ticker": ticker,
            "flag_ts": ts,
            "flag_price": fp,
            "ret_30m": ret(p30),
            "ret_60m": ret(p60),
            "peak_ret": ret(peak),
        })
    return out


def summary_stats(db_path: str) -> dict:
    """One-line credibility stats for the top of the history page."""
    rows = [r for r in outcomes_for_history(db_path) if r["ret_30m"] is not None]
    if not rows:
        return {"n": 0}
    n = len(rows)
    winners = sum(1 for r in rows if (r["peak_ret"] or 0) > 0)
    avg30 = sum(r["ret_30m"] for r in rows) / n
    return {
        "n": n,
        "pct_higher_after_flag": round(winners / n * 100, 1),
        "avg_ret_30m": round(avg30, 2),
    }
