import sqlite3
import json
from datetime import datetime, timedelta

# IronFrost port: default kept for standalone use; the engine points this at
# its own trading.db via set_db_path() before init_db() (one SQLite DB).
DB_PATH = "scanner.db"


def set_db_path(path) -> None:
    """Point the EDGE2 tables at a specific SQLite file (IronFrost's DB)."""
    global DB_PATH
    DB_PATH = str(path)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS flagged_stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            flagged_at TEXT NOT NULL,
            price REAL,
            prev_close REAL,
            gap_percent REAL,
            volume INTEGER,
            avg_volume INTEGER,
            volume_ratio REAL,
            setup_tag TEXT,
            entry_low REAL,
            entry_high REAL,
            stop_loss REAL,
            target_1 REAL,
            target_2 REAL,
            target_3 REAL,
            risk_reward REAL,
            status TEXT DEFAULT 'Active',
            session TEXT,
            news TEXT,
            catalyst_score REAL DEFAULT 0,
            catalyst_category TEXT DEFAULT '',
            catalyst_velocity TEXT DEFAULT ''
        )
    ''')
    # Migrations for older dbs that don't have newer columns yet
    for column, ddl in [
        ("news", "ALTER TABLE flagged_stocks ADD COLUMN news TEXT"),
        ("catalyst_score", "ALTER TABLE flagged_stocks ADD COLUMN catalyst_score REAL DEFAULT 0"),
        ("catalyst_category", "ALTER TABLE flagged_stocks ADD COLUMN catalyst_category TEXT DEFAULT ''"),
        ("catalyst_velocity", "ALTER TABLE flagged_stocks ADD COLUMN catalyst_velocity TEXT DEFAULT ''"),
    ]:
        try:
            c.execute(ddl)
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()

def save_flagged_stock(stock_data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO flagged_stocks 
        (ticker, flagged_at, price, prev_close, gap_percent, volume, avg_volume, 
         volume_ratio, setup_tag, entry_low, entry_high, stop_loss, 
         target_1, target_2, target_3, risk_reward, session, news,
         catalyst_score, catalyst_category, catalyst_velocity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        stock_data['ticker'], datetime.now().isoformat(),
        stock_data.get('price'), stock_data.get('prev_close'),
        stock_data.get('gap_percent'), stock_data.get('volume'),
        stock_data.get('avg_volume'), stock_data.get('volume_ratio'),
        stock_data.get('setup_tag'), stock_data.get('entry_low'),
        stock_data.get('entry_high'), stock_data.get('stop_loss'),
        stock_data.get('target_1'), stock_data.get('target_2'),
        stock_data.get('target_3'), stock_data.get('risk_reward'),
        stock_data.get('session'), stock_data.get('news'),
        stock_data.get('catalyst_score', 0.0),
        stock_data.get('catalyst_category', ''),
        stock_data.get('catalyst_velocity', '')
    ))
    conn.commit()
    conn.close()

def was_flagged_today(ticker):
    """True if this ticker already has at least one flag row today."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT 1 FROM flagged_stocks WHERE ticker = ? AND date(flagged_at) = date('now') LIMIT 1",
        (ticker,)
    )
    row = c.fetchone()
    conn.close()
    return row is not None

def get_flag_history(days=30):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT ticker, date(flagged_at) as scan_date,
               MAX(gap_percent) as gap_percent,
               MAX(price) as price,
               setup_tag, session,
               COUNT(*) as times_flagged
        FROM flagged_stocks
        WHERE date(flagged_at) < date('now')
          AND date(flagged_at) >= date('now', ? || ' days')
        GROUP BY ticker, date(flagged_at)
        ORDER BY scan_date DESC, gap_percent DESC
    """, (f'-{days}',))
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows

def get_active_flags():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT * FROM flagged_stocks 
        WHERE id IN (
            SELECT MAX(id) FROM flagged_stocks
            WHERE date(flagged_at) = date('now')
            GROUP BY ticker
        )
        AND status = 'Active'
        ORDER BY gap_percent DESC
    """)
    rows = []
    for row in c.fetchall():
        d = dict(row)
        try:
            d['news_items'] = json.loads(d.get('news') or '[]')
        except Exception:
            d['news_items'] = []
        rows.append(d)
    conn.close()
    return rows

def get_fresh_flags(fresh_minutes=30):
    """Today's flags whose FIRST sighting was within the last `fresh_minutes`.

    Keeps the front page focused on new movers; older setups age off to the
    Today's Tracker but the latest snapshot per ticker is what we display.
    """
    cutoff = (datetime.now() - timedelta(minutes=fresh_minutes)).isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT * FROM flagged_stocks
        WHERE id IN (
            SELECT MAX(id) FROM flagged_stocks
            WHERE date(flagged_at) = date('now')
            GROUP BY ticker
        )
        AND status = 'Active'
        AND ticker IN (
            SELECT ticker FROM flagged_stocks
            WHERE date(flagged_at) = date('now')
            GROUP BY ticker
            HAVING MIN(flagged_at) >= ?
        )
        ORDER BY gap_percent DESC
    """, (cutoff,))
    rows = []
    for row in c.fetchall():
        d = dict(row)
        try:
            d['news_items'] = json.loads(d.get('news') or '[]')
        except Exception:
            d['news_items'] = []
        rows.append(d)
    conn.close()
    return rows

def get_today_tracker():
    """Every ticker flagged today with its first/last sighting, peak gap,
    flag count, and latest snapshot. The persistent intraday tracking list."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT f.ticker, f.price, f.gap_percent, f.setup_tag, f.session,
               f.status, f.volume_ratio,
               f.catalyst_score, f.catalyst_category, f.catalyst_velocity,
               agg.first_seen, agg.last_seen, agg.times_flagged, agg.peak_gap
        FROM flagged_stocks f
        JOIN (
            SELECT ticker,
                   MIN(flagged_at) AS first_seen,
                   MAX(flagged_at) AS last_seen,
                   COUNT(*) AS times_flagged,
                   MAX(gap_percent) AS peak_gap,
                   MAX(id) AS latest_id
            FROM flagged_stocks
            WHERE date(flagged_at) = date('now')
            GROUP BY ticker
        ) agg ON f.id = agg.latest_id
        ORDER BY f.gap_percent DESC
    """)
    rows = []
    for row in c.fetchall():
        d = dict(row)
        d['first_seen_t'] = (d.get('first_seen') or '')[11:16]
        d['last_seen_t'] = (d.get('last_seen') or '')[11:16]
        rows.append(d)
    conn.close()
    return rows
