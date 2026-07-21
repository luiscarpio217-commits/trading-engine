import asyncio
import yfinance as yf
from datetime import datetime, time, timedelta
import pytz
import json
from .universe import UNIVERSE, get_dynamic_universe
from .database import save_flagged_stock, was_flagged_today
from . import database as _db
from .catalyst_nlp import CatalystNLPProcessor, Velocity
from .instrument_filter import is_common_stock
from .outcome_tracker import record_flag

_nlp = CatalystNLPProcessor(enable_model=False)  # keyword mode first

# IronFrost seam: set by the engine to bridge.open_paper_trade. When set, the
# first flag of the day per ticker opens a PAPER order from the td dict. A
# bridge failure must never kill the scan loop, hence the guard wrapper.
on_first_flag = None


def _open_ironfrost_paper_trade(td):
    if on_first_flag is None:
        return
    try:
        on_first_flag(td)
    except Exception as e:
        print(f"  [ironfrost] paper-trade bridge failed for {td.get('ticker')}: {e}")

# News cache: ticker -> (fetched_at, news_list). 15-min TTL so auto-refresh
# cycles don't re-hit Yahoo for the same symbol.
_NEWS_CACHE = {}
_NEWS_CACHE_TTL = timedelta(minutes=15)

def get_ticker_news(ticker):
    """Fetch yf.Ticker(ticker).news with a 15-minute per-ticker cache."""
    now = datetime.now()
    cached = _NEWS_CACHE.get(ticker)
    if cached and now - cached[0] < _NEWS_CACHE_TTL:
        return cached[1]
    try:
        news = yf.Ticker(ticker).news or []
    except Exception:
        news = []
    _NEWS_CACHE[ticker] = (now, news)
    return news

def score_catalysts(ticker: str, news=None) -> dict:
    """Sync wrapper: pulls recent headlines, returns best catalyst signal.

    Pass a pre-fetched `news` list (from yf.Ticker(ticker).news) to avoid a
    duplicate network call; if None, it fetches its own.
    """
    try:
        if news is None:
            news = get_ticker_news(ticker)
        headlines = []
        for n in news[:8]:
            title = n["content"].get("title", "") if "content" in n else n.get("title", "")
            if title:
                headlines.append((title, ticker))
        if not headlines:
            return {"catalyst_score": 0.0, "catalyst_category": "none", "catalyst_velocity": "none"}
        results = asyncio.run(_nlp.process_batch(headlines))
        # Negative catalyst anywhere = hard avoid, overrides everything
        neg = [r for r in results if r.is_negative]
        if neg:
            best = max(neg, key=lambda r: r.catalyst_score)
        else:
            best = max(results, key=lambda r: r.catalyst_score)
        return {
            "catalyst_score": best.catalyst_score,
            "catalyst_category": best.category.value,
            "catalyst_velocity": best.velocity.value,
        }
    except Exception as e:
        print(f"catalyst scoring failed for {ticker}: {e}")
        return {"catalyst_score": 0.0, "catalyst_category": "error", "catalyst_velocity": "none"}

# --- Filters ---
PRICE_MIN = 1.0
PRICE_MAX = 30.0
GAP_THRESHOLD = 5.0       # % move from prev close to flag
VOLUME_RATIO_MIN = 1.5    # current vol vs avg

def get_session():
    et = pytz.timezone('US/Eastern')
    now = datetime.now(et).time()
    if time(4, 0) <= now < time(9, 30):
        return 'pre-market'
    elif time(9, 30) <= now < time(16, 0):
        return 'intraday'
    elif time(16, 0) <= now < time(20, 0):
        return 'after-hours'
    return 'closed'

def calculate_setup(td):
    price = td['price']
    prev_close = td['prev_close']
    gap_pct = td['gap_percent']
    vol_ratio = td.get('volume_ratio', 0)

    # Setup tag - rockets first, then standard gappers
    if gap_pct >= 50:
        tag = '🚀 MEGA ROCKET'
    elif gap_pct >= 25:
        tag = '🚀 Rocket'
    elif gap_pct >= 10:
        tag = 'Gap-and-Go'
        if vol_ratio >= 3:
            tag = 'Gap-and-Go (High Vol)'
    elif gap_pct >= 5:
        tag = 'Momentum Breakout'
        if vol_ratio >= 2:
            tag = 'Momentum Breakout (Confirmed)'
    elif gap_pct <= -5:
        tag = 'Gap Down Reversal'
    else:
        tag = 'Watchlist'

    entry_low = round(price * 0.99, 2)
    entry_high = round(price * 1.01, 2)
    stop_loss = round(min(prev_close, price * 0.97), 2)

    risk = entry_low - stop_loss
    target_1 = round(entry_low + risk * 1.5, 2) if risk > 0 else round(price * 1.03, 2)
    target_2 = round(entry_low + risk * 3.0, 2) if risk > 0 else round(price * 1.06, 2)
    target_3 = round(entry_low + risk * 5.0, 2) if risk > 0 else round(price * 1.10, 2)
    rr = round((target_2 - entry_low) / risk, 2) if risk > 0 else 0

    return {
        'setup_tag': tag,
        'entry_low': entry_low,
        'entry_high': entry_high,
        'stop_loss': stop_loss,
        'target_1': target_1,
        'target_2': target_2,
        'target_3': target_3,
        'risk_reward': rr,
    }

def scan_universe():
    raw_universe = get_dynamic_universe()
    # Instrument filter: drop preferred shares, warrants, rights, and units
    # (AHT-PD, RVMDW, ...) before they can ever be flagged.
    universe = [t for t in raw_universe if is_common_stock(t)[0]]
    skipped = len(raw_universe) - len(universe)
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanning {len(universe)} tickers"
          + (f" ({skipped} non-common skipped)..." if skipped else "..."))
    session = get_session()

    try:
        data = yf.download(
            tickers=' '.join(universe),
            period='5d',
            interval='1d',
            group_by='ticker',
            progress=False,
            threads=True,
        )
    except Exception as e:
        print(f"Download error: {e}")
        return []

    flagged = []
    for ticker in universe:
        try:
            if ticker not in data.columns.get_level_values(0):
                continue
            df = data[ticker].dropna()
            if len(df) < 2:
                continue

            current = df.iloc[-1]
            prev = df.iloc[-2]
            price = float(current['Close'])
            prev_close = float(prev['Close'])
            volume = int(current['Volume'])
            avg_volume = int(df['Volume'].mean())

            if not (PRICE_MIN <= price <= PRICE_MAX):
                continue

            gap_pct = ((price - prev_close) / prev_close) * 100
            if abs(gap_pct) < GAP_THRESHOLD:
                continue

            vol_ratio = volume / avg_volume if avg_volume > 0 else 0
            if vol_ratio < VOLUME_RATIO_MIN:
                continue

            td = {
                'ticker': ticker,
                'price': round(price, 2),
                'prev_close': round(prev_close, 2),
                'gap_percent': round(gap_pct, 2),
                'volume': volume,
                'avg_volume': avg_volume,
                'volume_ratio': round(vol_ratio, 2),
                'session': session,
            }
            td.update(calculate_setup(td))
            raw_news = get_ticker_news(ticker)   # cached 15 min, shared below
            td['news'] = fetch_news(ticker, news=raw_news)
            td.update(score_catalysts(ticker, news=raw_news))

            first_flag_today = not was_flagged_today(ticker)
            flagged.append(td)
            save_flagged_stock(td)
            if first_flag_today:
                # Start outcome tracking from the first sighting of the day
                record_flag(_db.DB_PATH, ticker, td['price'])
                _open_ironfrost_paper_trade(td)

            print(f"  FLAGGED {ticker}: ${price} | {gap_pct:+.1f}% | Vol {vol_ratio:.1f}x | {td['setup_tag']}")
        except Exception:
            continue

    print(f"  -> {len(flagged)} stocks flagged this scan\n")
    return flagged

def get_live_price(ticker):
    """Lightweight current-price fetch for outcome tracking. None on failure."""
    try:
        info = yf.Ticker(ticker).fast_info
        price = getattr(info, 'last_price', None)
        if price is None:
            price = info['lastPrice']
        return float(price) if price else None
    except Exception:
        return None

def fetch_news(ticker_symbol, max_items=3, news=None):
    """Pull the latest news headlines for a ticker. Returns JSON string.

    Pass a pre-fetched `news` list (from yf.Ticker(ticker).news) to avoid a
    duplicate network call; if None, it fetches its own.
    """
    try:
        if news is None:
            news = get_ticker_news(ticker_symbol)
        news_items = news[:max_items]
        cleaned = []
        for item in news_items:
            # yfinance has two news formats depending on version - handle both
            if 'content' in item:
                content = item['content']
                title = content.get('title', '')
                publisher = (content.get('provider') or {}).get('displayName', '')
                url = (content.get('canonicalUrl') or {}).get('url', '') \
                      or (content.get('clickThroughUrl') or {}).get('url', '')
                pub_date = content.get('pubDate', '')
            else:
                title = item.get('title', '')
                publisher = item.get('publisher', '')
                url = item.get('link', '')
                pub_time = item.get('providerPublishTime', 0)
                pub_date = datetime.fromtimestamp(pub_time).isoformat() if pub_time else ''
            if title:
                cleaned.append({
                    'title': title,
                    'publisher': publisher,
                    'url': url,
                    'date': pub_date
                })
        return json.dumps(cleaned)
    except Exception:
        return json.dumps([])
