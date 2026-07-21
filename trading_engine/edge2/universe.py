import yfinance as yf

# Expanded universe - heavy on small/mid-cap biotechs, low-float names,
# and hot 2026 themes (AI, quantum, nuclear, space) where big surges happen.
# v3 will replace this with a dynamic screener.

UNIVERSE = [
    # === MOMENTUM / MEME / CRYPTO MINERS ===
    'GME', 'AMC', 'BB', 'NOK', 'PLTR', 'SOFI', 'HOOD', 'COIN', 'RIOT', 'MARA',
    'MSTR', 'CLSK', 'BTBT', 'HUT', 'CIFR', 'BITF', 'HIVE', 'IREN', 'WULF', 'CAN',
    'CORZ', 'GREE', 'CCCC', 'INVZ',

    # === CHINESE ADRs ===
    'NIO', 'XPEV', 'LI', 'BABA', 'JD', 'PDD', 'BIDU', 'BILI', 'TCOM', 'ZTO',
    'VIPS', 'IQ', 'HUYA', 'FUTU', 'TIGR', 'EH', 'BEKE', 'EDU', 'TAL', 'GOTU',

    # === BIOTECH / PHARMA (CATALYST NAMES - where 30-100% surges happen) ===
    'NTLA', 'CRSP', 'BEAM', 'EDIT', 'VKTX', 'ABCL', 'IOVA', 'OCGN', 'INO', 'NVAX',
    'SRPT', 'ALDX', 'ATAI', 'BPMC', 'CGEN', 'COGT', 'CRMD', 'CYTK', 'GRTS', 'KROS',
    'NUVB', 'PRTA', 'PTCT', 'RCKT', 'RIGL', 'ZNTL', 'ZYME', 'GERN', 'VOR', 'ALPN',
    'PRAX', 'RYTM', 'ABOS', 'ANNX', 'ADCT', 'CARA', 'MRSN', 'SVRA', 'SAGE', 'ITOS',
    'MRNS', 'MRTX', 'MNMD', 'GLPG', 'ATRA', 'MGTX', 'AKBA', 'FOLD', 'KALV', 'MORF',
    'IMUX', 'INMB', 'BNTC', 'CDXS', 'DRRX', 'NRBO', 'SLDB',

    # === EV / CLEAN ENERGY / NUCLEAR ===
    'RIVN', 'LCID', 'CHPT', 'BLNK', 'FCEL', 'PLUG', 'ENPH', 'RUN', 'WKHS', 'SOLO',
    'GOEV', 'FFIE', 'FSR', 'MULN', 'NRGV', 'SMR', 'OKLO', 'VST', 'TLN', 'NXE',
    'UEC', 'CCJ', 'DNN', 'LEU', 'ASPI', 'EVGO', 'OUST',

    # === AI / QUANTUM / SPACE / ROBOTICS ===
    'IONQ', 'RGTI', 'QBTS', 'BBAI', 'SOUN', 'ACHR', 'JOBY', 'RKLB', 'ASTS', 'LUNR',
    'DNA', 'BKSY', 'PL', 'MNTS', 'SPCE', 'ASTR', 'RDW', 'SYM', 'NAUT', 'NVTS',
    'ALAB', 'AMBA', 'CRDO', 'SMCI', 'INOD', 'GFAI', 'AIFU', 'AVPT', 'ARBE',

    # === RECENT IPOs / FINTECH / GROWTH ===
    'AFRM', 'DASH', 'ABNB', 'RBLX', 'PATH', 'GTLB', 'S', 'NET', 'DKNG', 'XYZ',
    'MNDY', 'TOST', 'KVYO', 'CART', 'RDDT', 'ARM', 'KIND', 'LSPD', 'INFA',

    # === CANNABIS / SHORT SQUEEZE / LOW-PRICED VOLATILE ===
    'SNDL', 'TLRY', 'CGC', 'ACB', 'CRON', 'MSOS', 'CURLF', 'GTBIF', 'TCNNF',
    'OPEN', 'CLOV', 'ATER', 'PROG', 'BIRD', 'ROOT', 'LMND', 'UPST',

    # === BIG TECH (context + occasional setups) ===
    'AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA', 'META', 'GOOGL', 'AMZN', 'NFLX', 'INTC',
    'SHOP', 'PYPL', 'ROKU', 'UBER', 'PINS', 'SNAP', 'LYFT',

    # === FINANCIALS / CYCLICAL / TRAVEL ===
    'F', 'GE', 'BAC', 'C', 'WFC', 'CCL', 'NCLH', 'AAL', 'DAL', 'UAL',
    'RCL', 'JBLU', 'SAVE', 'LUV', 'ALK', 'HA', 'BLDE',
]

def get_dynamic_universe():
    """Pull market-wide movers from Yahoo's predefined screeners and combine
    with the static watchlist. Falls back to the static list if screeners fail."""
    screens = ["day_gainers", "small_cap_gainers", "most_actives"]
    found = set()

    for name in screens:
        try:
            try:
                result = yf.screen(name, count=100)
            except TypeError:
                result = yf.screen(name)
            quotes = result.get("quotes", []) if isinstance(result, dict) else []
            for q in quotes:
                sym = q.get("symbol")
                if sym:
                    found.add(sym)
        except Exception as e:
            print(f"  [universe] screener '{name}' unavailable: {e}")

    if found:
        combined = sorted(found.union(set(UNIVERSE)))
        print(f"  [universe] {len(found)} live movers + {len(UNIVERSE)} watchlist = {len(combined)} total")
        return combined
    else:
        print(f"  [universe] screeners returned nothing — using static list of {len(UNIVERSE)}")
        return UNIVERSE
