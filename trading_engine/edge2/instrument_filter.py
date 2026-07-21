"""
EDGE2 patch 1 — Instrument filter
=================================
Fixes the bug where preferred shares (AHT-PD, AHT-PF, ...) and warrants
(RVMDW) get flagged as Rockets.

Wired into scanner.scan_universe(): the dynamic universe is filtered so
preferreds, warrants, rights, and units never reach the flagging logic.
"""

import re

# Suffix patterns by data-vendor convention. Covers the formats used by
# Polygon, Yahoo, Finnhub, Alpaca, and NASDAQ/NYSE raw feeds.
_PREFERRED_PATTERNS = [
    re.compile(r"-P[A-Z]?$"),       # AHT-PD, AHT-PF   (Yahoo/Polygon style)
    re.compile(r"\.PR[A-Z]?$"),     # AHT.PRD          (NYSE style)
    re.compile(r"p[A-Z]$"),         # AHTpD            (some feeds, lowercase p)
    re.compile(r"-$"),              # trailing dash artifacts
]

_WARRANT_PATTERNS = [
    re.compile(r"[-.+]W[ST]?$"),    # ABC-WT, ABC.WS, ABC+
    re.compile(r"^[A-Z]{4}W$"),     # RVMDW — 5-letter NASDAQ ending in W
]

_UNIT_RIGHT_PATTERNS = [
    re.compile(r"[-.]U$"),          # SPAC units: ABC.U / ABC-U
    re.compile(r"^[A-Z]{4}U$"),     # 5-letter NASDAQ unit
    re.compile(r"[-.]R$"),          # rights: ABC.R
    re.compile(r"^[A-Z]{4}R$"),     # 5-letter NASDAQ right
]

_NAME_KEYWORDS = (
    "preferred", "pfd", "warrant", " unit", "right", "depositary",
    " notes", "% notes", "debenture",
)


def is_common_stock(ticker: str, name: str | None = None,
                    price: float | None = None,
                    min_price: float = 1.00) -> tuple[bool, str]:
    """
    Returns (True, "") if the ticker looks like tradeable common stock,
    or (False, reason) if it should be skipped.
    """
    t = ticker.strip().upper().replace("/", "-")

    for pat in _PREFERRED_PATTERNS:
        if pat.search(ticker.strip()):    # check raw case for the pA style
            return False, "preferred share"
    for pat in _WARRANT_PATTERNS:
        if pat.search(t):
            return False, "warrant"
    for pat in _UNIT_RIGHT_PATTERNS:
        if pat.search(t):
            return False, "unit/right"

    # Company-name check catches things the ticker format misses
    if name:
        n = name.lower()
        for kw in _NAME_KEYWORDS:
            if kw in n:
                return False, f"name contains '{kw.strip()}'"

    # Optional price floor — sub-$1 flags are mostly noise and untradeable
    if price is not None and price < min_price:
        return False, f"price below ${min_price:.2f}"

    return True, ""


def dedupe_related(tickers: list[str]) -> list[str]:
    """
    Optional second pass: if several flagged tickers share the same root
    (AHT-PD, AHT-PF, AHT-PG...), keep only the first. Prevents one
    corporate event from filling the board.
    """
    seen_roots: set[str] = set()
    kept = []
    for t in tickers:
        root = re.split(r"[-.+]", t.upper())[0]
        if root in seen_roots:
            continue
        seen_roots.add(root)
        kept.append(t)
    return kept


if __name__ == "__main__":
    # Quick self-test — run: python instrument_filter.py
    cases = [
        ("AHT-PD", False), ("AHT-PF", False), ("AHT.PRD", False),
        ("RVMDW", False), ("ABC.WS", False), ("SPAQ.U", False),
        ("AAPL", True), ("TSLA", True), ("SOFI", True), ("GME", True),
    ]
    for tick, expected in cases:
        ok, why = is_common_stock(tick)
        status = "OK " if ok == expected else "FAIL"
        print(f"{status} {tick:8s} -> {ok} {why}")
