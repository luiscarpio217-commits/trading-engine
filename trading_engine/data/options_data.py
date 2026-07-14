"""Options chain data, normalized across providers (yfinance / Tradier).

Every provider returns an `OptionsChain` whose `contracts` DataFrame has the
same columns, so strategies never care where the chain came from:

    occ, type, strike, expiry, dte, bid, ask, last, mid,
    volume, open_interest, iv, delta, gamma, theta, vega

yfinance supplies implied volatility but not greeks (they stay NaN);
Tradier supplies ORATS greeks when requested.
"""

from __future__ import annotations

import logging
import threading
import time as _time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

from ..models import OptionContract, OptionType, Direction, occ_symbol

log = logging.getLogger(__name__)

CHAIN_COLUMNS = ["occ", "type", "strike", "expiry", "dte", "bid", "ask", "last",
                 "mid", "volume", "open_interest", "iv", "delta", "gamma",
                 "theta", "vega"]


@dataclass
class OptionsChain:
    underlying: str
    spot: float
    asof: date
    contracts: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=CHAIN_COLUMNS))

    @property
    def empty(self) -> bool:
        return self.contracts.empty

    @property
    def expiries(self) -> list[date]:
        if self.empty:
            return []
        return sorted(self.contracts["expiry"].unique())

    def nearest_expiry(self, min_dte: int = 1) -> Optional[date]:
        """Nearest expiry at least `min_dte` days out, else the furthest available."""
        expiries = self.expiries
        if not expiries:
            return None
        for exp in expiries:
            if (exp - self.asof).days >= min_dte:
                return exp
        return expiries[-1]

    def median_iv(self) -> float:
        if self.empty:
            return float("nan")
        traded = self.contracts[self.contracts["volume"] > 0]
        pool = traded if not traded.empty else self.contracts
        return float(pool["iv"].median())

    def pick_contract(self, direction: Direction, min_dte: int = 1,
                      target_delta: float = 0.40) -> Optional[OptionContract]:
        """Recommend a contract for a directional trade.

        Prefers the nearest expiry >= min_dte; picks by |delta| closest to
        `target_delta` when greeks exist, otherwise the strike nearest to
        slightly out-of-the-money (spot +/- 0.5%).
        """
        expiry = self.nearest_expiry(min_dte)
        if expiry is None:
            return None
        want = "call" if direction is Direction.LONG else "put"
        subset = self.contracts[
            (self.contracts["expiry"] == expiry) & (self.contracts["type"] == want)
        ]
        if subset.empty:
            return None
        deltas = subset["delta"]
        if deltas.notna().any():
            row = subset.loc[(deltas.abs() - target_delta).abs().idxmin()]
        else:
            otm_ref = self.spot * (1.005 if want == "call" else 0.995)
            row = subset.loc[(subset["strike"] - otm_ref).abs().idxmin()]
        return row_to_contract(row, self.underlying)

    def contract_mid(self, occ: str) -> Optional[float]:
        if self.empty:
            return None
        rows = self.contracts[self.contracts["occ"] == occ]
        if rows.empty:
            return None
        mid = float(rows["mid"].iloc[0])
        return mid if mid == mid else None


def row_to_contract(row: pd.Series, underlying: str) -> OptionContract:
    def _f(key: str) -> float:
        v = row.get(key)
        return float(v) if v is not None and v == v else float("nan")

    occ = row.get("occ")
    occ = occ if isinstance(occ, str) else ""

    def _opt(key: str) -> Optional[float]:
        v = row.get(key)
        return float(v) if v is not None and v == v else None

    return OptionContract(
        underlying=underlying,
        option_type=OptionType(row["type"]),
        strike=float(row["strike"]),
        expiry=row["expiry"] if isinstance(row["expiry"], date) else pd.Timestamp(row["expiry"]).date(),
        occ=occ,
        bid=_f("bid"), ask=_f("ask"), last=_f("last"),
        volume=int(row.get("volume") or 0),
        open_interest=int(row.get("open_interest") or 0),
        iv=_f("iv"),
        delta=_opt("delta"), gamma=_opt("gamma"),
        theta=_opt("theta"), vega=_opt("vega"),
    )


def build_chain_frame(rows: list[dict], underlying: str = "") -> pd.DataFrame:
    """Assemble a normalized contracts frame, computing occ/mid as needed."""
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=CHAIN_COLUMNS)
    for col in CHAIN_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    occ = df["occ"].astype("object")
    missing = occ.isna() | (occ == "")
    if underlying and missing.any():
        computed = df.apply(
            lambda r: occ_symbol(underlying, r["expiry"],
                                 OptionType(r["type"]), float(r["strike"])),
            axis=1)
        occ = occ.where(~missing, computed)
    df["occ"] = occ.fillna("")
    bid = df["bid"].fillna(0.0)
    ask = df["ask"].fillna(0.0)
    mid = (bid + ask) / 2.0
    mid = mid.where((bid > 0) & (ask > 0), df["last"])
    df["mid"] = mid
    df["volume"] = df["volume"].fillna(0).astype(int)
    df["open_interest"] = df["open_interest"].fillna(0).astype(int)
    return df[CHAIN_COLUMNS]


class YFinanceOptionsData:
    """Options chains from Yahoo Finance (IV yes, greeks NaN)."""

    def __init__(self, max_expiries: int = 3, cache_seconds: int = 300,
                 market_data=None) -> None:
        self._max_expiries = max_expiries
        self._ttl = cache_seconds
        self._cache: dict = {}
        self._lock = threading.Lock()
        self._market_data = market_data  # optional spot-price source

    def get_chain(self, symbol: str) -> Optional[OptionsChain]:
        with self._lock:
            hit = self._cache.get(symbol)
            if hit and _time.monotonic() - hit[0] <= self._ttl:
                return hit[1]
        chain = self._fetch(symbol)
        if chain is not None:
            with self._lock:
                self._cache[symbol] = (_time.monotonic(), chain)
        return chain

    def _spot(self, symbol: str, ticker) -> Optional[float]:
        if self._market_data is not None:
            price = self._market_data.get_latest_price(symbol)
            if price:
                return price
        try:
            fi = ticker.fast_info
            raw = getattr(fi, "last_price", None) or fi.get("lastPrice")
            return float(raw) if raw else None
        except Exception:
            return None

    def _fetch(self, symbol: str) -> Optional[OptionsChain]:
        import yfinance as yf

        try:
            ticker = yf.Ticker(symbol)
            expiries = list(ticker.options or [])[: self._max_expiries]
            if not expiries:
                return None
            spot = self._spot(symbol, ticker)
            today = date.today()
            rows: list[dict] = []
            for exp_str in expiries:
                exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
                chain = ticker.option_chain(exp_str)
                for opt_type, frame in (("call", chain.calls), ("put", chain.puts)):
                    if frame is None or frame.empty:
                        continue
                    for _, r in frame.iterrows():
                        strike = float(r["strike"])
                        rows.append({
                            "occ": str(r.get("contractSymbol")
                                       or occ_symbol(symbol, exp, OptionType(opt_type), strike)),
                            "type": opt_type,
                            "strike": strike,
                            "expiry": exp,
                            "dte": (exp - today).days,
                            "bid": r.get("bid"),
                            "ask": r.get("ask"),
                            "last": r.get("lastPrice"),
                            "volume": r.get("volume"),
                            "open_interest": r.get("openInterest"),
                            "iv": r.get("impliedVolatility"),
                        })
            if spot is None:
                return None
            return OptionsChain(underlying=symbol, spot=float(spot), asof=today,
                                contracts=build_chain_frame(rows, symbol))
        except Exception as exc:
            log.warning("yfinance options chain failed for %s: %s", symbol, exc)
            return None


class TradierOptionsData:
    """Options chains from Tradier's market-data API (includes greeks)."""

    def __init__(self, client, max_expiries: int = 3, cache_seconds: int = 300) -> None:
        self._client = client  # execution.tradier.TradierClient
        self._max_expiries = max_expiries
        self._ttl = cache_seconds
        self._cache: dict = {}
        self._lock = threading.Lock()

    def get_chain(self, symbol: str) -> Optional[OptionsChain]:
        with self._lock:
            hit = self._cache.get(symbol)
            if hit and _time.monotonic() - hit[0] <= self._ttl:
                return hit[1]
        chain = self._fetch(symbol)
        if chain is not None:
            with self._lock:
                self._cache[symbol] = (_time.monotonic(), chain)
        return chain

    def _fetch(self, symbol: str) -> Optional[OptionsChain]:
        try:
            spot = self._client.get_quote(symbol)
            expirations = self._client.get_option_expirations(symbol)[: self._max_expiries]
            if spot is None or not expirations:
                return None
            today = date.today()
            rows: list[dict] = []
            for exp_str in expirations:
                exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
                for opt in self._client.get_option_chain(symbol, exp_str):
                    greeks = opt.get("greeks") or {}
                    rows.append({
                        "occ": opt.get("symbol", ""),
                        "type": str(opt.get("option_type", "")).lower(),
                        "strike": float(opt.get("strike", 0.0)),
                        "expiry": exp,
                        "dte": (exp - today).days,
                        "bid": opt.get("bid"),
                        "ask": opt.get("ask"),
                        "last": opt.get("last"),
                        "volume": opt.get("volume"),
                        "open_interest": opt.get("open_interest"),
                        "iv": greeks.get("mid_iv") or greeks.get("smv_vol"),
                        "delta": greeks.get("delta"),
                        "gamma": greeks.get("gamma"),
                        "theta": greeks.get("theta"),
                        "vega": greeks.get("vega"),
                    })
            return OptionsChain(underlying=symbol, spot=float(spot), asof=today,
                                contracts=build_chain_frame(rows, symbol))
        except Exception as exc:
            log.warning("tradier options chain failed for %s: %s", symbol, exc)
            return None
