"""Reader for the watchlist file APEX writes (see APEX src/apex/watchlist.py).

    {"version": 1, "generated_at": "...", "ttl_days": 7,
     "entries": [{"ticker": "XYZ", "score": 72.0, "filed_date": "2026-11-13",
                  "expires_on": "2026-11-20", "insider_class": "opportunistic",
                  "cluster_size": 2, "role_tier": "ceo_cfo",
                  "total_value_usd": 125000.0, "insiders": [...], "reasons": [...]}]}

The file is re-read only when its modification time changes. A missing or
broken file means "no APEX candidates" - never an exception in the scan loop.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WatchlistEntry:
    ticker: str
    score: float
    filed_date: Optional[date]
    expires_on: Optional[date]
    insider_class: str = ""
    cluster_size: int = 1
    role_tier: str = ""
    total_value_usd: float = 0.0
    insiders: tuple[str, ...] = field(default_factory=tuple)
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, raw: dict) -> "WatchlistEntry":
        return cls(
            ticker=str(raw["ticker"]).strip().upper(),
            score=float(raw.get("score") or 0.0),
            filed_date=_date(raw.get("filed_date")),
            expires_on=_date(raw.get("expires_on")),
            insider_class=str(raw.get("insider_class") or ""),
            cluster_size=int(raw.get("cluster_size") or 1),
            role_tier=str(raw.get("role_tier") or ""),
            total_value_usd=float(raw.get("total_value_usd") or 0.0),
            insiders=tuple(raw.get("insiders") or ()),
            reasons=tuple(raw.get("reasons") or ()),
        )


def _date(raw) -> Optional[date]:
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


class ApexWatchlist:
    def __init__(self, path: str | Path, min_score: float = 50.0, max_symbols: int = 30,
                 stale_after_minutes: int = 180) -> None:
        self.path = Path(path)
        self.min_score = min_score
        self.max_symbols = max_symbols
        self.stale_after = timedelta(minutes=stale_after_minutes)
        self._mtime: Optional[float] = None
        self._entries: list[WatchlistEntry] = []
        self._generated_at: Optional[datetime] = None
        self._warned_stale = False

    def _reload(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            if self._mtime is not None or not self._entries:
                if self._mtime is not None:
                    log.warning("apex watchlist %s disappeared", self.path)
                self._mtime, self._entries, self._generated_at = None, [], None
            return
        if mtime == self._mtime:
            return
        self._mtime = mtime
        try:
            payload = json.loads(self.path.read_text())
            entries = []
            for raw in payload.get("entries", []):
                try:
                    entries.append(WatchlistEntry.from_dict(raw))
                except (KeyError, TypeError, ValueError):
                    continue
            gen = payload.get("generated_at")
            self._generated_at = datetime.fromisoformat(gen) if gen else None
            self._entries = entries
            self._warned_stale = False
            log.info("apex watchlist loaded: %d entr%s from %s", len(entries),
                     "y" if len(entries) == 1 else "ies", self.path)
        except (OSError, ValueError, AttributeError) as exc:
            log.warning("apex watchlist %s unreadable (%s); ignoring it", self.path, exc)
            self._entries, self._generated_at = [], None

    def entries(self, today: date, now: Optional[datetime] = None) -> list[WatchlistEntry]:
        """Live, high-enough-scoring entries, best first, capped at max_symbols."""
        self._reload()
        now = now or datetime.now(timezone.utc)
        if (self._generated_at is not None and now - self._generated_at > self.stale_after
                and not self._warned_stale):
            log.warning("apex watchlist is stale (generated %s) - is APEX running?",
                        self._generated_at.isoformat())
            self._warned_stale = True
        live = [e for e in self._entries
                if e.score >= self.min_score and (e.expires_on is None or e.expires_on >= today)
                and (e.filed_date is None or e.filed_date <= today)]
        live.sort(key=lambda e: (-e.score, e.ticker))
        return live[: self.max_symbols]

    def status(self) -> dict:
        self._reload()
        return {
            "path": str(self.path),
            "generated_at": self._generated_at.isoformat() if self._generated_at else None,
            "entries": len(self._entries),
        }
