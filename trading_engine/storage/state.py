"""Save and restore open positions + the paper account across restarts.

Multi-day (APEX) positions are held overnight, and the engine keeps orders,
positions and the paper account in memory only. Without this, a restart
(deploy, reboot, crash) would forget every open position and reset paper cash
to starting_cash. Enabled by engine.state_path; paper broker only, because a
real broker is the source of truth for its own positions.

Delete the state file to reset the paper account.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ..execution.paper import PaperBroker

log = logging.getLogger(__name__)

STATE_VERSION = 1


def save_state(path: str | Path, broker, orders) -> bool:
    if not isinstance(broker, PaperBroker):
        return False
    payload = {
        "version": STATE_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "broker": broker.export_state(),
        "positions": orders.export_state(),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".state-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, default=str)
        os.replace(tmp, target)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return True


def load_state(path: str | Path, broker, orders) -> int:
    """Restore a saved state. Returns positions restored (0 if no file)."""
    target = Path(path)
    if not target.exists():
        return 0
    if not isinstance(broker, PaperBroker):
        log.warning("state file %s ignored: broker is %s, not paper", target, broker.name)
        return 0
    try:
        payload = json.loads(target.read_text())
    except (OSError, ValueError) as exc:
        log.error("state file %s unreadable (%s); starting fresh", target, exc)
        return 0
    if payload.get("version") != STATE_VERSION:
        log.error("state file %s has version %s, expected %s; starting fresh",
                  target, payload.get("version"), STATE_VERSION)
        return 0
    broker.import_state(payload.get("broker", {}))
    restored = orders.import_state(payload.get("positions", []))
    log.info("restored paper account from %s (saved %s): cash %.2f, %d open position(s)",
             target, payload.get("saved_at"), broker.get_account().cash, restored)
    return restored
