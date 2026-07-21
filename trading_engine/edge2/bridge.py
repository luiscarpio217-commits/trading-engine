"""EDGE2 -> IronFrost seam: first flag of the day becomes a paper trade.

scanner.scan_universe() calls `open_paper_trade(td)` (via the injected
`scanner.on_first_flag` hook) once per ticker per day. The td dict already
carries EDGE2's own levels — entry_low/high, stop_loss, target_1/2/3,
setup_tag, gap_percent, catalyst_* — and they are mapped straight onto an
IronFrost Signal, never recomputed:

    entry_price  = td['price']       (flagged price; midpoint of the entry zone)
    stop_loss    = td['stop_loss']
    target_price = td['target_2']    (EDGE2's own risk_reward basis; t1/t3
                                      ride along in the signal reasons)
    direction    = LONG              (every EDGE2 setup is long)
    instrument   = EQUITY            ($1-30 names, shares)
    source       = 'edge2'           (per-source P&L split in the journal)

The signal then flows through IronFrost's EXISTING gates and plumbing —
risk manager (halt / max positions), position sizer, order manager (auto
stop, target, EOD flatten), journal — nothing execution-side is new.

PAPER ONLY: the bridge refuses to submit unless the engine's broker is the
built-in PaperBroker. If the config ever points at a live broker, EDGE2
flags log a warning and do nothing.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..execution.paper import PaperBroker
from ..models import Direction, Instrument, Signal

log = logging.getLogger(__name__)

SOURCE = "edge2"
STRATEGY = "edge2_scanner"


def td_to_signal(td: dict) -> Signal:
    """Map an EDGE2 flag dict onto an IronFrost Signal, straight off td."""
    reasons = [
        f"EDGE2 {td.get('setup_tag', 'flag')}: gap {td.get('gap_percent', 0):+.1f}%, "
        f"volume {td.get('volume_ratio', 0):.1f}x avg ({td.get('session', '?')})",
        f"entry zone {td.get('entry_low')}-{td.get('entry_high')}, "
        f"targets {td.get('target_1')} / {td.get('target_2')} / {td.get('target_3')} "
        f"(R:R {td.get('risk_reward')})",
        f"catalyst {td.get('catalyst_score', 0):.2f} "
        f"[{td.get('catalyst_category', 'none')}, {td.get('catalyst_velocity', 'none')}]",
    ]
    return Signal(
        symbol=td["ticker"],
        strategy=STRATEGY,
        direction=Direction.LONG,
        entry_price=float(td["price"]),
        stop_loss=float(td["stop_loss"]),
        target_price=float(td["target_2"]),
        instrument=Instrument.EQUITY,
        confidence=min(0.5 + float(td.get("catalyst_score", 0.0)) * 0.4, 0.9),
        reasons=reasons,
        source=SOURCE,
    )


class Edge2Bridge:
    """Holds the engine reference and exposes the scanner hook."""

    def __init__(self, engine) -> None:
        self._engine = engine

    def open_paper_trade(self, td: dict) -> Optional[Signal]:
        engine = self._engine
        if not isinstance(engine.broker, PaperBroker):
            log.warning("edge2: broker is %s, not paper - refusing to trade flag %s "
                        "(EDGE2 flags are paper-only by design)",
                        engine.broker.name, td.get("ticker"))
            return None
        try:
            signal = td_to_signal(td)
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("edge2: flag dict for %s not mappable (%s); skipped",
                        td.get("ticker"), exc)
            return None
        if signal.stop_loss >= signal.entry_price:
            log.info("edge2: %s stop %.2f not below entry %.2f; skipped",
                     signal.symbol, signal.stop_loss, signal.entry_price)
            return None

        engine._ensure_session(signal.created_at)
        engine.trade_log.log_signal(signal)
        log.info("edge2 signal: %s %s entry %.2f stop %.2f target %.2f (%s)",
                 signal.symbol, signal.direction.value, signal.entry_price,
                 signal.stop_loss, signal.target_price, td.get("setup_tag", ""))
        # Reuse the engine's whole execution path: exposure check, risk gates,
        # sizing, paper-mark priming, order submit, signal status updates.
        engine._execute_signal(signal)
        return signal
