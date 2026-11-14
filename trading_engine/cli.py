"""Command-line interface.

    trading-engine run [--config config.yaml] [--dashboard] [--web]
    trading-engine scan [--config ...] [--execute]
    trading-engine dashboard [--config ...]
    trading-engine web [--config ...]
    trading-engine report [--config ...]
    trading-engine weekly-report [--config ...] [--days 7] [--post]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from .config import Config
from .engine import TradingEngine


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load(args) -> Config:
    return Config.load(args.config) if args.config else Config()


def cmd_run(args) -> int:
    config = _load(args)
    _setup_logging(config.engine.log_level)
    engine = TradingEngine(config)

    web_server = None
    if args.web or config.web.enabled:
        from .dashboard.web import serve
        web_server = serve(engine.get_status, config.web.host, config.web.port,
                           in_thread=True)
        print(f"web dashboard: http://{config.web.host}:{config.web.port}")

    if args.dashboard:
        from .dashboard.terminal import run_dashboard
        scheduler = engine.run(blocking=False)
        try:
            run_dashboard(engine.get_status)
        finally:
            scheduler.shutdown(wait=False)
            engine.shutdown()
        return 0

    try:
        engine.run(blocking=True)
    finally:
        engine.shutdown()
        if web_server is not None:
            web_server.should_exit = True
    return 0


def cmd_scan(args) -> int:
    config = _load(args)
    _setup_logging(config.engine.log_level)
    engine = TradingEngine(config)
    signals = engine.scan_once(execute=args.execute)
    if not signals:
        print("no signals this pass")
    for s in signals:
        print(f"\n{s.strategy}  {s.symbol}  {s.direction.value.upper()} "
              f"({'CALL' if s.direction.value == 'long' else 'PUT'})")
        print(f"  entry {s.entry_price:.2f}  stop {s.stop_loss:.2f}  "
              f"target {s.target_price:.2f}  R:R {s.reward_risk:.1f}  "
              f"confidence {s.confidence:.0%}")
        print(f"  expiry: {s.expiry_recommendation}"
              + (f"  contract: {s.contract.describe()}" if s.contract else ""))
        for r in s.reasons:
            print(f"    - {r}")
    engine.shutdown()
    return 0


def cmd_dashboard(args) -> int:
    config = _load(args)
    _setup_logging("WARNING")
    engine = TradingEngine(config)
    from .dashboard.terminal import run_dashboard
    scheduler = engine.run(blocking=False)
    try:
        run_dashboard(engine.get_status)
    finally:
        scheduler.shutdown(wait=False)
        engine.shutdown()
    return 0


def cmd_web(args) -> int:
    config = _load(args)
    _setup_logging(config.engine.log_level)
    engine = TradingEngine(config)
    scheduler = engine.run(blocking=False)
    from .dashboard.web import serve
    print(f"web dashboard: http://{config.web.host}:{config.web.port}")
    try:
        serve(engine.get_status, config.web.host, config.web.port, in_thread=False)
    finally:
        scheduler.shutdown(wait=False)
        engine.shutdown()
    return 0


def cmd_report(args) -> int:
    config = _load(args)
    from .storage.trade_log import TradeLog
    log = TradeLog(config.engine.db_path, None)
    stats = log.stats()
    print(json.dumps(stats, indent=2, default=str))
    trades = log.recent_trades(args.limit)
    if trades:
        print(f"\nlast {len(trades)} trades:")
        for t in trades:
            print(f"  {t['exit_time'][:19]}  {t['underlying'] or t['symbol']:<6} "
                  f"{t['strategy']:<18} {t['direction']:<5} qty {t['qty']:>6g}  "
                  f"pnl {t['pnl']:>+10.2f}  {t['exit_reason']}")
    log.close()
    return 0


def weekly_summary(trades: list[dict], days: int, now=None) -> dict:
    """P&L per source (ironfrost / edge2 / apex) for trades closed in the window."""
    from datetime import datetime, timedelta, timezone

    now = now or datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    by_source: dict[str, dict] = {}
    for t in trades:
        try:
            closed = datetime.fromisoformat(str(t["exit_time"]))
        except (KeyError, ValueError):
            continue
        if closed.tzinfo is None:
            closed = closed.replace(tzinfo=timezone.utc)
        if closed < since:
            continue
        src = t.get("source") or "ironfrost"
        b = by_source.setdefault(src, {"trades": 0, "wins": 0, "pnl": 0.0,
                                       "best": None, "worst": None})
        pnl = float(t.get("pnl") or 0.0)
        b["trades"] += 1
        b["wins"] += 1 if pnl > 0 else 0
        b["pnl"] += pnl
        label = f"{t.get('underlying') or t.get('symbol')} {pnl:+.2f}"
        if b["best"] is None or pnl > b["best"][0]:
            b["best"] = (pnl, label)
        if b["worst"] is None or pnl < b["worst"][0]:
            b["worst"] = (pnl, label)
    total = sum(b["pnl"] for b in by_source.values())
    return {"since": since.isoformat(), "days": days, "total_pnl": round(total, 2),
            "by_source": by_source}


def format_weekly(summary: dict) -> str:
    lines = [f"Paper trading, last {summary['days']} days: total {summary['total_pnl']:+.2f}"]
    if not summary["by_source"]:
        lines.append("no closed trades")
    for src, b in sorted(summary["by_source"].items()):
        rate = b["wins"] / b["trades"] if b["trades"] else 0.0
        lines.append(f"- {src}: {b['trades']} trades, {rate:.0%} winners, {b['pnl']:+.2f}"
                     + (f" (best {b['best'][1]}, worst {b['worst'][1]})" if b["best"] else ""))
    return "\n".join(lines)


def cmd_weekly_report(args) -> int:
    import os

    config = _load(args)
    from .storage.trade_log import TradeLog
    tlog = TradeLog(config.engine.db_path, None)
    trades = tlog._rows("SELECT * FROM trades ORDER BY exit_time")
    tlog.close()
    text = format_weekly(weekly_summary(trades, args.days))
    print(text)
    if args.post:
        url = os.environ.get(args.webhook_env, "")
        if not url:
            # not an error: Discord is optional (the timer still runs harmlessly)
            print(f"--post: ${args.webhook_env} is not set; nothing sent")
            return 0
        import requests
        resp = requests.post(url, json={"content": text[:1900]}, timeout=15)
        if resp.status_code >= 300:
            print(f"discord post failed: HTTP {resp.status_code}", file=sys.stderr)
            return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="trading-engine",
                                     description="Day trading strategy engine")
    parser.add_argument("--config", "-c", help="path to config.yaml", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="start the live scan/manage loop")
    p_run.add_argument("--dashboard", action="store_true",
                       help="show the terminal dashboard while running")
    p_run.add_argument("--web", action="store_true",
                       help="serve the web dashboard while running")
    p_run.set_defaults(func=cmd_run)

    p_scan = sub.add_parser("scan", help="run one strategy pass and print signals")
    p_scan.add_argument("--execute", action="store_true",
                        help="also size and submit orders (default: signal only)")
    p_scan.set_defaults(func=cmd_scan)

    p_dash = sub.add_parser("dashboard", help="run engine with terminal dashboard")
    p_dash.set_defaults(func=cmd_dashboard)

    p_web = sub.add_parser("web", help="run engine with web dashboard")
    p_web.set_defaults(func=cmd_web)

    p_report = sub.add_parser("report", help="print performance stats from the trade log")
    p_report.add_argument("--limit", type=int, default=20)
    p_report.set_defaults(func=cmd_report)

    p_week = sub.add_parser("weekly-report",
                            help="P&L by source (ironfrost/edge2/apex) for the last N days")
    p_week.add_argument("--days", type=int, default=7)
    p_week.add_argument("--post", action="store_true",
                        help="also post to the Discord webhook in $DISCORD_WEBHOOK_URL")
    p_week.add_argument("--webhook-env", default="DISCORD_WEBHOOK_URL")
    p_week.set_defaults(func=cmd_weekly_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
