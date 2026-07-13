"""Command-line interface.

    trading-engine run [--config config.yaml] [--dashboard] [--web]
    trading-engine scan [--config ...] [--execute]
    trading-engine dashboard [--config ...]
    trading-engine web [--config ...]
    trading-engine report [--config ...]
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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
