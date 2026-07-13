"""Terminal dashboard (rich): account, signals, positions, trades, stats."""

from __future__ import annotations

import time
from typing import Callable

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _pnl_text(value: float, suffix: str = "") -> Text:
    style = "green" if value > 0 else "red" if value < 0 else "white"
    return Text(f"{value:+,.2f}{suffix}", style=style)


def _header(status: dict) -> Panel:
    account = status.get("account", {})
    day = status.get("day", {})
    market = "[green]OPEN[/green]" if status.get("market_open") else "[yellow]CLOSED[/yellow]"
    halted = status.get("halted")
    halt = f"  [bold red]HALTED[/bold red] {status.get('halt_reason', '')}" if halted else ""
    line1 = (f"[bold]Day Trading Engine[/bold]  broker=[cyan]{status.get('broker')}[/cyan]  "
             f"market {market}{halt}")
    realized = day.get("realized_pnl") or 0.0
    line2 = (f"equity [bold]${account.get('equity', 0):,.2f}[/bold]  "
             f"cash ${account.get('cash', 0):,.2f}  "
             f"day P&L {'[green]' if realized >= 0 else '[red]'}{realized:+,.2f}[/]  "
             f"daily loss limit ${day.get('max_daily_loss', 0):,.2f}  "
             f"tickers: {', '.join(status.get('tickers', []))}")
    return Panel(Group(Text.from_markup(line1), Text.from_markup(line2)), height=4)


def _positions_table(status: dict) -> Table:
    t = Table(title="Open Positions", expand=True, title_justify="left")
    for col in ("Symbol", "Dir", "Strat", "Qty", "Entry", "Mark", "Stop", "Target", "Unreal P&L"):
        t.add_column(col, justify="right" if col not in ("Symbol", "Dir", "Strat") else "left")
    for p in status.get("positions", []):
        t.add_row(
            p["symbol"], p["direction"], p["strategy"].replace("_", " "),
            f"{p['qty']:g}", f"{p['avg_entry']:,.2f}", f"{p['mark']:,.2f}",
            f"{p['stop_loss']:,.2f}", f"{p['target']:,.2f}",
            _pnl_text(p["unrealized_pnl"]),
        )
    if not status.get("positions"):
        t.add_row("-", "", "", "", "", "", "", "", Text("flat", style="dim"))
    return t


def _signals_table(status: dict) -> Table:
    t = Table(title="Recent Signals", expand=True, title_justify="left")
    for col in ("Time", "Symbol", "Strategy", "Dir", "Entry", "Stop", "Target", "Expiry", "Status"):
        t.add_column(col, overflow="fold")
    for s in status.get("signals", [])[:8]:
        t.add_row(
            str(s.get("created_at", ""))[11:19], s.get("symbol", ""),
            str(s.get("strategy", "")).replace("_", " "),
            "call" if s.get("direction") == "long" else "put",
            f"{s.get('entry_price') or 0:,.2f}", f"{s.get('stop_loss') or 0:,.2f}",
            f"{s.get('target_price') or 0:,.2f}",
            s.get("expiry_recommendation", ""), s.get("status", ""),
        )
    return t


def _trades_table(status: dict) -> Table:
    t = Table(title="Recent Trades", expand=True, title_justify="left")
    for col in ("Closed", "Symbol", "Strategy", "Dir", "Qty", "Entry", "Exit", "P&L", "Reason"):
        t.add_column(col, overflow="fold")
    for tr in status.get("trades", [])[:8]:
        t.add_row(
            str(tr.get("exit_time", ""))[11:19], tr.get("underlying") or tr.get("symbol", ""),
            str(tr.get("strategy", "")).replace("_", " "), tr.get("direction", ""),
            f"{tr.get('qty') or 0:g}", f"{tr.get('entry_price') or 0:,.2f}",
            f"{tr.get('exit_price') or 0:,.2f}", _pnl_text(tr.get("pnl") or 0.0),
            tr.get("exit_reason", ""),
        )
    return t


def _stats_panel(status: dict) -> Panel:
    s = status.get("stats", {})
    pf = s.get("profit_factor", 0.0)
    pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
    body = Text.from_markup(
        f"trades [bold]{s.get('total_trades', 0)}[/bold]   "
        f"win rate [bold]{s.get('win_rate', 0.0) * 100:.1f}%[/bold]   "
        f"total P&L {'[green]' if s.get('total_pnl', 0) >= 0 else '[red]'}"
        f"{s.get('total_pnl', 0.0):+,.2f}[/]   "
        f"avg win {s.get('avg_win', 0.0):,.2f}   avg loss {s.get('avg_loss', 0.0):,.2f}   "
        f"profit factor {pf_str}"
    )
    return Panel(body, title="Performance", title_align="left", height=3)


def render(status: dict) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(_header(status), size=4),
        Layout(_positions_table(status), name="positions"),
        Layout(_signals_table(status), name="signals"),
        Layout(_trades_table(status), name="trades"),
        Layout(_stats_panel(status), size=3),
    )
    return layout


def run_dashboard(status_provider: Callable[[], dict],
                  refresh_seconds: float = 2.0) -> None:
    """Blocking rich Live loop; Ctrl-C exits."""
    console = Console()
    with Live(render(status_provider()), console=console,
              refresh_per_second=4, screen=True) as live:
        try:
            while True:
                time.sleep(refresh_seconds)
                live.update(render(status_provider()))
        except KeyboardInterrupt:
            pass
