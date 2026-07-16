"""FastAPI web dashboard: JSON API + a self-contained HTML page.

Endpoints:
    GET /              dashboard page (polls the API below)
    GET /api/status    full engine status snapshot
    GET /api/signals   recent signals
    GET /api/positions open positions
    GET /api/trades    recent closed trades
    GET /api/stats     aggregate performance
"""

from __future__ import annotations

from typing import Callable

_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Day Trading Engine</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 24px;
         background: #0f1115; color: #e6e6e6; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .meta { color: #9aa0a6; margin-bottom: 16px; font-size: 13px; }
  .halted { color: #ff5252; font-weight: 700; }
  .cards { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
  .card { background: #171a21; border: 1px solid #262b36; border-radius: 8px;
          padding: 12px 16px; min-width: 150px; }
  .card .label { font-size: 11px; text-transform: uppercase; color: #9aa0a6; }
  .card .value { font-size: 20px; font-weight: 600; margin-top: 2px; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 24px; font-size: 13px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #262b36; }
  th { color: #9aa0a6; font-weight: 500; font-size: 11px; text-transform: uppercase; }
  .pos { color: #4caf7d; } .neg { color: #ff5252; }
  h2 { font-size: 14px; color: #c8ccd4; margin: 18px 0 8px; }
</style></head>
<body>
<h1>Day Trading Engine</h1>
<div class="meta" id="meta">loading…</div>
<div class="cards" id="cards"></div>
<h2>Open Positions</h2><table id="positions"></table>
<h2>Active Signals</h2><table id="signals"></table>
<h2>Recent Trades</h2><table id="trades"></table>
<script>
const fmt = (v, d=2) => v == null ? "" : Number(v).toLocaleString(undefined,
  {minimumFractionDigits: d, maximumFractionDigits: d});
const pnlCell = v => `<td class="${v >= 0 ? 'pos' : 'neg'}">${fmt(v)}</td>`;
// options are always bought: bearish thesis == long puts, never short premium
const sideLabel = (instrument, direction) => instrument === "option"
  ? (direction === "long" ? "long call" : "long put")
  : direction;
function fillTable(id, headers, rows) {
  const el = document.getElementById(id);
  el.innerHTML = "<tr>" + headers.map(h => `<th>${h}</th>`).join("") + "</tr>" +
    (rows.length ? rows.join("") : `<tr><td colspan="${headers.length}">none</td></tr>`);
}
async function refresh() {
  try {
    const s = await (await fetch("/api/status")).json();
    const acct = s.account || {}, day = s.day || {}, stats = s.stats || {};
    document.getElementById("meta").innerHTML =
      `broker <b>${s.broker}</b> · market ${s.market_open ? "OPEN" : "CLOSED"}` +
      (s.halted ? ` · <span class="halted">HALTED: ${s.halt_reason}</span>` : "") +
      ` · ${new Date(s.time).toLocaleTimeString()}`;
    document.getElementById("cards").innerHTML = [
      ["Equity", "$" + fmt(acct.equity)],
      ["Cash", "$" + fmt(acct.cash)],
      ["Day P&L (realized)", fmt(day.realized_pnl)],
      ["Day P&L (total)", day.total_pnl == null ? "n/a" : fmt(day.total_pnl)],
      ["Loss Limit", "$" + fmt(day.max_daily_loss)],
      ["Win Rate", fmt((stats.win_rate || 0) * 100, 1) + "%"],
      ["Total P&L", fmt(stats.total_pnl)],
      ["Trades", stats.total_trades ?? 0],
    ].map(([l, v]) => `<div class="card"><div class="label">${l}</div>` +
                      `<div class="value">${v}</div></div>`).join("");
    fillTable("positions",
      ["Symbol", "Dir", "Strategy", "Qty", "Entry", "Mark", "Stop", "Target", "Unreal P&L"],
      (s.positions || []).map(p => `<tr><td>${p.symbol}</td>` +
        `<td>${p.side || sideLabel(p.instrument, p.direction)}</td>` +
        `<td>${p.strategy}</td><td>${p.qty}</td><td>${fmt(p.avg_entry)}</td>` +
        `<td>${fmt(p.mark)}</td><td>${fmt(p.stop_loss)}</td><td>${fmt(p.target)}</td>` +
        pnlCell(p.unrealized_pnl) + "</tr>"));
    fillTable("signals",
      ["Time", "Symbol", "Strategy", "Side", "Entry", "Stop", "Target", "Expiry", "Status"],
      (s.signals || []).map(x => `<tr><td>${(x.created_at || "").slice(11, 19)}</td>` +
        `<td>${x.symbol}</td><td>${x.strategy}</td>` +
        `<td>${x.direction === "long" ? "call" : "put"}</td><td>${fmt(x.entry_price)}</td>` +
        `<td>${fmt(x.stop_loss)}</td><td>${fmt(x.target_price)}</td>` +
        `<td>${x.expiry_recommendation || ""}</td><td>${x.status}</td></tr>`));
    fillTable("trades",
      ["Closed", "Symbol", "Strategy", "Dir", "Qty", "Entry", "Exit", "P&L", "Reason"],
      (s.trades || []).map(t => `<tr><td>${(t.exit_time || "").slice(11, 19)}</td>` +
        `<td>${t.underlying || t.symbol}</td><td>${t.strategy}</td>` +
        `<td>${sideLabel(t.instrument, t.direction)}</td>` +
        `<td>${t.qty}</td><td>${fmt(t.entry_price)}</td><td>${fmt(t.exit_price)}</td>` +
        pnlCell(t.pnl) + `<td>${t.exit_reason}</td></tr>`));
  } catch (e) { document.getElementById("meta").textContent = "engine unreachable: " + e; }
}
refresh(); setInterval(refresh, 5000);
</script>
</body></html>"""


def create_app(status_provider: Callable[[], dict]):
    """Build the FastAPI app around any `() -> status dict` provider."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="Day Trading Engine", docs_url="/docs")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _PAGE

    @app.get("/api/status")
    def status() -> dict:
        return status_provider()

    @app.get("/api/signals")
    def signals() -> list:
        return status_provider().get("signals", [])

    @app.get("/api/positions")
    def positions() -> list:
        return status_provider().get("positions", [])

    @app.get("/api/trades")
    def trades() -> list:
        return status_provider().get("trades", [])

    @app.get("/api/stats")
    def stats() -> dict:
        return status_provider().get("stats", {})

    return app


def serve(status_provider: Callable[[], dict], host: str, port: int,
          in_thread: bool = False):
    """Run uvicorn; optionally on a daemon thread next to the engine loop."""
    import uvicorn

    app = create_app(status_provider)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    if in_thread:
        import threading

        thread = threading.Thread(target=server.run, daemon=True, name="webdash")
        thread.start()
        return server
    server.run()
    return server
