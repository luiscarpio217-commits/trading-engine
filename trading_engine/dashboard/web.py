"""FastAPI web dashboard: JSON API + a self-contained HTML page.

Endpoints:
    GET /              dashboard page (polls the API below)
    GET /api/status    full engine status snapshot
    GET /api/signals   recent signals
    GET /api/positions open positions
    GET /api/trades    recent closed trades
    GET /api/stats     aggregate performance

Authentication: HTTP Basic over every route (page, API, /docs). Credentials
come from the DASHBOARD_USERNAME / DASHBOARD_PASSWORD environment variables.
When they are set, every request must authenticate. Binding to anything other
than loopback WITHOUT credentials is refused outright (fail closed) — an
internet-reachable dashboard must never run open. Plain HTTP Basic is not
encrypted in transit; use a strong password and consider a TLS reverse proxy.
"""

from __future__ import annotations

import base64
import logging
import os
import secrets
from typing import Callable, Optional

log = logging.getLogger(__name__)

USERNAME_ENV = "DASHBOARD_USERNAME"
PASSWORD_ENV = "DASHBOARD_PASSWORD"
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def resolve_credentials() -> tuple[Optional[str], Optional[str]]:
    """(username, password) from the environment; blank counts as unset."""
    user = os.environ.get(USERNAME_ENV, "").strip() or None
    password = os.environ.get(PASSWORD_ENV, "").strip() or None
    return user, password

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


def create_app(status_provider: Callable[[], dict],
               username: Optional[str] = None,
               password: Optional[str] = None):
    """Build the FastAPI app around any `() -> status dict` provider.

    When `username` and `password` are both given, EVERY route (page, API,
    docs, openapi) requires HTTP Basic auth with those credentials.
    """
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="Day Trading Engine", docs_url="/docs")

    if username and password:
        expected_user = username.encode()
        expected_pass = password.encode()

        @app.middleware("http")
        async def require_basic_auth(request: Request, call_next):
            header = request.headers.get("authorization", "")
            authorized = False
            if header[:6].lower() == "basic ":
                try:
                    decoded = base64.b64decode(header[6:], validate=True).decode("utf-8")
                    got_user, _, got_pass = decoded.partition(":")
                    # compare both unconditionally: constant-time, no short-circuit
                    user_ok = secrets.compare_digest(got_user.encode(), expected_user)
                    pass_ok = secrets.compare_digest(got_pass.encode(), expected_pass)
                    authorized = user_ok and pass_ok
                except Exception:
                    authorized = False
            if not authorized:
                return Response(
                    status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="trading-engine"'},
                    content="authentication required",
                )
            return await call_next(request)

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
          in_thread: bool = False,
          username: Optional[str] = None, password: Optional[str] = None):
    """Run uvicorn; optionally on a daemon thread next to the engine loop.

    Credentials default to DASHBOARD_USERNAME / DASHBOARD_PASSWORD from the
    environment. Binding a non-loopback host without credentials raises —
    fail closed rather than exposing an open dashboard to the internet.
    """
    import uvicorn

    if username is None and password is None:
        username, password = resolve_credentials()
    if not (username and password):
        if host not in _LOOPBACK_HOSTS:
            raise RuntimeError(
                f"refusing to bind {host}:{port} without authentication - set the "
                f"{USERNAME_ENV} and {PASSWORD_ENV} environment variables "
                f"(dashboard would be reachable by anyone)")
        log.warning("dashboard auth disabled (%s/%s not set); allowed on "
                    "loopback %s only", USERNAME_ENV, PASSWORD_ENV, host)
        username = password = None
    else:
        log.info("dashboard basic auth enabled for user %r", username)

    app = create_app(status_provider, username, password)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    if in_thread:
        import threading

        thread = threading.Thread(target=server.run, daemon=True, name="webdash")
        thread.start()
        return server
    server.run()
    return server
