# Day Trading Strategy Engine

A Python day-trading engine: market data and options chains in, filtered and
risk-sized trade signals out, executed against a paper simulator or a live
brokerage (Alpaca / Tradier), with dashboards and a full trade journal.

> **Disclaimer**: educational software. Day trading and options carry
> substantial risk of loss. Nothing here is financial advice. Always start
> with the built-in paper broker or a brokerage sandbox account.

## Architecture

```
                 ┌──────────────── Data Layer ────────────────┐
   yfinance ───► │ OHLCV (intraday + daily)   Options chains  │ ◄── Tradier
                 │ Indicators: EMA 9/21/50, RSI14, VWAP,      │     (greeks)
                 │ Bollinger, ATR, MACD, RVOL, Volume Profile │
                 └──────────────────┬─────────────────────────┘
                                    ▼
                 ┌────────────── Strategy Engine ─────────────┐
                 │ momentum_breakout      options_flow (UOA)  │
                 │ Filters: market hours (9:30–16:00 ET),     │
                 │ min volume, earnings blackout              │
                 │ Signal: direction (call/put), entry, stop, │
                 │ target, expiry + contract recommendation   │
                 └──────────────────┬─────────────────────────┘
                                    ▼
                 ┌────────────── Risk Management ─────────────┐
                 │ per-trade risk % (fixed fractional/Kelly)  │
                 │ daily loss limit → auto-shutoff (+flatten) │
                 │ max positions, notional caps               │
                 └──────────────────┬─────────────────────────┘
                                    ▼
                 ┌────────────── Execution Layer ─────────────┐
                 │ PaperBroker | Alpaca REST | Tradier REST   │
                 │ order manager: partial fills, auto         │
                 │ stop-loss, targets, EOD flatten            │
                 └──────────────────┬─────────────────────────┘
                                    ▼
                 ┌───────── Journal + Dashboards ─────────────┐
                 │ SQLite + CSV (signals/orders/trades/equity)│
                 │ rich terminal UI      FastAPI web UI       │
                 └────────────────────────────────────────────┘
```

## Quick start

```bash
pip install -r requirements.txt          # or: pip install -e .[web,dev]

# one signal-only pass over the default tickers (no orders)
python -m trading_engine scan

# paper-trade the loop with the terminal dashboard
python -m trading_engine run --dashboard

# paper-trade with the web dashboard at http://127.0.0.1:8000
python -m trading_engine web

# performance report from the trade journal
python -m trading_engine report
```

Everything defaults to the **built-in paper broker** with $100k — no
credentials needed. Copy `config.example.yaml` to `config.yaml`, adjust, and
pass `--config config.yaml`.

**Run it 24/7 on a server**: `deploy/` has a systemd unit, a one-shot
`setup.sh`, and a step-by-step [deploy/README.md](deploy/README.md) for a
fresh Ubuntu VPS (written for DigitalOcean's browser console). The web
dashboard requires HTTP Basic auth via `DASHBOARD_USERNAME` /
`DASHBOARD_PASSWORD` and **refuses to bind non-loopback hosts without
credentials**; `web.host`/`web.port` are set in `config.yaml`
(`127.0.0.1:8000` default locally, `0.0.0.0` on the server).

## Strategies

**Momentum breakout** (`momentum_breakout`) — long when the current bar makes
a *fresh* close above the prior 20-bar high with volume confirmation
(RVOL ≥ 1.5×), bullish EMA 9>21>50 alignment, price above session VWAP, and
RSI(14) in [50, 80]; mirrored short setup on breakdowns (optional). Stop is
1.5× ATR(14), target is 2R, and MACD histogram / volume-profile POC add
confidence. When a chain is available the signal carries a contract
recommendation (nearest expiry ≥ 1 DTE, ~0.40 delta).

**Options flow** (`options_flow`) — scans near-dated contracts (≤ 14 DTE)
for unusual activity: volume ≥ 2× open interest, volume ≥ 500, IV ≥ 1.25×
chain median, strike within 10% of spot. Unusual premium is aggregated per
side; when calls out-weigh puts 1.5× (or vice versa) the dominant side sets
the direction, and the top contract becomes the recommendation with its
strike acting as the price magnet for the target.

Both run behind shared filters: **market hours only** (9:30–16:00 ET,
weekdays), **minimum average daily volume** (default 1M shares), and an
**earnings blackout** (default ±2 days). Signals per symbol/strategy are
rate-limited by a cooldown.

## Risk management

- **Per-trade risk**: default 1% of equity (`max_risk_per_trade_pct`).
  Equity size = budget / (entry − stop); option size = budget /
  (premium × 100 × `premium_stop_pct`) — risk is budgeted at the
  engine-enforced premium stop (default 50%), mirroring equity
  risk-to-stop sizing. Both capped by `max_position_notional_pct`
  (full premium notional for options), plus `min_option_premium`
  (default $0.10 — no sub-dime lottery tickets) and
  `max_option_contracts` (default 25). Zero-size outcomes carry the
  gate that caused them in the signal status
  (`sized_zero:risk_budget_lt_one_contract`, `sized_zero:premium_below_min`, …).
- **Kelly criterion** (optional, `sizing_method: kelly`): half-Kelly on
  realized win rate / payoff from the journal, hard-capped, with automatic
  fall-back to fixed fractional until ≥ 20 closed trades exist.
- **Daily loss auto-shutoff**: when realized P&L or mark-to-market drawdown
  hits `max_daily_loss_pct` (default 3%), new entries halt for the rest of
  the session and (optionally) all positions are flattened.
- **Order management**: every equity fill is protected by a broker stop order
  resized as partial fills accrete; options are stop-managed engine-side
  (underlying stop + 50% premium stop). Everything is flattened at 15:55 ET
  by default — day trading, no overnight risk.
- **Premium profit protection** (optional, per strategy, off by default):
  `profit_protection.take_profit_pct` closes an option position at +N%
  premium gain; `profit_protection.trailing` (`arm_pct`/`giveback_pct`)
  arms once up N% and closes after a retrace off the peak mark. Whichever
  triggers first wins; journal exit reasons `take_profit` and
  `trailing_lock` keep the exit types analyzable. See the commented
  examples in `config.example.yaml`.

## Brokers

| broker    | config                 | credentials (env vars)                       |
|-----------|------------------------|----------------------------------------------|
| `paper`   | default, no setup      | —                                            |
| `alpaca`  | `alpaca.paper: true`   | `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY` |
| `tradier` | `tradier.sandbox: true`| `TRADIER_ACCESS_TOKEN`, `TRADIER_ACCOUNT_ID` |

Both live adapters speak the plain REST APIs directly (no SDK pinning):
market/limit/stop orders for equities and single-leg options (OCC symbols),
order polling with partial-fill sync, positions, and account balances.
Tradier also serves options chains **with greeks** — set
`data.options_provider: tradier` to use them for contract selection.

Paper-mode option marks come from a Black-Scholes model **calibrated to the
entry fill** (implied vol backed out from the actual entry premium, then held
constant): the mark equals the entry premium at entry, converges to intrinsic
value at expiry (0DTE-safe), and extrinsic value stays bounded — see
`trading_engine/data/pricing.py`. Still an approximation; treat paper option
P&L accordingly when reviewing the journal. Live brokers use real quotes.

## Journal & review

Every signal, order, fill, trade, and equity snapshot lands in
`data/trading.db` (SQLite) and mirrors to `data/trades.csv` /
`data/signals.csv`. `trading-engine report` prints win rate, profit factor,
average win/loss, and per-strategy breakdowns — the same stats the Kelly
sizer consumes.

## Layout

```
trading_engine/
├── config.py                # YAML + env configuration
├── models.py                # Signal / Order / Position / TradeRecord
├── engine.py                # APScheduler scan + manage loops
├── filters.py               # market hours, volume, earnings
├── data/
│   ├── market_data.py       # yfinance OHLCV + earnings (TTL-cached)
│   ├── options_data.py      # normalized chains (yfinance / Tradier)
│   └── indicators.py        # EMA, RSI, VWAP, BBands, ATR, MACD, RVOL, volume profile
├── strategies/              # momentum_breakout, options_flow
├── risk/                    # position sizing (fixed/Kelly), daily-loss kill switch
├── execution/               # paper / alpaca / tradier + order manager
├── storage/trade_log.py     # SQLite + CSV journal
└── dashboard/               # rich terminal UI, FastAPI web UI
tests/                       # 136 offline tests, synthetic data only
```

Indicators are implemented natively on pandas/numpy (Wilder smoothing for
RSI/ATR, session-anchored VWAP) — no ta-lib binary dependency; swap in
pandas-ta/ta-lib later if you prefer, the strategy code only reads columns.

## Tests

```bash
python -m pytest
```

The suite runs fully offline: synthetic OHLCV/chains exercise indicators,
both strategies, filters, sizing math, the daily-loss shutoff, paper-broker
fills (market/limit/stop, partial fills), the order manager's stop
management, the SQLite/CSV journal, and an end-to-end
signal → order → stop-out engine pass.
