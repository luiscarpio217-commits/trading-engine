# Deploying to a DigitalOcean Droplet (paper mode, 24/7)

This guide assumes **no prior server experience** and uses only DigitalOcean's
browser-based console — you never need a local SSH client. At the end, the
engine runs around the clock in paper mode (no broker keys, no real money),
restarts itself after crashes or reboots, and serves its dashboard at
`http://<your-droplet-ip>:8000` behind a username and password.

---

## 1. Create the droplet

1. In the DigitalOcean control panel: **Create → Droplets**.
2. Choose image **Ubuntu 24.04 (LTS) x64**.
3. The smallest plan (1 vCPU / 1 GB RAM) is enough.
4. Under authentication pick either option (password is simplest — you'll use
   the browser console anyway).
5. Create the droplet and wait until it shows an IP address. **Write the IP
   down** — you'll use it at the end.

## 2. Open the browser console

On the droplet's page choose **Access → Launch Droplet Console**. A terminal
opens in your browser, already logged in as `root`. Every command below is
typed (or pasted) into that window, then run with Enter.

> Paste tip: in the browser console use **Ctrl+Shift+V** (or right-click →
> Paste) if plain Ctrl+V does nothing.

## 3. Install the engine

Copy-paste these three commands, one at a time:

```bash
apt update && apt install -y git
```

```bash
git clone https://github.com/luiscarpio217-commits/trading-engine.git /opt/trading-engine
```

```bash
bash /opt/trading-engine/deploy/setup.sh
```

The setup script takes a couple of minutes. It installs Python, creates an
isolated virtualenv, installs the engine, creates a restricted `trading`
system user, and registers the background service (enabled so it also starts
on reboot). If the repository is private, GitHub will ask for a username and
token during the clone — use a personal access token as the password.

## 4. Set your dashboard username and password

The dashboard will be reachable from the whole internet, so it requires a
login. The service **refuses to start until you set one**.

Run this command **after replacing `pick-a-username` and
`pick-a-long-random-password` with your own values** (keep the quotes):

```bash
tee /etc/trading-engine/env > /dev/null <<'EOF'
DASHBOARD_USERNAME=pick-a-username
DASHBOARD_PASSWORD=pick-a-long-random-password
PYTHONUNBUFFERED=1
EOF
```

Use a long, unique password — this is plain HTTP, so the password protects
you only as much as it is strong. To generate a good one right in the
console:

```bash
openssl rand -base64 24
```

## 5. Start the engine

```bash
systemctl start trading-engine
```

Check that it is running:

```bash
systemctl status trading-engine --no-pager
```

You want to see `Active: active (running)`. To watch the live logs (Ctrl+C
stops watching, not the engine):

```bash
journalctl -u trading-engine -f
```

## 6. Open the dashboard

In your normal browser visit (replace with your droplet's IP):

```
http://YOUR_DROPLET_IP:8000
```

Your browser will prompt for the username and password from step 4. You
should see the dashboard: equity, day P&L (realized and total), positions,
signals, and trades. Outside US market hours (9:30–16:00 US/Eastern,
weekdays) it is normal for everything to be quiet — the engine scans but
does not trade. The server's own clock stays on UTC; market-hours logic
always keys off US/Eastern automatically, including daylight saving.

## Everyday commands

| what                         | command                                      |
|------------------------------|----------------------------------------------|
| status                       | `systemctl status trading-engine --no-pager` |
| live logs                    | `journalctl -u trading-engine -f`            |
| restart                      | `systemctl restart trading-engine`           |
| stop                         | `systemctl stop trading-engine`              |
| performance report           | `/opt/trading-engine/.venv/bin/python -m trading_engine --config /opt/trading-engine/config.yaml report` |
| update to the latest code    | `cd /opt/trading-engine && git pull && .venv/bin/pip install -q -r requirements.txt && systemctl restart trading-engine` |
| change username/password     | edit `/etc/trading-engine/env` (`nano /etc/trading-engine/env`), then `systemctl restart trading-engine` |
| change tickers/settings      | `nano /opt/trading-engine/config.yaml`, then `systemctl restart trading-engine` |

## Troubleshooting

**`Active: failed` right after starting** — almost always missing
credentials. The engine refuses to serve `0.0.0.0` without them. Check the
exact reason with:

```bash
journalctl -u trading-engine -n 30 --no-pager
```

If you see `refusing to bind 0.0.0.0:8000 without authentication`, redo
step 4, then `systemctl restart trading-engine`.

**Browser can't reach the page** — first confirm the service is running and
listening:

```bash
systemctl status trading-engine --no-pager
```

Fresh DigitalOcean droplets have no firewall blocking port 8000. If you (or
a team policy) enabled `ufw` or a DigitalOcean Cloud Firewall, allow the
port: `ufw allow 8000/tcp`, or add an inbound rule for TCP 8000 in the
control panel.

**Login loop / wrong password** — values in `/etc/trading-engine/env` must
have no spaces around `=` and no quotes. Fix, then restart the service.

## Security notes (please read once)

- **Paper mode only.** `config.yaml` pins `broker: paper`; there are no
  broker credentials anywhere on this box, so the worst case is a confused
  paper journal, never real orders.
- Basic auth over plain HTTP is sent unencrypted. For hobby paper trading
  behind a strong password this is a reasonable trade-off; if you want TLS,
  put Caddy or nginx with a domain in front and proxy to `127.0.0.1:8000`
  (then set `web.host: 127.0.0.1` in `config.yaml`).
- The service runs as the unprivileged `trading` user with a read-only view
  of the system (`ProtectSystem=strict`) — it can only write its own
  `data/` journal directory.
