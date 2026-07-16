#!/usr/bin/env bash
# One-shot server setup for the day trading engine (paper mode).
#
# Expects the repo cloned at /opt/trading-engine (see deploy/README.md):
#   git clone https://github.com/luiscarpio217-commits/trading-engine.git /opt/trading-engine
#   bash /opt/trading-engine/deploy/setup.sh
#
# What it does:
#   * installs Python + venv + tzdata via apt
#   * creates a dedicated system user `trading`
#   * builds the virtualenv and installs the engine into it
#   * seeds /opt/trading-engine/config.yaml from deploy/config.server.yaml
#   * seeds /etc/trading-engine/env (dashboard credentials go here)
#   * installs and enables the systemd service (start is a separate,
#     deliberate step after you set credentials)

set -euo pipefail

APP_DIR="/opt/trading-engine"
ENV_FILE="/etc/trading-engine/env"
SERVICE="trading-engine"

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: run as root (on a fresh droplet you already are; otherwise: sudo bash deploy/setup.sh)" >&2
    exit 1
fi
if [[ ! -f "$APP_DIR/pyproject.toml" ]]; then
    echo "ERROR: repo not found at $APP_DIR" >&2
    echo "clone it first:  git clone https://github.com/luiscarpio217-commits/trading-engine.git $APP_DIR" >&2
    exit 1
fi

echo "==> installing system packages (python, venv, tzdata)"
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip tzdata

echo "==> creating system user 'trading' (no login shell)"
id -u trading &>/dev/null || useradd --system --home-dir "$APP_DIR" --shell /usr/sbin/nologin trading

echo "==> building virtualenv and installing the engine"
cd "$APP_DIR"
python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt
.venv/bin/pip install --quiet -e .

echo "==> seeding server config and journal directory"
if [[ ! -f config.yaml ]]; then
    cp deploy/config.server.yaml config.yaml
    echo "    created $APP_DIR/config.yaml (paper broker, 0.0.0.0:8000)"
else
    echo "    keeping existing $APP_DIR/config.yaml"
fi
mkdir -p data
chown -R trading:trading "$APP_DIR"

echo "==> seeding $ENV_FILE (dashboard credentials live here)"
mkdir -p /etc/trading-engine
if [[ ! -f "$ENV_FILE" ]]; then
    cat > "$ENV_FILE" <<'EOF'
# Dashboard basic-auth credentials - REQUIRED.
# The service refuses to start on 0.0.0.0 while these are empty.
DASHBOARD_USERNAME=
DASHBOARD_PASSWORD=
PYTHONUNBUFFERED=1
EOF
    chmod 600 "$ENV_FILE"
    echo "    created $ENV_FILE - set your username/password next (see deploy/README.md)"
else
    echo "    keeping existing $ENV_FILE"
fi

echo "==> installing and enabling the systemd service"
cp deploy/trading-engine.service "/etc/systemd/system/$SERVICE.service"
systemctl daemon-reload
systemctl enable "$SERVICE"

echo
echo "Setup complete. Next steps (see deploy/README.md):"
echo "  1) set your dashboard credentials:   nano $ENV_FILE"
echo "  2) start the engine:                 systemctl start $SERVICE"
echo "  3) check it:                         systemctl status $SERVICE --no-pager"
echo "  4) open http://<your-droplet-ip>:8000 and log in"
