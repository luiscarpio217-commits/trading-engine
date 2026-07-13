from datetime import time
from pathlib import Path

import pytest

from trading_engine.config import Config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_defaults():
    cfg = Config()
    assert cfg.risk.max_risk_per_trade_pct == pytest.approx(0.01)
    assert cfg.execution.broker == "paper"
    assert cfg.filters.start_time() == time(9, 30)
    assert cfg.filters.end_time() == time(16, 0)


def test_example_config_loads():
    cfg = Config.load(REPO_ROOT / "config.example.yaml")
    assert cfg.engine.tickers == ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]
    assert cfg.strategies.momentum_breakout.enabled
    assert cfg.risk.max_daily_loss_pct == pytest.approx(0.03)
    assert cfg.execution.tradier.sandbox is True


def test_partial_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "engine:\n  tickers: [AMD]\nrisk:\n  max_risk_per_trade_pct: 0.02\n"
        "strategies:\n  options_flow:\n    enabled: false\n"
        "unknown_section:\n  ignored: true\n")
    cfg = Config.load(p)
    assert cfg.engine.tickers == ["AMD"]
    assert cfg.risk.max_risk_per_trade_pct == pytest.approx(0.02)
    assert cfg.strategies.options_flow.enabled is False
    # untouched defaults survive
    assert cfg.strategies.momentum_breakout.breakout_lookback == 20
