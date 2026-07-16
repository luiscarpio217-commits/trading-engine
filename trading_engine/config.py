"""Configuration: dataclass defaults, YAML overrides, env-var credentials."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import time
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class EngineSettings:
    tickers: list[str] = field(default_factory=lambda: ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"])
    scan_interval_seconds: int = 60
    manage_interval_seconds: int = 10
    db_path: str = "data/trading.db"
    csv_dir: str = "data"
    log_level: str = "INFO"


@dataclass
class DataSettings:
    intraday_interval: str = "5m"
    intraday_lookback_days: int = 5
    daily_lookback_days: int = 120
    options_provider: str = "yfinance"        # yfinance | tradier
    max_chain_expiries: int = 3
    quote_cache_seconds: int = 10
    intraday_cache_seconds: int = 30
    daily_cache_seconds: int = 3600
    chain_cache_seconds: int = 300


@dataclass
class FilterSettings:
    market_hours_only: bool = True
    timezone: str = "America/New_York"
    session_start: str = "09:30"
    session_end: str = "16:00"
    eod_flatten: bool = True
    eod_flatten_time: str = "15:55"
    min_avg_daily_volume: float = 1_000_000
    earnings_blackout_days: int = 2

    def start_time(self) -> time:
        return _parse_hhmm(self.session_start)

    def end_time(self) -> time:
        return _parse_hhmm(self.session_end)

    def flatten_time(self) -> time:
        return _parse_hhmm(self.eod_flatten_time)


@dataclass
class MomentumSettings:
    enabled: bool = True
    breakout_lookback: int = 20
    volume_multiple: float = 1.5
    rsi_min: float = 50.0
    rsi_max: float = 80.0
    atr_stop_multiple: float = 1.5
    reward_risk: float = 2.0
    allow_short: bool = True
    cooldown_minutes: int = 30
    min_dte: int = 1
    target_delta: float = 0.40


@dataclass
class OptionsFlowSettings:
    enabled: bool = True
    max_dte: int = 14
    min_volume: int = 500
    min_premium: float = 0.10        # ignore sub-dime lottery tickets in the scan
    volume_oi_ratio: float = 2.0
    iv_multiple: float = 1.25
    moneyness_pct: float = 0.10
    dominance_ratio: float = 1.5
    stop_pct: float = 0.01
    reward_risk: float = 2.0
    cooldown_minutes: int = 60


@dataclass
class StrategySettings:
    momentum_breakout: MomentumSettings = field(default_factory=MomentumSettings)
    options_flow: OptionsFlowSettings = field(default_factory=OptionsFlowSettings)


@dataclass
class RiskSettings:
    max_risk_per_trade_pct: float = 0.01       # fraction of equity risked per trade
    max_daily_loss_pct: float = 0.03           # daily loss auto-shutoff threshold
    sizing_method: str = "fixed_fractional"    # fixed_fractional | kelly
    kelly_multiplier: float = 0.5              # half-Kelly by default
    kelly_cap: float = 0.25                    # never risk more than this Kelly fraction
    kelly_min_trades: int = 20                 # history needed before Kelly kicks in
    max_open_positions: int = 5
    max_position_notional_pct: float = 0.20    # cap on position notional vs equity
    premium_stop_pct: float = 0.50             # option premium stop (fraction of entry premium)
    min_option_premium: float = 0.10           # never trade contracts cheaper than this
    max_option_contracts: int = 25             # hard cap on contracts per trade
    flatten_on_shutoff: bool = True


@dataclass
class PaperSettings:
    starting_cash: float = 100_000.0
    slippage_bps: float = 5.0


@dataclass
class AlpacaSettings:
    paper: bool = True
    key_id_env: str = "ALPACA_API_KEY_ID"
    secret_env: str = "ALPACA_API_SECRET_KEY"

    @property
    def key_id(self) -> str:
        return os.environ.get(self.key_id_env, "")

    @property
    def secret(self) -> str:
        return os.environ.get(self.secret_env, "")

    @property
    def base_url(self) -> str:
        return ("https://paper-api.alpaca.markets" if self.paper
                else "https://api.alpaca.markets")


@dataclass
class TradierSettings:
    sandbox: bool = True
    token_env: str = "TRADIER_ACCESS_TOKEN"
    account_env: str = "TRADIER_ACCOUNT_ID"

    @property
    def token(self) -> str:
        return os.environ.get(self.token_env, "")

    @property
    def account_id(self) -> str:
        return os.environ.get(self.account_env, "")

    @property
    def base_url(self) -> str:
        return ("https://sandbox.tradier.com/v1" if self.sandbox
                else "https://api.tradier.com/v1")


@dataclass
class ExecutionSettings:
    broker: str = "paper"                      # paper | alpaca | tradier
    trade_options: bool = True                 # express signals as option orders when possible
    order_type: str = "market"                 # market | limit
    limit_offset_bps: float = 10.0             # limit price offset from signal entry
    paper: PaperSettings = field(default_factory=PaperSettings)
    alpaca: AlpacaSettings = field(default_factory=AlpacaSettings)
    tradier: TradierSettings = field(default_factory=TradierSettings)


@dataclass
class WebSettings:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class Config:
    engine: EngineSettings = field(default_factory=EngineSettings)
    data: DataSettings = field(default_factory=DataSettings)
    filters: FilterSettings = field(default_factory=FilterSettings)
    strategies: StrategySettings = field(default_factory=StrategySettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
    web: WebSettings = field(default_factory=WebSettings)

    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> "Config":
        """Build config from defaults, overridden by a YAML file if given."""
        cfg = cls()
        if path is None:
            return cfg
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config file {path} must contain a mapping")
        _apply_overrides(cfg, raw)
        return cfg


def _parse_hhmm(value: str) -> time:
    hh, mm = value.strip().split(":")
    return time(int(hh), int(mm))


def _apply_overrides(obj: Any, overrides: dict) -> None:
    """Recursively apply a dict of overrides onto nested dataclasses."""
    valid = {f.name: f for f in fields(obj)}
    for key, value in overrides.items():
        if key not in valid:
            continue  # ignore unknown keys so configs stay forward-compatible
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_overrides(current, value)
        else:
            setattr(obj, key, value)
