"""
Configuration loader — reads all YAML config files and validates them
using Pydantic models. This ensures the system fails fast on bad config
rather than silently using wrong parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from src.utils.logger import logger


# ─── Pydantic Config Models ───────────────────────────────────────────────────

class UniverseConfig(BaseModel):
    name: str
    description: str
    exchange: str
    currency: str
    data_source: str
    suffix: str
    benchmark: str
    start_date: str
    end_date: str
    train_end: str
    validation_end: str
    oos_start: str
    min_trading_days: int = Field(ge=1)
    symbols: list[str]

    @field_validator("symbols")
    @classmethod
    def symbols_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("symbols list cannot be empty")
        return v


class MomentumSignalConfig(BaseModel):
    name: str
    enabled: bool
    lookback_days: int = Field(ge=20)
    skip_days: int = Field(ge=0)
    top_n: int = Field(ge=1)
    min_valid_stocks: int = Field(ge=1)
    rebalance_freq: str
    min_price: float = Field(ge=0.0)
    min_avg_volume_shares: int = Field(ge=0)


class MeanReversionSignalConfig(BaseModel):
    name: str
    enabled: bool
    rsi_window: int = Field(ge=2)
    rsi_oversold: float = Field(ge=1.0, le=50.0)
    zscore_window: int = Field(ge=5)
    zscore_threshold: float
    max_positions: int = Field(ge=1)
    rebalance_freq: str
    rsi_exit: float = Field(ge=1.0, le=100.0)
    min_history_days: int = Field(ge=1)


class RegimeFilterConfig(BaseModel):
    name: str
    enabled: bool
    vol_window: int = Field(ge=5)
    vol_longrun_window: int = Field(ge=20)
    vol_expansion_threshold: float = Field(ge=1.0)
    high_vol_risk_factor: float = Field(ge=0.0, le=1.0)
    breadth_window: int = Field(ge=20)
    breadth_low_threshold: float = Field(ge=0.0, le=1.0)
    breadth_risk_factor: float = Field(ge=0.0, le=1.0)
    eval_freq: str


class MetaAllocatorConfig(BaseModel):
    name: str
    rolling_sharpe_window: int = Field(ge=10)
    risk_free_rate: float = Field(ge=0.0, le=0.5)
    max_drawdown_gate: float = Field(ge=0.0, le=1.0)
    drawdown_reduction_factor: float = Field(ge=0.0, le=1.0)
    correlation_window: int = Field(ge=5)
    high_correlation_threshold: float = Field(ge=0.0, le=1.0)
    min_weight: float = Field(ge=0.0, le=1.0)
    normalise_weights: bool
    rebalance_freq: str


class SignalsConfig(BaseModel):
    momentum: MomentumSignalConfig
    mean_reversion: MeanReversionSignalConfig
    regime_filter: RegimeFilterConfig


class StrategyConfig(BaseModel):
    signals: SignalsConfig
    meta_allocator: MetaAllocatorConfig


class CostsConfig(BaseModel):
    stt_rate: float
    exchange_charges_rate: float
    sebi_fee_rate: float
    stamp_duty_rate: float
    brokerage_rate: float
    gst_rate: float


class SlippageConfig(BaseModel):
    fixed_bps: float = Field(ge=0.0)
    volume_dependent: bool
    volume_threshold_pct: float


class PortfolioConstraintsConfig(BaseModel):
    max_stock_weight: float = Field(ge=0.0, le=1.0)
    max_positions: int = Field(ge=1)
    min_position_weight: float = Field(ge=0.0)
    cash_buffer: float = Field(ge=0.0, le=1.0)
    rebalance_tolerance: float = Field(ge=0.0)


class RiskConfig(BaseModel):
    portfolio_stop_drawdown: float = Field(ge=0.0, le=1.0)
    hard_max_stock_weight: float = Field(ge=0.0, le=1.0)


class DataPolicyConfig(BaseModel):
    max_fill_days: int = Field(ge=0)
    skip_missing_execution_price: bool
    delisting_threshold_days: int = Field(ge=1)


class BacktestConfig(BaseModel):
    execution: str
    initial_capital: float = Field(ge=1.0)
    costs: CostsConfig
    slippage: SlippageConfig
    portfolio: PortfolioConstraintsConfig
    risk: RiskConfig
    data: DataPolicyConfig
    random_seed: int


class FullBacktestConfig(BaseModel):
    backtest: BacktestConfig


# ─── Loader ──────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a single YAML file and return its contents as a dict."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class AppConfig:
    """Centralised configuration object.

    Load once at startup and pass around by reference.

    Example:
        cfg = AppConfig.from_dir("config/")
        symbols = cfg.universe.symbols
    """

    def __init__(
        self,
        universe: UniverseConfig,
        strategy: StrategyConfig,
        backtest: BacktestConfig,
    ) -> None:
        self.universe = universe
        self.strategy = strategy
        self.backtest = backtest

    @classmethod
    def from_dir(cls, config_dir: str | Path = "config") -> "AppConfig":
        """Load and validate all config files from a directory.

        Args:
            config_dir: Path to directory containing universe.yaml,
                        strategy.yaml, and backtest.yaml.

        Returns:
            Validated AppConfig instance.

        Raises:
            FileNotFoundError: If any config file is missing.
            ValidationError: If config values fail Pydantic validation.
        """
        config_dir = Path(config_dir)
        logger.info(f"Loading configuration from: {config_dir.resolve()}")

        raw_universe = _load_yaml(config_dir / "universe.yaml")
        raw_strategy = _load_yaml(config_dir / "strategy.yaml")
        raw_backtest = _load_yaml(config_dir / "backtest.yaml")

        universe = UniverseConfig(**raw_universe["universe"])
        strategy = StrategyConfig(**raw_strategy)
        backtest = FullBacktestConfig(**raw_backtest).backtest

        logger.info(
            f"Config loaded | Universe: {universe.name} | "
            f"Symbols: {len(universe.symbols)} | "
            f"Period: {universe.start_date} to {universe.end_date}"
        )
        return cls(universe=universe, strategy=strategy, backtest=backtest)


__all__ = ["AppConfig", "UniverseConfig", "StrategyConfig", "BacktestConfig"]
