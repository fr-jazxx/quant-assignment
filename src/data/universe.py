"""
Universe management — resolves symbol lists, handles aliases,
and provides a clean interface to the configured stock universe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.utils.config import UniverseConfig
from src.utils.logger import logger


@dataclass
class Universe:
    """Represents the investable stock universe for a backtest run.

    Attributes:
        symbols: List of ticker symbols (with exchange suffix, e.g. 'TCS.NS').
        benchmark: Benchmark index ticker (e.g. '^NSEI').
        start_date: Start of the data / backtest window.
        end_date: End of the data / backtest window.
        train_end: Last date of the training period.
        validation_end: Last date of the validation period.
        oos_start: First date of the out-of-sample period.
        suffix: Exchange suffix appended to raw tickers (e.g. '.NS').
        name: Human-readable universe name.
    """

    symbols: list[str]
    benchmark: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    oos_start: pd.Timestamp
    suffix: str
    name: str
    _active_symbols: list[str] = field(default_factory=list, repr=False)

    @classmethod
    def from_config(cls, cfg: UniverseConfig) -> "Universe":
        """Build a Universe from the validated config object.

        Args:
            cfg: Validated UniverseConfig from AppConfig.

        Returns:
            Universe instance ready to use.
        """
        universe = cls(
            symbols=cfg.symbols,
            benchmark=cfg.benchmark,
            start_date=pd.Timestamp(cfg.start_date),
            end_date=pd.Timestamp(cfg.end_date),
            train_end=pd.Timestamp(cfg.train_end),
            validation_end=pd.Timestamp(cfg.validation_end),
            oos_start=pd.Timestamp(cfg.oos_start),
            suffix=cfg.suffix,
            name=cfg.name,
        )
        logger.info(
            f"Universe '{universe.name}' built | "
            f"{len(universe.symbols)} symbols | "
            f"{cfg.start_date} → {cfg.end_date}"
        )
        return universe

    @property
    def active_symbols(self) -> list[str]:
        """Symbols confirmed to have been successfully fetched."""
        return self._active_symbols if self._active_symbols else self.symbols

    def set_active_symbols(self, symbols: list[str]) -> None:
        """Update the active symbol list after data availability check."""
        removed = set(self.symbols) - set(symbols)
        if removed:
            logger.warning(
                f"Removed {len(removed)} symbols due to data unavailability: "
                f"{sorted(removed)}"
            )
        self._active_symbols = symbols
        logger.info(f"Active universe: {len(self._active_symbols)} symbols")

    def get_period_mask(
        self,
        index: pd.DatetimeIndex,
        period: str = "full",
    ) -> pd.Series:
        """Return a boolean mask for a given period.

        Args:
            index: DatetimeIndex to apply the mask to.
            period: One of 'full', 'train', 'validation', 'oos'.

        Returns:
            Boolean Series aligned to index.
        """
        period_map = {
            "full": (self.start_date, self.end_date),
            "train": (self.start_date, self.train_end),
            "validation": (self.train_end, self.validation_end),
            "oos": (self.oos_start, self.end_date),
        }
        if period not in period_map:
            raise ValueError(
                f"Unknown period '{period}'. Choose from: {list(period_map)}"
            )
        start, end = period_map[period]
        return pd.Series(
            (index >= start) & (index <= end),
            index=index,
            name=period,
        )

    def __len__(self) -> int:
        return len(self.active_symbols)

    def __repr__(self) -> str:
        return (
            f"Universe(name='{self.name}', symbols={len(self.symbols)}, "
            f"active={len(self.active_symbols)}, "
            f"period='{self.start_date.date()}→{self.end_date.date()}')"
        )


__all__ = ["Universe"]
