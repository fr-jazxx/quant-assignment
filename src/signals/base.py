"""
Abstract base class for all trading signals.

Every signal must implement generate() which returns a DataFrame of
weights indexed by date and symbol. Weights represent desired portfolio
allocation — they are NOT guaranteed to sum to 1.0. The portfolio
construction layer handles normalisation and constraints.

Signal Design Contract:
    - generate() must only use data available UP TO date t.
    - The returned weights are applied at open of date t+1 (enforced by engine).
    - Weights of 0.0 mean "no position" (not "sell short").
    - NaN weights mean "unknown / insufficient data" — engine will skip.
    - All signals must declare their rebalance_freq for the scheduler.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from src.utils.logger import logger


@dataclass
class SignalOutput:
    """Structured output from a signal generator.

    Attributes:
        name: Signal identifier.
        weights: DataFrame of target weights (Date × Symbol). Range [0, 1].
        scores: Raw underlying scores before ranking/filtering (optional).
        metadata: Any additional debug info per date.
    """

    name: str
    weights: pd.DataFrame          # (Date × Symbol), values in [0, 1]
    scores: pd.DataFrame | None = None
    metadata: dict = None          # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class BaseSignal(ABC):
    """Abstract base class for all trading signals.

    Subclasses must implement:
        - name (property): Unique identifier for this signal.
        - rebalance_freq (property): 'daily', 'weekly', or 'monthly'.
        - generate(prices, **kwargs): Returns SignalOutput.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique signal identifier."""
        ...

    @property
    @abstractmethod
    def rebalance_freq(self) -> str:
        """Rebalance frequency: 'daily', 'weekly', or 'monthly'."""
        ...

    @abstractmethod
    def generate(
        self,
        prices: pd.DataFrame,
        **kwargs,
    ) -> SignalOutput:
        """Generate signal weights from adjusted close prices.

        IMPORTANT: This method must ONLY use data available at time t.
        Any use of future data constitutes look-ahead bias.

        Args:
            prices: Wide DataFrame of adjusted close prices (Date × Symbol).
            **kwargs: Additional data (e.g. volume, index prices).

        Returns:
            SignalOutput with weights DataFrame.
        """
        ...

    def validate_no_lookahead(
        self,
        signal_date: pd.Timestamp,
        prices: pd.DataFrame,
    ) -> bool:
        """Assert that prices used for signal don't extend beyond signal_date.

        This is a runtime guard against accidental look-ahead bias.

        Args:
            signal_date: The date for which the signal is being generated.
            prices: The price data being used.

        Returns:
            True if no violation, raises AssertionError otherwise.
        """
        max_price_date = prices.index.max()
        if max_price_date > signal_date:
            raise AssertionError(
                f"[{self.name}] Look-ahead bias detected! "
                f"Signal date: {signal_date.date()}, "
                f"Max price date in data: {max_price_date.date()}. "
                f"Future data is being used."
            )
        return True

    def get_rebalance_dates(
        self,
        index: pd.DatetimeIndex,
        freq: str | None = None,
    ) -> pd.DatetimeIndex:
        """Return the subset of dates that are rebalance dates.

        Args:
            index: Full trading date index.
            freq: Override rebalance frequency. Defaults to self.rebalance_freq.

        Returns:
            DatetimeIndex of rebalance dates.
        """
        freq = freq or self.rebalance_freq
        if freq == "daily":
            return index
        elif freq == "weekly":
            # Last trading day of each week
            return index[index.to_series().dt.dayofweek == 4].union(
                index.groupby(index.to_period("W")).agg("last")
            )
        elif freq == "monthly":
            # Last trading day of each month
            return index[
                index.to_series().dt.is_month_end
                | (index.to_series().dt.month != index.to_series().shift(-1).dt.month)
            ]
        else:
            raise ValueError(
                f"Unknown rebalance_freq: '{freq}'. Use 'daily', 'weekly', 'monthly'."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', freq='{self.rebalance_freq}')"


__all__ = ["BaseSignal", "SignalOutput"]
