"""
Rebalance scheduler — determines which dates trigger portfolio rebalances
for each signal and coordinates the combined schedule.

Design:
    Different signals rebalance at different frequencies:
        - Momentum: Monthly (last trading day of each month)
        - Mean Reversion: Weekly (last trading day of each week)
        - Regime Filter: Daily (evaluated every day)

    The combined rebalance schedule is the union of all signal schedules.
    The engine checks per-signal rebalance dates to avoid unnecessary trades.

Turnover Tracking:
    Turnover is computed on each rebalance as:
        turnover = 0.5 × Σ |new_weight_i - old_weight_i|
    Annualised turnover = mean(monthly_turnover) × 12
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import logger


class RebalanceScheduler:
    """Manages rebalance date schedules for multiple signals.

    Usage:
        scheduler = RebalanceScheduler(trading_dates)
        scheduler.register("momentum", "monthly")
        scheduler.register("mean_reversion", "weekly")
        if scheduler.is_rebalance_date("momentum", date):
            ...
    """

    FREQ_MAP = {
        "daily": "D",
        "weekly": "W",
        "monthly": "ME",
    }

    def __init__(self, trading_dates: pd.DatetimeIndex) -> None:
        self._trading_dates = trading_dates
        self._schedules: dict[str, pd.DatetimeIndex] = {}

    def register(self, signal_name: str, freq: str) -> None:
        """Register a signal with its rebalance frequency.

        Args:
            signal_name: Unique identifier for the signal.
            freq: 'daily', 'weekly', or 'monthly'.
        """
        if freq not in self.FREQ_MAP:
            raise ValueError(
                f"Unknown frequency: '{freq}'. Choose from: {list(self.FREQ_MAP)}"
            )

        if freq == "daily":
            dates = self._trading_dates
        elif freq == "weekly":
            dates = self._last_day_of_period(self._trading_dates, "W")
        elif freq == "monthly":
            dates = self._last_day_of_period(self._trading_dates, "ME")
        else:
            dates = self._trading_dates

        self._schedules[signal_name] = dates
        logger.debug(
            f"Registered rebalance schedule: {signal_name} ({freq}) | "
            f"{len(dates)} rebalance dates"
        )

    def is_rebalance_date(self, signal_name: str, date: pd.Timestamp) -> bool:
        """Check if a date is a rebalance date for a given signal.

        Args:
            signal_name: Signal to check.
            date: Date to check.

        Returns:
            True if this date triggers a rebalance for the signal.
        """
        if signal_name not in self._schedules:
            raise KeyError(f"Signal '{signal_name}' not registered")
        return date in self._schedules[signal_name]

    def get_combined_schedule(self) -> pd.DatetimeIndex:
        """Return the union of all signal rebalance dates."""
        if not self._schedules:
            return self._trading_dates
        combined = pd.DatetimeIndex([])
        for dates in self._schedules.values():
            combined = combined.union(dates)
        return combined.sort_values()

    def compute_rebalance_turnover(
        self,
        weights_history: pd.DataFrame,
        rebalance_dates: pd.DatetimeIndex,
    ) -> pd.Series:
        """Compute one-way turnover at each rebalance date.

        Args:
            weights_history: DataFrame (Date × Symbol) of portfolio weights.
            rebalance_dates: DatetimeIndex of rebalance dates.

        Returns:
            Series of one-way turnover values indexed by rebalance date.
        """
        turnovers: dict[pd.Timestamp, float] = {}
        prev_weights = pd.Series(dtype=float)

        for date in rebalance_dates:
            if date not in weights_history.index:
                continue
            curr_weights = weights_history.loc[date].fillna(0.0)
            if prev_weights.empty:
                prev_weights = curr_weights
                continue
            aligned = curr_weights.reindex(
                prev_weights.index.union(curr_weights.index), fill_value=0.0
            )
            prev_aligned = prev_weights.reindex(aligned.index, fill_value=0.0)
            turnover = (aligned - prev_aligned).abs().sum() / 2.0
            turnovers[date] = float(turnover)
            prev_weights = curr_weights

        return pd.Series(turnovers)

    @staticmethod
    def _last_day_of_period(
        dates: pd.DatetimeIndex,
        freq: str,
    ) -> pd.DatetimeIndex:
        """Return the last trading date in each period."""
        series = pd.Series(dates, index=dates)
        resampled = series.resample(freq).last().dropna()
        return pd.DatetimeIndex(resampled.values)


__all__ = ["RebalanceScheduler"]
