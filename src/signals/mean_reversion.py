"""
Signal 2: Short-Term Mean Reversion (RSI + Z-Score).

A contrarian strategy targeting short-term oversold conditions.
Uses RSI(5) and price Z-score(20) to identify candidate stocks for weekly entry and exit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.base import BaseSignal, SignalOutput
from src.features.returns import compute_rsi, compute_zscore
from src.utils.config import MeanReversionSignalConfig
from src.utils.logger import logger


class MeanReversionSignal(BaseSignal):
    """Short-term mean reversion signal using RSI and price z-score."""

    def __init__(self, cfg: MeanReversionSignalConfig) -> None:
        self._cfg = cfg

    @property
    def name(self) -> str:
        return "mean_reversion_rsi"

    @property
    def rebalance_freq(self) -> str:
        return self._cfg.rebalance_freq

    def generate(
        self,
        prices: pd.DataFrame,
        **kwargs,
    ) -> SignalOutput:
        """Generate mean reversion signal weights.

        Args:
            prices: Adjusted close price panel (Date × Symbol).

        Returns:
            SignalOutput with weekly-frequency weights.
        """
        cfg = self._cfg
        logger.info(
            f"[{self.name}] Generating mean reversion signal | "
            f"RSI({cfg.rsi_window}) < {cfg.rsi_oversold} or "
            f"Z({cfg.zscore_window}) < {cfg.zscore_threshold}"
        )

        # 1. Compute indicators (RSI and Price Z-score)
        rsi = compute_rsi(prices, window=cfg.rsi_window)
        zscore = compute_zscore(prices, window=cfg.zscore_window)

        # 2. Define Entry signals (RSI oversold OR Z-score extreme)
        entry_signal = (rsi < cfg.rsi_oversold) | (zscore < cfg.zscore_threshold)

        # 3. Define Exit signals (RSI exits oversold AND Z-score rises above mean)
        exit_signal = (rsi > cfg.rsi_exit) & (zscore > 0.0)

        # 4. Generate equal weights weekly for active entries
        weights_all = pd.DataFrame(
            np.nan, index=prices.index, columns=prices.columns
        )

        rebalance_dates = self._get_weekly_rebalance_dates(prices.index)
        active_positions: set[str] = set()

        for date in rebalance_dates:
            if date not in entry_signal.index:
                continue

            # Skip dates with insufficient price history
            lookback_start = prices.index[
                max(0, prices.index.get_loc(date) - cfg.min_history_days)
            ]
            if (date - lookback_start).days < cfg.min_history_days:
                continue

            # Remove exited positions
            if exit_signal.index.get_loc(date) >= 0 and date in exit_signal.index:
                exiting = set(
                    exit_signal.loc[date][
                        exit_signal.loc[date] & pd.Series(
                            {s: True for s in active_positions},
                            dtype=bool
                        ).reindex(exit_signal.columns, fill_value=False)
                    ].index.tolist()
                )
                active_positions -= exiting

            # Add new entry positions, prioritizing the most oversold by RSI
            if date in entry_signal.index:
                entry_row = entry_signal.loc[date]
                new_entries = set(entry_row[entry_row].index.tolist())
                rsi_row = rsi.loc[date]
                new_entries_ranked = (
                    rsi_row[list(new_entries)]
                    .dropna()
                    .nsmallest(cfg.max_positions)
                    .index.tolist()
                )
                active_positions.update(new_entries_ranked)

            # Cap active positions to max limit
            if len(active_positions) > cfg.max_positions:
                rsi_row = rsi.loc[date].reindex(list(active_positions))
                active_positions = set(
                    rsi_row.nsmallest(cfg.max_positions).index.tolist()
                )

            # Assign equal weights to active holdings
            if active_positions:
                valid_positions = [
                    s for s in active_positions
                    if s in prices.columns and not pd.isna(prices.loc[date, s])
                ]
                if valid_positions:
                    weight = 1.0 / len(valid_positions)
                    row = pd.Series(0.0, index=prices.columns)
                    row[valid_positions] = weight
                    weights_all.loc[date] = row
                else:
                    weights_all.loc[date] = 0.0
            else:
                weights_all.loc[date] = 0.0

        # Hold weights constant between rebalance dates
        weights_filled = weights_all.ffill().fillna(0.0)

        # Zero-out weights where price is missing (delistings / gaps)
        weights_filled = weights_filled.where(prices.notna(), 0.0)

        logger.info(
            f"[{self.name}] Signal generated | "
            f"Rebalance dates with positions: "
            f"{(weights_all.sum(axis=1) > 0).sum()}"
        )

        return SignalOutput(
            name=self.name,
            weights=weights_filled,
            scores=rsi,
            metadata={"zscore": zscore},
        )

    @staticmethod
    def _get_weekly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Return last trading day of each week."""
        series = pd.Series(index, index=index)
        weekly = series.resample("W").last()
        return pd.DatetimeIndex(weekly.dropna().values)


__all__ = ["MeanReversionSignal"]
