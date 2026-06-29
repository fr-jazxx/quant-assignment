"""
Signal 2: Short-Term Mean Reversion (RSI + Z-Score)

Economic Rationale:
    De Bondt and Thaler (1985) documented that stocks experiencing extreme
    negative short-term returns tend to rebound — the market "overreacts"
    to recent information. At shorter horizons (days to weeks), this is
    amplified by:
      - Liquidity-driven selling pressure: forced liquidations push prices
        temporarily below fair value
      - Market microstructure: bid-ask bounce, inventory risk in market-making
      - Panic selling by retail investors during sharp drawdowns

    This signal uses two complementary measures of oversold conditions:
      1. RSI (Relative Strength Index): captures momentum exhaustion
      2. Z-score: captures statistical deviation from recent mean

    The signal is CONTRARIAN — it bets that recent losers will recover.
    It is intentionally designed to be negatively correlated with the
    momentum signal, providing diversification at the portfolio level.

Failure Conditions:
    - Value traps: Some stocks are "cheap" for fundamental reasons and
      continue declining. Without fundamental data, we cannot distinguish.
    - Trend continuation: In strongly trending down-markets, mean reversion
      signals generate sustained false positives ("catching falling knives").
    - Sector crashes: Industry-wide shocks don't revert quickly.

References:
    - De Bondt, W.F.M. & Thaler, R. (1985). Does the Stock Market Overreact?
      Journal of Finance, 40(3).
    - Lehmann, B.N. (1990). Fads, Martingales, and Market Efficiency.
      Quarterly Journal of Economics, 105(1).
    - Jegadeesh, N. (1990). Evidence of Predictable Behavior of Security Returns.
      Journal of Finance, 45(3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.base import BaseSignal, SignalOutput
from src.features.returns import compute_rsi, compute_zscore
from src.utils.config import MeanReversionSignalConfig
from src.utils.logger import logger


class MeanReversionSignal(BaseSignal):
    """Short-term mean reversion signal using RSI and price z-score.

    Entry (long) conditions on date t:
        - RSI(5) < oversold_threshold (default: 30), OR
        - Z-score(20) < -zscore_threshold (default: -2.0)

    Exit conditions:
        - RSI(5) > rsi_exit (default: 50), OR
        - Z-score returns to > 0 (mean)

    On each weekly rebalance:
        1. Identify all stocks meeting entry conditions.
        2. Rank by RSI ascending (most oversold first).
        3. Select top max_positions stocks.
        4. Assign equal weight.
    """

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

        # ── Compute indicators ──
        rsi = compute_rsi(prices, window=cfg.rsi_window)
        zscore = compute_zscore(prices, window=cfg.zscore_window)

        # ── Entry signals: either RSI oversold OR z-score extreme ──
        rsi_signal = rsi < cfg.rsi_oversold         # True = oversold by RSI
        zscore_signal = zscore < cfg.zscore_threshold  # True = oversold by z-score
        entry_signal = rsi_signal | zscore_signal    # Union of both

        # ── Exit signals ──
        rsi_exit = rsi > cfg.rsi_exit
        zscore_exit = zscore > 0.0
        exit_signal = rsi_exit & zscore_exit         # Both must recover

        # ── Build weights on each weekly rebalance date ──
        weights_all = pd.DataFrame(
            np.nan, index=prices.index, columns=prices.columns
        )

        rebalance_dates = self._get_weekly_rebalance_dates(prices.index)
        active_positions: set[str] = set()

        for date in rebalance_dates:
            if date not in entry_signal.index:
                continue

            # Skip if insufficient history
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

            # Add new entry positions
            if date in entry_signal.index:
                entry_row = entry_signal.loc[date]
                new_entries = set(entry_row[entry_row].index.tolist())
                # Rank by RSI ascending (most oversold first) for tie-breaking
                rsi_row = rsi.loc[date]
                new_entries_ranked = (
                    rsi_row[list(new_entries)]
                    .dropna()
                    .nsmallest(cfg.max_positions)
                    .index.tolist()
                )
                active_positions.update(new_entries_ranked)

            # Enforce max_positions cap (keep most oversold)
            if len(active_positions) > cfg.max_positions:
                rsi_row = rsi.loc[date].reindex(list(active_positions))
                active_positions = set(
                    rsi_row.nsmallest(cfg.max_positions).index.tolist()
                )

            # Assign equal weights
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

        # Forward-fill between rebalance dates
        weights_filled = weights_all.ffill().fillna(0.0)

        # Zero-out where prices are missing
        weights_filled = weights_filled.where(prices.notna(), 0.0)

        logger.info(
            f"[{self.name}] Signal generated | "
            f"Rebalance dates with positions: "
            f"{(weights_all.sum(axis=1) > 0).sum()}"
        )

        return SignalOutput(
            name=self.name,
            weights=weights_filled,
            scores=rsi,  # RSI as the primary score
            metadata={"zscore": zscore},
        )

    @staticmethod
    def _get_weekly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Return last trading day of each week (Friday or last available)."""
        series = pd.Series(index, index=index)
        weekly = series.resample("W").last()
        return pd.DatetimeIndex(weekly.dropna().values)


__all__ = ["MeanReversionSignal"]
