"""
Signal 1: Cross-Sectional Momentum (12-1 Month).

Ranks stocks by their past 12-month return (excluding the most recent month)
to capture medium-term trend continuation while avoiding short-term reversal effects.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.signals.base import BaseSignal, SignalOutput
from src.features.returns import compute_rolling_returns, cross_sectional_rank
from src.utils.config import MomentumSignalConfig
from src.utils.logger import logger


class MomentumSignal(BaseSignal):
    """Cross-sectional 12-1 month momentum signal.

    Selects top-N stocks based on past 12-month return (excluding the most recent month).
    """

    def __init__(self, cfg: MomentumSignalConfig) -> None:
        self._cfg = cfg

    @property
    def name(self) -> str:
        return "momentum_12_1"

    @property
    def rebalance_freq(self) -> str:
        return self._cfg.rebalance_freq

    def generate(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame | None = None,
        **kwargs,
    ) -> SignalOutput:
        """Generate momentum signal weights.

        Args:
            prices: Adjusted close price panel (Date × Symbol).
            volume: Volume panel for liquidity filter (optional).

        Returns:
            SignalOutput with monthly-frequency weights.
        """
        cfg = self._cfg
        logger.info(
            f"[{self.name}] Generating momentum signal | "
            f"lookback={cfg.lookback_days}d, skip={cfg.skip_days}d, top_n={cfg.top_n}"
        )

        # 1. Compute rolling returns (excluding the most recent skip_days)
        momentum_scores = compute_rolling_returns(
            prices,
            window=cfg.lookback_days,
            skip_days=cfg.skip_days,
        )

        # 2. Filter for stocks above minimum price
        price_filter = prices >= cfg.min_price
        momentum_scores = momentum_scores.where(price_filter)

        # 3. Filter for stocks meeting minimum volume requirement
        if volume is not None:
            vol_30d_avg = volume.rolling(30, min_periods=15).mean()
            volume_filter = vol_30d_avg >= cfg.min_avg_volume_shares
            momentum_scores = momentum_scores.where(volume_filter)

        # 4. Rank stocks cross-sectionally (1 = highest return)
        ranked = cross_sectional_rank(momentum_scores, ascending=True)

        # 5. Generate equal weights for top_n stocks on monthly rebalance dates
        weights_all = pd.DataFrame(
            np.nan, index=prices.index, columns=prices.columns
        )

        rebalance_dates = self._get_monthly_rebalance_dates(prices.index)

        for date in rebalance_dates:
            if date not in ranked.index:
                continue
            scores_row = ranked.loc[date].dropna()
            valid_count = scores_row.notna().sum()

            if valid_count < cfg.min_valid_stocks:
                logger.debug(
                    f"[{self.name}] {date.date()}: insufficient valid stocks "
                    f"({valid_count} < {cfg.min_valid_stocks}) — skipping"
                )
                continue

            top_stocks = scores_row.nlargest(cfg.top_n).index.tolist()
            weight = 1.0 / len(top_stocks)
            row = pd.Series(0.0, index=prices.columns)
            row[top_stocks] = weight
            weights_all.loc[date] = row

        # Hold positions between rebalance dates
        weights_filled = weights_all.ffill().fillna(0.0)

        # Zero-out weights where price data is missing (e.g. delisted stocks)
        weights_filled = weights_filled.where(prices.notna(), 0.0)

        logger.info(
            f"[{self.name}] Signal generated | "
            f"Non-zero rebalances: {weights_all.notna().any(axis=1).sum()}"
        )

        return SignalOutput(
            name=self.name,
            weights=weights_filled,
            scores=momentum_scores,
        )

    @staticmethod
    def _get_monthly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Return last trading day of each month."""
        series = pd.Series(index, index=index)
        monthly = series.resample("ME").last()
        return pd.DatetimeIndex(monthly.dropna().values)


__all__ = ["MomentumSignal"]
