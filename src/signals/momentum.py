"""
Signal 1: Cross-Sectional Momentum (12-1 Month)

Economic Rationale:
    Jegadeesh and Titman (1993) documented that stocks with strong 12-month
    returns continue to outperform over the next 3-12 months. This anomaly
    is attributed to behavioural factors:
      - Underreaction to information: investors are slow to update beliefs
      - Herding: institutional investors chase performance
      - Disposition effect: winners are sold too early, losers held too long

    The "12-1" construction (12 months excluding the most recent month) avoids
    short-term reversal contamination documented by Jegadeesh (1990).

    Indian market evidence: Momentum has been documented in Indian equities
    by Sehgal & Balakrishnan (2002) and more recently in several NSE working
    papers, though it is weaker and more crash-prone than in US markets.

Failure Conditions:
    - Momentum crashes: Following sharp market drawdowns, prior winners
      crash hardest during recovery as short-sellers cover. March 2020 is
      a clear example in Indian markets.
    - Highly correlated market environment: When all stocks move together,
      cross-sectional dispersion collapses and signal strength deteriorates.
    - Whipsaw: In choppy, directionless markets, momentum frequently reverses.

References:
    - Jegadeesh, N. & Titman, S. (1993). Returns to Buying Winners and Selling
      Losers: Implications for Stock Market Efficiency. Journal of Finance, 48(1).
    - Asness, C., Moskowitz, T. & Pedersen, L. (2013). Value and Momentum
      Everywhere. Journal of Finance, 68(3).
    - Sehgal, S. & Balakrishnan, I. (2002). Contrarian and Momentum Strategies
      in the Indian Capital Market. Vikalpa, 27(1).
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

    Ranks stocks by their past 12-month return (excluding the most recent
    month) and generates equal weights for the top-N stocks.

    On each rebalance date t:
        1. Compute return from (t - 252) to (t - 21) for all symbols.
        2. Rank stocks cross-sectionally.
        3. Select top_n stocks.
        4. Assign equal weight to selected stocks (normalised to 1.0 total).
        5. Apply minimum price and volume filters before ranking.
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

        # ── Step 1: Compute 12-1 rolling returns ──
        momentum_scores = compute_rolling_returns(
            prices,
            window=cfg.lookback_days,
            skip_days=cfg.skip_days,
        )

        # ── Step 2: Liquidity filter — remove stocks below min price ──
        price_filter = prices >= cfg.min_price
        momentum_scores = momentum_scores.where(price_filter)

        # ── Step 3: Volume filter (if volume data available) ──
        if volume is not None:
            vol_30d_avg = volume.rolling(30, min_periods=15).mean()
            volume_filter = vol_30d_avg >= cfg.min_avg_volume_shares
            momentum_scores = momentum_scores.where(volume_filter)

        # ── Step 4: Cross-sectional rank (percentile) ──
        ranked = cross_sectional_rank(momentum_scores, ascending=True)
        # ranked is now in [0, 1] where 1 = highest momentum

        # ── Step 5: Select top_n on each date, assign equal weight ──
        weights_all = pd.DataFrame(
            np.nan, index=prices.index, columns=prices.columns
        )

        # Get rebalance dates
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

        # Forward-fill weights between rebalance dates (hold positions)
        weights_filled = weights_all.ffill()
        # First valid rebalance: before that, no positions
        weights_filled = weights_filled.fillna(0.0)

        # Zero-out positions where price data is missing (delisted, etc.)
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
