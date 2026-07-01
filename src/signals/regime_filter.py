"""
Signal 3: Volatility Regime and Market Breadth Filter.

A risk overlay that scales portfolio exposure between [0, 1] using:
  1. Realised index volatility compared to its long-run average.
  2. Market breadth (fraction of stocks above their 200-day SMA).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.base import BaseSignal, SignalOutput
from src.features.returns import compute_index_realized_vol, compute_market_breadth
from src.utils.config import RegimeFilterConfig
from src.utils.logger import logger


class RegimeFilterSignal(BaseSignal):
    """Volatility regime and market breadth overlay.

    Outputs a daily scalar risk multiplier in [0, 1] applied to combined weights.
    """

    def __init__(self, cfg: RegimeFilterConfig) -> None:
        self._cfg = cfg

    @property
    def name(self) -> str:
        return "regime_filter_vol_breadth"

    @property
    def rebalance_freq(self) -> str:
        return self._cfg.eval_freq

    def generate(
        self,
        prices: pd.DataFrame,
        index_prices: pd.Series | None = None,
        **kwargs,
    ) -> SignalOutput:
        """Generate daily regime risk multipliers.

        Args:
            prices: Adjusted close price panel (Date × Symbol).
            index_prices: Benchmark index adjusted close prices.

        Returns:
            SignalOutput where weights is a single-column DataFrame named
            'regime_multiplier' with values in [0, 1].
        """
        cfg = self._cfg
        logger.info(
            f"[{self.name}] Computing regime filter | "
            f"vol_threshold={cfg.vol_expansion_threshold}x, "
            f"breadth_threshold={cfg.breadth_low_threshold}"
        )

        # 1. Align or calculate benchmark index prices
        if index_prices is None or index_prices.empty:
            logger.warning(
                f"[{self.name}] No index prices provided — using cross-sectional mean as proxy"
            )
            index_prices = prices.mean(axis=1)
        else:
            index_prices = index_prices.reindex(prices.index).ffill()

        # 2. Compute realized volatility ratio and determine volatility factor
        vol_data = compute_index_realized_vol(
            index_prices,
            short_window=cfg.vol_window,
            long_window=cfg.vol_longrun_window,
        )
        vol_ratio = vol_data["vol_ratio"]
        vol_factor = pd.Series(1.0, index=prices.index)
        high_vol_mask = vol_ratio > cfg.vol_expansion_threshold
        vol_factor[high_vol_mask] = cfg.high_vol_risk_factor

        # 3. Compute market breadth and determine breadth factor
        breadth = compute_market_breadth(prices, sma_window=cfg.breadth_window)
        breadth_factor = pd.Series(1.0, index=prices.index)
        low_breadth_mask = breadth < cfg.breadth_low_threshold
        breadth_factor[low_breadth_mask] = cfg.breadth_risk_factor

        # 4. Combined risk multiplier (most restrictive of vol or breadth)
        multiplier = pd.concat([vol_factor, breadth_factor], axis=1).min(axis=1)

        # Construct metadata for debugging and analysis
        regime_df = pd.DataFrame(
            {
                "regime_multiplier": multiplier,
                "vol_ratio": vol_ratio,
                "vol_factor": vol_factor,
                "breadth": breadth,
                "breadth_factor": breadth_factor,
                "high_vol_regime": high_vol_mask.astype(int),
                "low_breadth_regime": low_breadth_mask.astype(int),
            }
        )

        n_high_vol = high_vol_mask.sum()
        n_low_breadth = low_breadth_mask.sum()
        n_reduced = (multiplier < 1.0).sum()
        pct_reduced = n_reduced / max(len(multiplier), 1) * 100
        logger.info(
            f"[{self.name}] Regime stats | "
            f"High-vol days: {n_high_vol} | "
            f"Low-breadth days: {n_low_breadth} | "
            f"Risk-reduced days: {n_reduced} ({pct_reduced:.1f}%)"
        )

        return SignalOutput(
            name=self.name,
            weights=multiplier.to_frame(name="regime_multiplier"),
            scores=vol_data,
            metadata={
                "regime_df": regime_df,
                "high_vol_days": int(n_high_vol),
                "low_breadth_days": int(n_low_breadth),
            },
        )


__all__ = ["RegimeFilterSignal"]
