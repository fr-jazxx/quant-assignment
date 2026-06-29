"""
Signal 3: Volatility Regime and Market Breadth Filter

Economic Rationale:
    This signal is NOT a return-generating signal in isolation. It is a
    RISK OVERLAY that modulates the exposure of other signals based on
    the estimated market regime.

    Rationale for risk reduction in high-volatility regimes:
      - Kelly Criterion: Optimal position size is proportional to
        (expected return) / (variance). As variance rises, optimal size falls.
      - Volatility clustering (GARCH effects): High volatility today predicts
        high volatility tomorrow. Increased uncertainty warrants reduced risk.
      - Regime asymmetry: Most large drawdowns occur in high-vol environments.
        Pre-emptively reducing exposure is more efficient than stopping out
        after losses materialise.

    Market breadth (% stocks above 200-day SMA):
      - When breadth is low (< 40%), the market is deteriorating broadly.
        Individual stock momentum becomes noise-dominated.
      - This is inspired by Zweig Breadth Thrust and other technical breadth
        indicators used by institutional risk managers.

    India-specific context:
      - India VIX (NSE's fear index) would be the ideal regime input.
        However, its history is shorter and it is not available via yfinance.
        We use realised NIFTY volatility as a proxy.
      - NSE implements circuit breakers at 10%, 15%, 20% market-wide drops,
        creating risk of intraday halt. High-vol regimes correlate with
        increased circuit-breaker risk.

    References:
      - Ang, A. & Bekaert, G. (2002). International Asset Allocation with
        Regime Shifts. Review of Financial Studies, 15(4).
      - Asness, C. et al. (2012). Leverage Aversion and Risk Parity.
        Financial Analysts Journal, 68(1).
      - Lo, A. (2002). The Statistics of Sharpe Ratios. Financial Analysts
        Journal, 58(4).

Failure Conditions:
    - Fast-recovering markets: The filter may exit risk positions during
      the volatility spike of a V-shaped recovery, missing the rebound.
    - False positives: Temporary vol spikes (e.g., index reconstitution,
      budget announcements) may trigger regime switch unnecessarily.
    - Parameter sensitivity: The vol_expansion_threshold (1.5x) is a
      design choice — results are sensitive to this value.
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

    This signal outputs a SCALAR risk multiplier in [0, 1] per date
    rather than per-stock weights. It is applied multiplicatively to
    the combined portfolio weights by the portfolio construction layer.

    The multiplier is computed as:
        multiplier = min(vol_factor, breadth_factor)

    Where:
        vol_factor = 1.0 if current vol ≤ 1.5 × long-run vol, else 0.3
        breadth_factor = 1.0 if breadth ≥ 0.40, else 0.5
    """

    def __init__(self, cfg: RegimeFilterConfig) -> None:
        self._cfg = cfg

    @property
    def name(self) -> str:
        return "regime_filter_vol_breadth"

    @property
    def rebalance_freq(self) -> str:
        return self._cfg.eval_freq  # Daily

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
                          If None, uses the average of all symbols as proxy.

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

        # ── Index for vol computation ──
        if index_prices is None or index_prices.empty:
            logger.warning(
                f"[{self.name}] No index prices provided — using cross-sectional "
                f"mean of universe as proxy"
            )
            index_prices = prices.mean(axis=1)
        else:
            # Align to price index
            index_prices = index_prices.reindex(prices.index).ffill()

        # ── Compute realised volatility ratio ──
        vol_data = compute_index_realized_vol(
            index_prices,
            short_window=cfg.vol_window,
            long_window=cfg.vol_longrun_window,
        )

        # Vol factor: reduce risk when vol is elevated
        vol_ratio = vol_data["vol_ratio"]
        vol_factor = pd.Series(1.0, index=prices.index)
        high_vol_mask = vol_ratio > cfg.vol_expansion_threshold
        vol_factor[high_vol_mask] = cfg.high_vol_risk_factor

        # ── Compute market breadth ──
        breadth = compute_market_breadth(prices, sma_window=cfg.breadth_window)

        # Breadth factor: reduce risk when market breadth is poor
        breadth_factor = pd.Series(1.0, index=prices.index)
        low_breadth_mask = breadth < cfg.breadth_low_threshold
        breadth_factor[low_breadth_mask] = cfg.breadth_risk_factor

        # ── Combined multiplier = minimum of both factors ──
        multiplier = pd.concat(
            [vol_factor, breadth_factor], axis=1
        ).min(axis=1)

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

        # Regime summary stats
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
