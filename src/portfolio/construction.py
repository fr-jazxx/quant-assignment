"""
Portfolio construction.

Applies risk limits and capital constraints to convert raw signal weights
into constrained, normalised portfolio weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import BacktestConfig
from src.utils.logger import logger


class PortfolioConstructor:
    """Applies portfolio constraints to raw signal weights."""

    def __init__(self, cfg: BacktestConfig) -> None:
        self._cfg = cfg.portfolio

    def apply_constraints(
        self,
        raw_weights: dict[str, float],
        prices: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Apply all portfolio constraints to raw signal weights.

        Args:
            raw_weights: Dict of symbol → unconstrained weight.
            prices: Current prices (used to filter unavailable stocks).

        Returns:
            Dict of symbol → constrained, normalised weight.
        """
        cfg = self._cfg
        if not raw_weights:
            return {}

        weights = dict(raw_weights)

        # 1. Filter out symbols with missing prices
        if prices:
            weights = {
                s: w
                for s, w in weights.items()
                if prices.get(s, 0.0) > 0 and not np.isnan(prices.get(s, float("nan")))
            }

        if not weights:
            return {}

        # 2. Filter out positions below minimum weight
        weights = {
            s: w for s, w in weights.items()
            if w >= cfg.min_position_weight
        }

        # 3. Limit the maximum number of positions
        if len(weights) > cfg.max_positions:
            weights = dict(
                sorted(weights.items(), key=lambda x: x[1], reverse=True)[: cfg.max_positions]
            )

        if not weights:
            return {}

        # 4. Scale weights and apply maximum single-stock cap via water-filling
        investable = 1.0 - cfg.cash_buffer
        n = len(weights)
        effective_cap = cfg.max_stock_weight

        # Check if max stock weight cap is mathematically satisfiable
        if n * effective_cap < investable:
            effective_cap = investable / n  # Fall back to equal weighting
            logger.debug(
                f"Max weight cap relaxed to {effective_cap:.3f} — "
                f"too few stocks ({n}) to satisfy {cfg.max_stock_weight:.1%} cap"
            )

        # Normalize and iteratively redistribute excess weights
        normalised = {s: (w / sum(weights.values())) * investable for s, w in weights.items()}
        for _ in range(20):
            over_cap = {s: w for s, w in normalised.items() if w > effective_cap + 1e-9}
            if not over_cap:
                break  # Converged
            
            excess = sum(w - effective_cap for w in over_cap.values())
            uncapped = {s: w for s, w in normalised.items() if w <= effective_cap + 1e-9}
            
            # Cap excess positions and distribute overflow to uncapped ones
            for s in over_cap:
                normalised[s] = effective_cap
            if uncapped:
                extra_per = excess / len(uncapped)
                for s in uncapped:
                    normalised[s] = min(normalised[s] + extra_per, effective_cap)

        # Final normalisation to correct floating point drift
        total = sum(normalised.values())
        if total <= 0:
            return {}
        normalised = {s: (w / total) * investable for s, w in normalised.items()}

        # 5. Sanity checks
        total = sum(normalised.values())
        assert abs(total - investable) < 1e-4, (
            f"Weight normalisation error: sum={total:.6f}, expected={investable:.6f}"
        )

        logger.debug(
            f"Portfolio constraints applied | "
            f"Input: {len(raw_weights)} stocks | "
            f"Output: {len(normalised)} stocks | "
            f"Total weight: {total:.3f}"
        )
        return normalised


    def compute_turnover(
        self,
        old_weights: dict[str, float],
        new_weights: dict[str, float],
    ) -> float:
        """Compute one-way portfolio turnover between two rebalances.

        Turnover = 0.5 × sum(|new_weight - old_weight|) for all stocks.
        A value of 1.0 means the entire portfolio was replaced.

        Args:
            old_weights: Portfolio weights before rebalance.
            new_weights: Portfolio weights after rebalance.

        Returns:
            One-way turnover as a fraction [0, 1].
        """
        all_symbols = set(old_weights) | set(new_weights)
        total_change = sum(
            abs(new_weights.get(s, 0.0) - old_weights.get(s, 0.0))
            for s in all_symbols
        )
        return total_change / 2.0


__all__ = ["PortfolioConstructor"]
