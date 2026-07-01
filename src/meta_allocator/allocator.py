"""
Meta-Allocation Layer.

Dynamically allocates capital across different strategies (sleeves)
based on their rolling Sharpe ratio, drawdown state, and pairwise correlation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import MetaAllocatorConfig
from src.utils.logger import logger


class MetaAllocator:
    """Rule-based dynamic strategy allocation.

    Distributes portfolio capital based on Sharpe ratio, drawdown gating, and correlation.
    """

    def __init__(self, cfg: MetaAllocatorConfig) -> None:
        self._cfg = cfg

    def compute(
        self,
        strategy_returns: pd.DataFrame,
        rebalance_dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Compute dynamic meta-allocation weights over time.

        Args:
            strategy_returns: DataFrame (Date × strategy_name) of daily returns.
            rebalance_dates: Dates on which to recompute weights.

        Returns:
            DataFrame (Date × strategy_name) of normalised allocation weights.
        """
        cfg = self._cfg
        strategies = strategy_returns.columns.tolist()
        n = len(strategies)

        logger.info(
            f"MetaAllocator: computing for {n} strategies over "
            f"{len(strategy_returns)} days"
        )

        all_weights = pd.DataFrame(
            np.nan, index=strategy_returns.index, columns=strategies
        )

        # Compute rolling performance and risk metrics
        rolling_sharpe = self._compute_rolling_sharpe(strategy_returns, cfg)
        running_drawdown = self._compute_drawdown(strategy_returns)
        rolling_corr = strategy_returns.rolling(
            window=cfg.correlation_window, min_periods=cfg.correlation_window // 2
        ).corr()

        prev_weights: dict[str, float] = {s: 1.0 / n for s in strategies}

        for date in rebalance_dates:
            if date not in strategy_returns.index:
                continue

            # 1. Base Score: max(0, rolling Sharpe ratio)
            sharpe_row = rolling_sharpe.loc[date] if date in rolling_sharpe.index else pd.Series()
            raw_scores: dict[str, float] = {}
            for s in strategies:
                sr = sharpe_row.get(s, np.nan)
                raw_scores[s] = max(0.0, sr) if not np.isnan(sr) else 0.0

            # 2. Drawdown Gate: cut allocation if strategy drawdown exceeds threshold
            dd_row = running_drawdown.loc[date] if date in running_drawdown.index else pd.Series()
            drawdown_factors: dict[str, float] = {}
            for s in strategies:
                dd = dd_row.get(s, 0.0)
                if not np.isnan(dd) and abs(dd) > cfg.max_drawdown_gate:
                    drawdown_factors[s] = cfg.drawdown_reduction_factor
                    logger.debug(
                        f"[Meta] {date.date()} | {s}: Drawdown gate triggered "
                        f"(DD={dd:.1%}) — weight × {cfg.drawdown_reduction_factor}"
                    )
                else:
                    drawdown_factors[s] = 1.0

            # 3. Correlation Penalty: penalize strategies highly correlated to others
            corr_factors: dict[str, float] = {}
            try:
                corr_matrix = rolling_corr.loc[date] if date in rolling_corr.index else pd.DataFrame()
                for s in strategies:
                    if corr_matrix.empty or s not in corr_matrix.index:
                        corr_factors[s] = 1.0
                        continue
                    other_strats = [x for x in strategies if x != s]
                    if not other_strats:
                        corr_factors[s] = 1.0
                        continue
                    max_corr = corr_matrix.loc[s, other_strats].abs().max()
                    if max_corr > cfg.high_correlation_threshold:
                        penalty = 1.0 - (max_corr - cfg.high_correlation_threshold)
                        corr_factors[s] = max(0.1, penalty)
                    else:
                        corr_factors[s] = 1.0
            except Exception:
                corr_factors = {s: 1.0 for s in strategies}

            # 4. Compute adjusted scores
            adjusted: dict[str, float] = {}
            for s in strategies:
                adjusted[s] = (
                    raw_scores[s]
                    * drawdown_factors[s]
                    * corr_factors[s]
                )

            # 5. Normalise scores to sum to 1.0 (fallback to equal weight if all scores <= 0)
            total = sum(adjusted.values())
            if total <= 0:
                logger.debug(
                    f"[Meta] {date.date()}: All strategy scores ≤ 0, "
                    f"falling back to equal weight"
                )
                weights = {s: 1.0 / n for s in strategies}
            else:
                weights = {s: adjusted[s] / total for s in strategies}

            # 6. Apply minimum weight floor and re-normalise
            if cfg.normalise_weights:
                final_weights: dict[str, float] = {}
                for s in strategies:
                    if raw_scores[s] > 0:
                        final_weights[s] = max(weights[s], cfg.min_weight)
                    else:
                        final_weights[s] = 0.0

                total2 = sum(final_weights.values())
                if total2 > 0:
                    final_weights = {s: w / total2 for s, w in final_weights.items()}
                else:
                    final_weights = {s: 1.0 / n for s in strategies}
            else:
                final_weights = weights

            prev_weights = final_weights
            all_weights.loc[date] = pd.Series(final_weights)

            logger.debug(
                f"[Meta] {date.date()} | Weights: "
                + " | ".join(f"{s}={w:.2%}" for s, w in final_weights.items())
            )

        # Forward-fill between rebalance dates
        all_weights = all_weights.ffill()
        # Before first rebalance: equal weight
        all_weights = all_weights.fillna(1.0 / n)

        logger.info(
            f"MetaAllocator complete | "
            f"Avg weights: {all_weights.mean().to_dict()}"
        )
        return all_weights

    def _compute_rolling_sharpe(
        self,
        returns: pd.DataFrame,
        cfg: MetaAllocatorConfig,
    ) -> pd.DataFrame:
        """Compute rolling annualised Sharpe ratio for each strategy."""
        excess = returns - cfg.risk_free_rate / 252  # Daily excess return
        rolling_mean = excess.rolling(
            window=cfg.rolling_sharpe_window,
            min_periods=cfg.rolling_sharpe_window // 2,
        ).mean()
        rolling_std = excess.rolling(
            window=cfg.rolling_sharpe_window,
            min_periods=cfg.rolling_sharpe_window // 2,
        ).std()
        sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(252)
        return sharpe

    def _compute_drawdown(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute running drawdown from peak for each strategy."""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max.replace(0, np.nan)
        return drawdown

    @staticmethod
    def compute_strategy_returns(
        signal_weights: pd.DataFrame,
        universe_returns: pd.DataFrame,
    ) -> pd.Series:
        """Compute daily returns for a single strategy sleeve.

        Args:
            signal_weights: DataFrame (Date × Symbol) of signal weights.
            universe_returns: DataFrame (Date × Symbol) of daily returns.
                              Note: returns are for t+1 (one-day forward shift
                              to avoid look-ahead bias).

        Returns:
            Series of daily strategy returns.
        """
        # Align dimensions
        weights = signal_weights.reindex(
            index=universe_returns.index, columns=universe_returns.columns
        ).ffill().fillna(0.0)

        # Strategy return = sum(weight_i × return_i)
        # Use SHIFTED returns: weight at t applied to return from t to t+1
        strategy_returns = (weights.shift(1) * universe_returns).sum(axis=1)
        return strategy_returns


__all__ = ["MetaAllocator"]
