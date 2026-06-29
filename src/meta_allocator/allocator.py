"""
Meta-Allocation Layer — dynamically allocates capital across strategy sleeves.

Philosophy:
    The meta-allocator is the "fund of funds" layer of this system. Instead of
    committing 100% to a single signal, it distributes capital based on the
    recent performance, drawdown state, and correlation of each strategy.

    This is inspired by risk-parity and adaptive allocation literature:
      - Bridgewater's All Weather: allocate to risk, not dollars
      - AQR's multi-strategy: scale up strategies with recent positive SR
      - RPM (Risk-Parity + Momentum): combine both concepts

    Key design decisions:
      1. Rule-based (not ML-based): Transparent, interpretable, avoids
         overfitting to the training period.
      2. Uses rolling metrics computed on OUT-OF-BAND returns: the
         meta-allocator sees signal PnL, not raw prices — reducing leakage.
      3. Drawdown gate: Hard risk reduction prevents catastrophic losses
         from a single strategy.

Meta-Allocation Formula:
    For each strategy i on date t:
        raw_score_i = max(0, rolling_sharpe_i(t))
        drawdown_factor_i = 0.5 if in_drawdown > threshold else 1.0
        correlation_penalty_i = 1.0 - max(0, correlation_with_others - threshold)
        adjusted_score_i = raw_score_i × drawdown_factor_i × correlation_penalty_i

    Normalised weight_i = adjusted_score_i / sum(adjusted_score_j)

    Final weight = max(min_weight, weight_i) if raw_score_i > 0 else 0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import MetaAllocatorConfig
from src.utils.logger import logger


class MetaAllocator:
    """Rule-based dynamic strategy allocation.

    Allocates capital among strategy sleeves based on:
        1. Rolling Sharpe ratio (performance quality)
        2. Drawdown gate (risk protection)
        3. Correlation check (diversification preservation)

    The meta-allocation is recomputed on each rebalance date.
    Between rebalance dates, weights are held constant (forward-filled).
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
            strategy_returns: DataFrame (Date × strategy_name) of daily returns
                              for each strategy sleeve. This is computed from
                              the signal weights applied to the market data.
            rebalance_dates: Dates on which to recompute weights.

        Returns:
            DataFrame (Date × strategy_name) of normalised allocation weights.
            Index covers all dates in strategy_returns.index.
        """
        cfg = self._cfg
        strategies = strategy_returns.columns.tolist()
        n = len(strategies)

        logger.info(
            f"MetaAllocator: computing for {n} strategies over "
            f"{len(strategy_returns)} days"
        )

        # Initialise weights — equal weight as default
        all_weights = pd.DataFrame(
            np.nan, index=strategy_returns.index, columns=strategies
        )

        # ── Compute rolling metrics ──
        rolling_sharpe = self._compute_rolling_sharpe(strategy_returns, cfg)
        running_drawdown = self._compute_drawdown(strategy_returns)
        rolling_corr = strategy_returns.rolling(
            window=cfg.correlation_window, min_periods=cfg.correlation_window // 2
        ).corr()

        prev_weights: dict[str, float] = {s: 1.0 / n for s in strategies}

        for date in rebalance_dates:
            if date not in strategy_returns.index:
                continue

            # ── Step 1: Raw score = max(0, rolling Sharpe) ──
            sharpe_row = rolling_sharpe.loc[date] if date in rolling_sharpe.index else pd.Series()
            raw_scores: dict[str, float] = {}
            for s in strategies:
                sr = sharpe_row.get(s, np.nan)
                raw_scores[s] = max(0.0, sr) if not np.isnan(sr) else 0.0

            # ── Step 2: Drawdown gate ──
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

            # ── Step 3: Correlation penalty ──
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
                        # Penalty proportional to excess correlation
                        penalty = 1.0 - (max_corr - cfg.high_correlation_threshold)
                        corr_factors[s] = max(0.1, penalty)
                    else:
                        corr_factors[s] = 1.0
            except Exception:
                corr_factors = {s: 1.0 for s in strategies}

            # ── Step 4: Compute adjusted scores ──
            adjusted: dict[str, float] = {}
            for s in strategies:
                adjusted[s] = (
                    raw_scores[s]
                    * drawdown_factors[s]
                    * corr_factors[s]
                )

            # ── Step 5: Normalise ──
            total = sum(adjusted.values())
            if total <= 0:
                # All strategies have non-positive Sharpe — go to equal weight
                # (could also go to 0 / cash, but equal weight is more conservative)
                logger.debug(
                    f"[Meta] {date.date()}: All strategy scores ≤ 0, "
                    f"falling back to equal weight"
                )
                weights = {s: 1.0 / n for s in strategies}
            else:
                weights = {s: adjusted[s] / total for s in strategies}

            # ── Step 6: Apply minimum weight floor ──
            if cfg.normalise_weights:
                final_weights: dict[str, float] = {}
                for s in strategies:
                    if raw_scores[s] > 0:
                        final_weights[s] = max(weights[s], cfg.min_weight)
                    else:
                        final_weights[s] = 0.0

                # Re-normalise after floor application
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
