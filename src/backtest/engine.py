"""
Core backtesting engine.

This engine simulates a realistic daily portfolio simulation with:
    - NO look-ahead bias: signal computed at close of t, executed at open of t+1
    - Transaction costs and slippage applied on every trade
    - Cash tracking with no implicit leverage
    - Handling of missing prices (delisting, data gaps)
    - Multiple strategy sleeves with meta-allocation weights

Engine Flow (per trading day):
    1. Check if today is a rebalance date for any signal.
    2. If rebalance → generate target weights from signals.
    3. Apply meta-allocator weights to combine signal outputs.
    4. Apply regime multiplier to scale overall risk.
    5. Apply portfolio constraints (max weight, max positions, cash buffer).
    6. At open of NEXT day (t+1): execute trades at open prices.
    7. At close of day: value portfolio at close prices.
    8. Record daily P&L, holdings, and trade blotter.

Execution Assumption:
    Trades execute at the OPEN price of t+1. This is conservative because:
      - It introduces a realistic 1-day delay between signal and execution.
      - It avoids using same-day close prices for both signal and execution.
      - In practice, execution quality may be better or worse, depending on
        order size, time of day, and market conditions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.backtest.costs import CostModel
from src.backtest.positions import PositionsTracker
from src.signals.base import SignalOutput
from src.portfolio.construction import PortfolioConstructor
from src.utils.config import AppConfig
from src.utils.logger import logger


class BacktestResult:
    """Container for all backtest outputs."""

    def __init__(
        self,
        equity_curve: pd.Series,
        blotter: pd.DataFrame,
        holdings_history: pd.DataFrame,
        signal_weights: dict[str, pd.DataFrame],
        meta_weights: pd.DataFrame,
        regime_multiplier: pd.Series,
        benchmark: pd.Series,
        config: AppConfig,
    ) -> None:
        self.equity_curve = equity_curve
        self.blotter = blotter
        self.holdings_history = holdings_history
        self.signal_weights = signal_weights
        self.meta_weights = meta_weights
        self.regime_multiplier = regime_multiplier
        self.benchmark = benchmark
        self.config = config

    def daily_returns(self) -> pd.Series:
        return self.equity_curve.pct_change().dropna()

    def benchmark_returns(self) -> pd.Series:
        return self.benchmark.pct_change().dropna()


class BacktestEngine:
    """Event-driven backtesting engine for multi-signal portfolio simulation.

    Orchestrates signal generation, meta-allocation, portfolio construction,
    and trade execution across the full backtest date range.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cost_model = CostModel(config.backtest)
        self.constructor = PortfolioConstructor(config.backtest)

        # Log estimated round-trip cost
        rt_bps = self.cost_model.round_trip_bps_estimate
        logger.info(
            f"BacktestEngine initialised | "
            f"Estimated round-trip cost: {rt_bps:.1f} bps | "
            f"Execution: {config.backtest.execution}"
        )

    def run(
        self,
        prices: pd.DataFrame,
        signal_outputs: dict[str, SignalOutput],
        meta_weights: pd.DataFrame,
        regime_multiplier: pd.Series,
        benchmark_prices: pd.Series,
        open_prices: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """Run the full backtest simulation.

        Args:
            prices: Adjusted close price panel (Date × Symbol).
            signal_outputs: Dict of signal_name → SignalOutput (weights).
            meta_weights: DataFrame of (Date × signal_name) meta-allocation weights.
            regime_multiplier: Series (Date) of risk multiplier in [0, 1].
            benchmark_prices: Series of benchmark adjusted close prices.
            open_prices: Optional open price panel. If None, uses close prices
                         shifted by 1 day as execution price proxy.

        Returns:
            BacktestResult with all simulation outputs.
        """
        cfg = self.config.backtest
        initial_capital = cfg.initial_capital

        tracker = PositionsTracker(initial_capital=initial_capital)

        # ── Execution prices: open of next day ──
        if open_prices is not None:
            exec_prices = open_prices
        else:
            # Fallback: use close prices as execution proxy
            # In reality, opening prices differ — this is a simplification
            # that we document as a limitation.
            logger.warning(
                "No open prices provided — using close prices as execution proxy. "
                "This slightly understates transaction costs."
            )
            exec_prices = prices

        # ── Align all date indices ──
        trading_dates = prices.index
        logger.info(
            f"Backtest period: {trading_dates[0].date()} → {trading_dates[-1].date()} "
            f"({len(trading_dates)} trading days)"
        )

        # ── Combine signal weights using meta-allocation ──
        combined_weights = self._combine_signals(
            signal_outputs=signal_outputs,
            meta_weights=meta_weights,
            regime_multiplier=regime_multiplier,
            price_index=trading_dates,
        )

        # ── Daily simulation loop ──
        pending_rebalance: dict[str, float] | None = None  # target weights for t+1
        pending_rebalance_date: pd.Timestamp | None = None

        for i, date in enumerate(tqdm(trading_dates, desc="Backtesting")):
            # ── Execute pending rebalance (from yesterday's signal) ──
            if pending_rebalance is not None and pending_rebalance_date is not None:
                exec_price_row = exec_prices.loc[date] if date in exec_prices.index else None
                if exec_price_row is not None:
                    exec_price_dict = exec_price_row.dropna().to_dict()
                    if exec_price_dict:
                        tracker.execute_rebalance(
                            date=date,
                            target_weights=pending_rebalance,
                            execution_prices=exec_price_dict,
                            cost_model=self.cost_model,
                            cash_buffer=cfg.portfolio.cash_buffer,
                        )
                pending_rebalance = None
                pending_rebalance_date = None

            # ── Check if today generates a new signal ──
            if date in combined_weights.index:
                target_row = combined_weights.loc[date]
                target_dict = target_row[target_row > 0].to_dict()

                if target_dict:
                    # Apply portfolio constraints
                    constrained = self.constructor.apply_constraints(
                        target_dict,
                        prices.loc[date].to_dict() if date in prices.index else {},
                    )
                    pending_rebalance = constrained
                    pending_rebalance_date = date

            # ── Record end-of-day portfolio value ──
            close_price_row = prices.loc[date] if date in prices.index else pd.Series()
            close_price_dict = close_price_row.dropna().to_dict()
            tracker.record_daily_value(date, close_price_dict)

        # ── Align benchmark ──
        bench_aligned = benchmark_prices.reindex(trading_dates).ffill()

        # ── Compile results ──
        result = BacktestResult(
            equity_curve=tracker.get_equity_curve(),
            blotter=tracker.get_blotter(),
            holdings_history=tracker.get_holdings_history(),
            signal_weights={k: v.weights for k, v in signal_outputs.items()},
            meta_weights=meta_weights,
            regime_multiplier=regime_multiplier,
            benchmark=bench_aligned,
            config=self.config,
        )

        logger.info(
            f"Backtest complete | "
            f"Total trades: {len(result.blotter)} | "
            f"Final value: ₹{result.equity_curve.iloc[-1]:,.0f}"
        )
        return result

    def _combine_signals(
        self,
        signal_outputs: dict[str, SignalOutput],
        meta_weights: pd.DataFrame,
        regime_multiplier: pd.Series,
        price_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Combine per-signal weights into a single portfolio weight vector.

        Combination logic:
            portfolio_weight[symbol] = sum over signals of:
                meta_weight[signal] × signal_weight[symbol] × regime_multiplier

        This is a linear combination — no interaction terms.

        Args:
            signal_outputs: Dict of signal_name → SignalOutput.
            meta_weights: DataFrame (Date × signal_name), weights summing to 1.
            regime_multiplier: Series (Date) with values in [0, 1].
            price_index: Full trading date index to align everything to.

        Returns:
            Wide DataFrame (Date × Symbol) of combined portfolio weights.
        """
        all_symbols: list[str] = []
        for sig_out in signal_outputs.values():
            all_symbols.extend(sig_out.weights.columns.tolist())
        all_symbols = list(set(all_symbols))

        combined = pd.DataFrame(0.0, index=price_index, columns=all_symbols)

        for signal_name, sig_out in signal_outputs.items():
            if signal_name not in meta_weights.columns:
                logger.warning(
                    f"Signal '{signal_name}' not in meta_weights — skipping"
                )
                continue

            sig_weights = sig_out.weights.reindex(price_index).ffill().fillna(0.0)
            meta_w = meta_weights[signal_name].reindex(price_index).ffill().fillna(0.0)

            # Broadcast meta weight across symbols
            scaled = sig_weights.multiply(meta_w, axis=0)

            # Add to combined (align columns)
            combined = combined.add(
                scaled.reindex(columns=all_symbols, fill_value=0.0),
                fill_value=0.0,
            )

        # Apply regime multiplier
        regime = regime_multiplier.reindex(price_index).ffill().fillna(1.0)
        combined = combined.multiply(regime, axis=0)

        logger.debug(
            f"Combined weights: shape={combined.shape}, "
            f"mean nonzero per day={combined.gt(0).sum(axis=1).mean():.1f}"
        )
        return combined


__all__ = ["BacktestEngine", "BacktestResult"]
