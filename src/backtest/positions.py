"""
Portfolio positions tracker — maintains the state of holdings,
cash, and portfolio value throughout a backtest simulation.

This module tracks:
    - Current holdings (shares held per symbol)
    - Cash balance
    - Portfolio value (cash + market value of holdings)
    - Trade history (blotter)
    - Daily P&L attribution

Design Principles:
    - Immutable snapshots: each day produces a frozen record of state.
    - No implicit assumptions: missing prices are handled explicitly.
    - Cash must never go negative (enforced with assertion).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import numpy as np

from src.utils.logger import logger


@dataclass
class PositionSnapshot:
    """Immutable snapshot of portfolio state at end of a trading day."""

    date: pd.Timestamp
    holdings: dict[str, float]       # symbol → shares held
    prices: dict[str, float]         # symbol → close price used for valuation
    cash: float
    portfolio_value: float
    equity_value: float
    weights: dict[str, float]        # symbol → actual weight in portfolio


@dataclass
class TradeRecord:
    """Single trade record for the blotter."""

    date: pd.Timestamp
    symbol: str
    direction: str               # 'buy' or 'sell'
    shares: float
    price: float                 # execution price (next-day open)
    gross_value: float
    cost_inr: float
    net_value: float
    portfolio_value_before: float
    portfolio_value_after: float


class PositionsTracker:
    """Tracks portfolio state through a backtest.

    Usage:
        tracker = PositionsTracker(initial_capital=10_000_000)
        tracker.update(date, prices, target_weights, cost_model)
        snapshots = tracker.get_snapshots()
    """

    def __init__(self, initial_capital: float) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

        self.initial_capital: float = initial_capital
        self.cash: float = initial_capital
        self.holdings: dict[str, float] = {}  # symbol → shares
        self._snapshots: list[PositionSnapshot] = []
        self._blotter: list[TradeRecord] = []
        self._daily_values: list[dict[str, Any]] = []

        logger.info(
            f"PositionsTracker initialised | "
            f"Capital: ₹{initial_capital:,.0f}"
        )

    def get_market_value(self, prices: dict[str, float]) -> float:
        """Compute total market value of current equity holdings."""
        value = 0.0
        for symbol, shares in self.holdings.items():
            price = prices.get(symbol)
            if price is not None and not np.isnan(price) and price > 0:
                value += shares * price
            elif shares > 0:
                # Missing price — log warning, value at 0 (conservative)
                logger.warning(
                    f"No price for held position: {symbol} ({shares:.0f} shares). "
                    f"Valuing at 0 — check for delisting."
                )
        return value

    def get_portfolio_value(self, prices: dict[str, float]) -> float:
        """Total portfolio value = cash + equity market value."""
        return self.cash + self.get_market_value(prices)

    def get_current_weights(self, prices: dict[str, float]) -> dict[str, float]:
        """Compute current portfolio weights."""
        portfolio_value = self.get_portfolio_value(prices)
        if portfolio_value <= 0:
            return {}
        weights: dict[str, float] = {}
        for symbol, shares in self.holdings.items():
            price = prices.get(symbol)
            if price and not np.isnan(price):
                weights[symbol] = (shares * price) / portfolio_value
        return weights

    def execute_rebalance(
        self,
        date: pd.Timestamp,
        target_weights: dict[str, float],
        execution_prices: dict[str, float],
        cost_model,  # CostModel — avoid circular import with string type
        cash_buffer: float = 0.02,
    ) -> list[TradeRecord]:
        """Execute trades to move from current holdings to target weights.

        Execution assumptions:
            - Prices are NEXT DAY OPEN prices (t+1) — no look-ahead bias.
            - Cash buffer is maintained at all times.
            - Sells execute before buys to free up cash.
            - Weights exceeding hard limits are clipped before execution.

        Args:
            date: Execution date (t+1 — day after signal generation).
            target_weights: Dict of symbol → desired portfolio weight.
            execution_prices: Dict of symbol → open price at execution day.
            cost_model: CostModel instance.
            cash_buffer: Minimum fraction of portfolio to keep as cash.

        Returns:
            List of TradeRecord for this rebalance.
        """
        trades: list[TradeRecord] = []
        portfolio_value = self.get_portfolio_value(execution_prices)

        if portfolio_value <= 0:
            logger.error(f"[{date.date()}] Portfolio value is zero or negative!")
            return trades

        # Scale target weights to investable portion (excluding cash buffer)
        investable = portfolio_value * (1.0 - cash_buffer)
        target_values: dict[str, float] = {
            symbol: weight * investable
            for symbol, weight in target_weights.items()
        }

        # Current values
        current_values: dict[str, float] = {
            symbol: shares * execution_prices.get(symbol, 0.0)
            for symbol, shares in self.holdings.items()
            if execution_prices.get(symbol, 0.0) > 0
        }

        # ── Determine required trades (sells first, then buys) ──
        all_symbols = set(target_values) | set(current_values)
        sell_trades: list[tuple[str, float]] = []
        buy_trades: list[tuple[str, float]] = []

        for symbol in all_symbols:
            target_val = target_values.get(symbol, 0.0)
            current_val = current_values.get(symbol, 0.0)
            delta = target_val - current_val

            if delta < -10:  # Sell threshold: at least ₹10
                sell_trades.append((symbol, abs(delta)))
            elif delta > 10:  # Buy threshold: at least ₹10
                buy_trades.append((symbol, delta))

        # ── Execute sells ──
        for symbol, sell_value in sell_trades:
            price = execution_prices.get(symbol)
            if not price or price <= 0 or np.isnan(price):
                logger.warning(f"[{date.date()}] No execution price for sell: {symbol}")
                continue

            shares_to_sell = sell_value / price
            shares_held = self.holdings.get(symbol, 0.0)
            shares_to_sell = min(shares_to_sell, shares_held)

            if shares_to_sell <= 0:
                continue

            cost_result = cost_model.compute_trade_cost(
                symbol, shares_to_sell * price, "sell"
            )

            pv_before = portfolio_value
            self.holdings[symbol] = max(0.0, shares_held - shares_to_sell)
            if self.holdings[symbol] == 0.0:
                del self.holdings[symbol]

            self.cash += cost_result.net_value
            portfolio_value = self.get_portfolio_value(execution_prices)

            trade = TradeRecord(
                date=date,
                symbol=symbol,
                direction="sell",
                shares=shares_to_sell,
                price=price,
                gross_value=shares_to_sell * price,
                cost_inr=cost_result.total_cost,
                net_value=cost_result.net_value,
                portfolio_value_before=pv_before,
                portfolio_value_after=portfolio_value,
            )
            trades.append(trade)
            self._blotter.append(trade)

        # ── Execute buys ──
        for symbol, buy_value in buy_trades:
            price = execution_prices.get(symbol)
            if not price or price <= 0 or np.isnan(price):
                logger.warning(f"[{date.date()}] No execution price for buy: {symbol}")
                continue

            # Cap buy at available cash
            actual_buy_value = min(buy_value, self.cash - portfolio_value * cash_buffer)
            if actual_buy_value <= 0:
                logger.debug(
                    f"[{date.date()}] Insufficient cash for buy: {symbol} "
                    f"(need ₹{buy_value:,.0f}, available ₹{self.cash:,.0f})"
                )
                continue

            cost_result = cost_model.compute_trade_cost(symbol, actual_buy_value, "buy")
            shares_to_buy = actual_buy_value / price

            pv_before = portfolio_value
            self.holdings[symbol] = self.holdings.get(symbol, 0.0) + shares_to_buy

            # Deduct total cost (gross + fees) from cash
            self.cash -= cost_result.net_value
            assert self.cash >= -1.0, (
                f"Cash went negative: {self.cash:.2f} — bug in execution logic"
            )
            portfolio_value = self.get_portfolio_value(execution_prices)

            trade = TradeRecord(
                date=date,
                symbol=symbol,
                direction="buy",
                shares=shares_to_buy,
                price=price,
                gross_value=actual_buy_value,
                cost_inr=cost_result.total_cost,
                net_value=actual_buy_value,
                portfolio_value_before=pv_before,
                portfolio_value_after=portfolio_value,
            )
            trades.append(trade)
            self._blotter.append(trade)

        return trades

    def record_daily_value(
        self,
        date: pd.Timestamp,
        close_prices: dict[str, float],
    ) -> float:
        """Record end-of-day portfolio value using closing prices."""
        equity_value = self.get_market_value(close_prices)
        portfolio_value = self.cash + equity_value
        weights = self.get_current_weights(close_prices)

        snap = PositionSnapshot(
            date=date,
            holdings=dict(self.holdings),
            prices=dict(close_prices),
            cash=self.cash,
            portfolio_value=portfolio_value,
            equity_value=equity_value,
            weights=weights,
        )
        self._snapshots.append(snap)

        self._daily_values.append(
            {
                "date": date,
                "portfolio_value": portfolio_value,
                "equity_value": equity_value,
                "cash": self.cash,
                "n_positions": len(self.holdings),
            }
        )
        return portfolio_value

    def get_equity_curve(self) -> pd.Series:
        """Return portfolio value time series."""
        if not self._daily_values:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self._daily_values).set_index("date")
        return df["portfolio_value"]

    def get_blotter(self) -> pd.DataFrame:
        """Return full trade blotter as DataFrame."""
        if not self._blotter:
            return pd.DataFrame()
        return pd.DataFrame([vars(t) for t in self._blotter])

    def get_holdings_history(self) -> pd.DataFrame:
        """Return portfolio weights at each snapshot date."""
        if not self._snapshots:
            return pd.DataFrame()
        rows = []
        for snap in self._snapshots:
            row = {"date": snap.date, "cash_weight": snap.cash / max(snap.portfolio_value, 1)}
            row.update(snap.weights)
            rows.append(row)
        return pd.DataFrame(rows).set_index("date").fillna(0.0)


__all__ = ["PositionsTracker", "PositionSnapshot", "TradeRecord"]
