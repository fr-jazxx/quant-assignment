"""
Transaction cost model for Indian equity markets.

Includes brokerage, GST, STT, Exchange transaction charges, SEBI fees, Stamp Duty, and Slippage.
All rates are parameterised in the configuration file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.config import BacktestConfig
from src.utils.logger import logger


@dataclass
class TradeResult:
    """Result of a single trade including all cost components."""

    symbol: str
    trade_value: float    # Absolute INR value traded (always positive)
    direction: str        # 'buy' or 'sell'
    gross_value: float
    stt: float
    exchange_charges: float
    sebi_fee: float
    stamp_duty: float
    brokerage: float
    gst: float
    slippage_cost: float
    total_cost: float
    net_value: float      # Effective value after all costs


class CostModel:
    """Computes transaction costs for Indian equity trades."""

    def __init__(self, cfg: BacktestConfig) -> None:
        self._costs = cfg.costs
        self._slippage = cfg.slippage

    def compute_trade_cost(
        self,
        symbol: str,
        trade_value_inr: float,
        direction: str,
    ) -> TradeResult:
        """Compute all transaction costs for a single trade.

        Args:
            symbol: Ticker symbol (for logging).
            trade_value_inr: Absolute trade value in INR (always positive).
            direction: 'buy' or 'sell'.

        Returns:
            TradeResult with all cost components broken down.
        """
        if trade_value_inr <= 0:
            raise ValueError(f"trade_value_inr must be positive, got {trade_value_inr}")
        if direction not in ("buy", "sell"):
            raise ValueError(f"direction must be 'buy' or 'sell', got '{direction}'")

        c = self._costs

        # Calculate individual transaction cost components
        stt = trade_value_inr * c.stt_rate if direction == "sell" else 0.0
        exchange_charges = trade_value_inr * c.exchange_charges_rate
        sebi_fee = trade_value_inr * c.sebi_fee_rate
        stamp_duty = trade_value_inr * c.stamp_duty_rate if direction == "buy" else 0.0
        brokerage = min(trade_value_inr * c.brokerage_rate, 20.0)  # Capped at ₹20 per order
        gst = (brokerage + exchange_charges) * c.gst_rate
        slippage_cost = trade_value_inr * (self._slippage.fixed_bps / 10_000)

        total_cost = stt + exchange_charges + sebi_fee + stamp_duty + brokerage + gst + slippage_cost

        # Adjust execution cash value by the total cost drag
        if direction == "buy":
            net_value = trade_value_inr + total_cost
        else:
            net_value = trade_value_inr - total_cost

        result = TradeResult(
            symbol=symbol,
            trade_value=trade_value_inr,
            direction=direction,
            gross_value=trade_value_inr,
            stt=stt,
            exchange_charges=exchange_charges,
            sebi_fee=sebi_fee,
            stamp_duty=stamp_duty,
            brokerage=brokerage,
            gst=gst,
            slippage_cost=slippage_cost,
            total_cost=total_cost,
            net_value=net_value,
        )

        cost_bps = (total_cost / trade_value_inr) * 10_000
        logger.debug(
            f"Cost [{direction.upper():4}] {symbol}: "
            f"₹{trade_value_inr:,.0f} | "
            f"Cost: ₹{total_cost:,.1f} ({cost_bps:.1f} bps)"
        )
        return result

    def compute_portfolio_costs(
        self,
        trades: dict[str, tuple[float, str]],
    ) -> dict[str, TradeResult]:
        """Compute costs for a set of trades on a rebalance date.

        Args:
            trades: Dict of symbol → (trade_value_inr, direction).

        Returns:
            Dict of symbol → TradeResult.
        """
        results: dict[str, TradeResult] = {}
        for symbol, (value, direction) in trades.items():
            if value > 0:
                results[symbol] = self.compute_trade_cost(symbol, value, direction)
        return results

    @property
    def round_trip_bps_estimate(self) -> float:
        """Approximate round-trip cost in basis points (for ₹1M trade)."""
        reference_value = 1_000_000.0
        buy_cost = self.compute_trade_cost("REF", reference_value, "buy")
        sell_cost = self.compute_trade_cost("REF", reference_value, "sell")
        rt_bps = (
            (buy_cost.total_cost + sell_cost.total_cost) / reference_value
        ) * 10_000
        return rt_bps


__all__ = ["CostModel", "TradeResult"]
