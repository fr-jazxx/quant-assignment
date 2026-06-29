"""
Transaction cost model for Indian equity markets.

Cost Components (as of 2023-2024):
    1. Securities Transaction Tax (STT):
       - Equity delivery: 0.1% on SELL side only
       - Source: Finance Act, applicable to NSE/BSE listed equities
    2. Exchange Transaction Charges (NSE):
       - ~0.00345% of turnover (both sides)
       - Source: NSE Circular NSE/MEMB/45765
    3. SEBI Regulatory Fee:
       - 0.0001% of turnover
    4. Stamp Duty:
       - 0.015% on BUY side only
       - Source: Finance Act 2019 (effective July 2020)
    5. Brokerage:
       - Assumed discount broker: 0.03% or ₹20/order flat, whichever lower
       - Discount brokers (Zerodha, Upstox, etc.) are now market standard
    6. GST on brokerage + exchange charges: 18%
    7. Slippage:
       - Fixed 5 bps one-way for NIFTY 100 stocks (highly liquid)
       - Conservative estimate during normal conditions

Total approximate one-way cost: ~14.5–15 bps
Total round-trip cost: ~29–30 bps

Note on realism:
    The exact cost level matters less than ensuring costs are applied
    consistently and transparently. All cost components are parameterised
    in backtest.yaml so they can be adjusted for sensitivity analysis.
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
    """Computes transaction costs for Indian equity trades.

    All costs are computed per trade and logged for the trade blotter.
    Costs are always a drag on portfolio returns:
        - Buys: net_value = trade_value * (1 + total_cost_rate)
        - Sells: net_value = trade_value * (1 - total_cost_rate)

    This is intentionally conservative — real costs may be slightly lower
    for large institutions with negotiated rates, but we do not model that.
    """

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

        # STT: 0.1% on sell side only
        stt = trade_value_inr * c.stt_rate if direction == "sell" else 0.0

        # Exchange charges: both sides
        exchange_charges = trade_value_inr * c.exchange_charges_rate

        # SEBI fee: both sides
        sebi_fee = trade_value_inr * c.sebi_fee_rate

        # Stamp duty: 0.015% on buy side only
        stamp_duty = trade_value_inr * c.stamp_duty_rate if direction == "buy" else 0.0

        # Brokerage: min(0.03%, ₹20)
        brokerage = min(
            trade_value_inr * c.brokerage_rate,
            20.0,  # ₹20 cap per order
        )

        # GST on brokerage + exchange charges
        gst = (brokerage + exchange_charges) * c.gst_rate

        # Slippage: fixed bps applied to trade value
        slippage_cost = trade_value_inr * (self._slippage.fixed_bps / 10_000)

        total_cost = stt + exchange_charges + sebi_fee + stamp_duty + brokerage + gst + slippage_cost

        # Net value: what we effectively pay/receive
        if direction == "buy":
            net_value = trade_value_inr + total_cost   # Pay more when buying
        else:
            net_value = trade_value_inr - total_cost   # Receive less when selling

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
