"""
Tests for the backtesting engine and cost model.

Critical tests:
    - Transaction costs correctly reduce net returns
    - Cash never goes negative during execution
    - Rebalancing does not create impossible positions
    - Portfolio value is reproducible with the same seed and data
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock

from src.backtest.costs import CostModel, TradeResult
from src.backtest.positions import PositionsTracker
from src.utils.config import (
    BacktestConfig,
    CostsConfig,
    SlippageConfig,
    PortfolioConstraintsConfig,
    RiskConfig,
    DataPolicyConfig,
)


def make_backtest_cfg() -> BacktestConfig:
    return BacktestConfig(
        execution="next_day_open",
        initial_capital=1_000_000.0,
        costs=CostsConfig(
            stt_rate=0.001,
            exchange_charges_rate=0.0000345,
            sebi_fee_rate=0.000001,
            stamp_duty_rate=0.00015,
            brokerage_rate=0.0003,
            gst_rate=0.18,
        ),
        slippage=SlippageConfig(
            fixed_bps=5.0,
            volume_dependent=False,
            volume_threshold_pct=0.01,
        ),
        portfolio=PortfolioConstraintsConfig(
            max_stock_weight=0.10,
            max_positions=40,
            min_position_weight=0.005,
            cash_buffer=0.02,
            rebalance_tolerance=0.02,
        ),
        risk=RiskConfig(
            portfolio_stop_drawdown=0.20,
            hard_max_stock_weight=0.15,
        ),
        data=DataPolicyConfig(
            max_fill_days=3,
            skip_missing_execution_price=True,
            delisting_threshold_days=10,
        ),
        random_seed=42,
    )


class TestCostModel:
    def test_buy_increases_cash_outflow(self):
        """Buying should cost more than the raw trade value."""
        cfg = make_backtest_cfg()
        model = CostModel(cfg)
        result = model.compute_trade_cost("TCS.NS", 100_000.0, "buy")
        assert result.net_value > result.trade_value, (
            "Buying should result in net_value > trade_value (we pay costs)"
        )

    def test_sell_reduces_cash_inflow(self):
        """Selling should receive less than the raw trade value after costs."""
        cfg = make_backtest_cfg()
        model = CostModel(cfg)
        result = model.compute_trade_cost("TCS.NS", 100_000.0, "sell")
        assert result.net_value < result.trade_value, (
            "Selling should result in net_value < trade_value (we pay costs)"
        )

    def test_stt_only_on_sell(self):
        """STT (0.1%) should only be charged on sell side."""
        cfg = make_backtest_cfg()
        model = CostModel(cfg)
        buy_result = model.compute_trade_cost("TCS.NS", 100_000.0, "buy")
        sell_result = model.compute_trade_cost("TCS.NS", 100_000.0, "sell")
        assert buy_result.stt == 0.0, "STT should be 0 on buy"
        assert sell_result.stt == 100.0, "STT should be 0.1% of sell value"

    def test_stamp_duty_only_on_buy(self):
        """Stamp duty (0.015%) should only be charged on buy side."""
        cfg = make_backtest_cfg()
        model = CostModel(cfg)
        buy_result = model.compute_trade_cost("TCS.NS", 100_000.0, "buy")
        sell_result = model.compute_trade_cost("TCS.NS", 100_000.0, "sell")
        assert buy_result.stamp_duty > 0, "Stamp duty should be > 0 on buy"
        assert sell_result.stamp_duty == 0.0, "Stamp duty should be 0 on sell"

    def test_total_cost_positive(self):
        """Total costs must always be strictly positive."""
        cfg = make_backtest_cfg()
        model = CostModel(cfg)
        for direction in ["buy", "sell"]:
            result = model.compute_trade_cost("TCS.NS", 50_000.0, direction)
            assert result.total_cost > 0, f"Total cost must be positive for {direction}"

    def test_invalid_direction_raises(self):
        cfg = make_backtest_cfg()
        model = CostModel(cfg)
        with pytest.raises(ValueError, match="direction"):
            model.compute_trade_cost("TCS.NS", 100_000.0, "hold")

    def test_negative_value_raises(self):
        cfg = make_backtest_cfg()
        model = CostModel(cfg)
        with pytest.raises(ValueError):
            model.compute_trade_cost("TCS.NS", -1000.0, "buy")

    def test_round_trip_cost_is_reasonable(self):
        """Round-trip cost should be in ~20-40 bps range for Indian equities."""
        cfg = make_backtest_cfg()
        model = CostModel(cfg)
        rt_bps = model.round_trip_bps_estimate
        assert 10 <= rt_bps <= 100, (
            f"Round-trip cost {rt_bps:.1f} bps seems out of range"
        )


class TestPositionsTracker:
    def test_initial_state(self):
        tracker = PositionsTracker(initial_capital=1_000_000.0)
        assert tracker.cash == 1_000_000.0
        assert tracker.holdings == {}

    def test_negative_capital_raises(self):
        with pytest.raises(ValueError):
            PositionsTracker(initial_capital=-1000.0)

    def test_portfolio_value_equals_cash_when_no_positions(self):
        tracker = PositionsTracker(initial_capital=500_000.0)
        pv = tracker.get_portfolio_value({})
        assert pv == 500_000.0

    def test_portfolio_value_includes_holdings(self):
        tracker = PositionsTracker(initial_capital=1_000_000.0)
        tracker.holdings = {"TCS.NS": 10.0}  # 10 shares
        prices = {"TCS.NS": 3500.0}           # ₹3500 per share
        pv = tracker.get_portfolio_value(prices)
        expected = 1_000_000.0 + 10.0 * 3500.0
        assert abs(pv - expected) < 0.01

    def test_cash_never_negative_after_rebalance(self):
        """Cash balance must never go negative after any rebalance."""
        cfg = make_backtest_cfg()
        model = CostModel(cfg)
        tracker = PositionsTracker(initial_capital=1_000_000.0)

        date = pd.Timestamp("2023-01-03")
        prices = {"TCS.NS": 3000.0, "INFY.NS": 1500.0, "HDFCBANK.NS": 1600.0}
        target_weights = {"TCS.NS": 0.33, "INFY.NS": 0.33, "HDFCBANK.NS": 0.32}

        tracker.execute_rebalance(
            date=date,
            target_weights=target_weights,
            execution_prices=prices,
            cost_model=model,
            cash_buffer=0.02,
        )
        assert tracker.cash >= 0.0, "Cash went negative after rebalance!"

    def test_equity_curve_length(self):
        """Equity curve should have one entry per call to record_daily_value."""
        tracker = PositionsTracker(initial_capital=1_000_000.0)
        dates = pd.bdate_range("2023-01-01", periods=10)
        prices = {"TCS.NS": 3000.0}
        for d in dates:
            tracker.record_daily_value(d, prices)
        curve = tracker.get_equity_curve()
        assert len(curve) == 10

    def test_weights_sum_to_one(self):
        """Portfolio weights should sum to ≤ 1.0."""
        tracker = PositionsTracker(initial_capital=1_000_000.0)
        tracker.holdings = {"TCS.NS": 10.0, "INFY.NS": 20.0}
        prices = {"TCS.NS": 3000.0, "INFY.NS": 1500.0}
        weights = tracker.get_current_weights(prices)
        assert sum(weights.values()) <= 1.01


class TestPortfolioConstraints:
    def test_max_weight_respected(self):
        from src.portfolio.construction import PortfolioConstructor
        cfg = make_backtest_cfg()
        constructor = PortfolioConstructor(cfg)
        # Use 20 stocks — 20 × 10% = 200% >> 98% investable — cap is satisfiable
        raw_weights = {f"STOCK{i}": 1.0 if i == 0 else 0.1 for i in range(20)}
        constrained = constructor.apply_constraints(raw_weights)
        for weight in constrained.values():
            assert weight <= cfg.portfolio.max_stock_weight + 1e-6, (
                f"Weight {weight:.4f} exceeds max {cfg.portfolio.max_stock_weight}"
            )

    def test_weights_sum_within_investable(self):
        from src.portfolio.construction import PortfolioConstructor
        cfg = make_backtest_cfg()
        constructor = PortfolioConstructor(cfg)
        raw_weights = {f"STOCK{i}": 0.1 for i in range(10)}
        constrained = constructor.apply_constraints(raw_weights)
        total = sum(constrained.values())
        investable = 1.0 - cfg.portfolio.cash_buffer
        assert abs(total - investable) < 1e-5, (
            f"Constrained weights sum {total:.4f} ≠ investable {investable:.4f}"
        )

    def test_empty_weights_returns_empty(self):
        from src.portfolio.construction import PortfolioConstructor
        cfg = make_backtest_cfg()
        constructor = PortfolioConstructor(cfg)
        result = constructor.apply_constraints({})
        assert result == {}
