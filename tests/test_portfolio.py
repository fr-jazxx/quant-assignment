"""
Tests for portfolio construction constraints.

Ensures max weight, min weight, position count, and normalisation
all behave correctly.
"""

from __future__ import annotations

import pytest
from src.portfolio.construction import PortfolioConstructor
from src.utils.config import (
    BacktestConfig, CostsConfig, SlippageConfig,
    PortfolioConstraintsConfig, RiskConfig, DataPolicyConfig,
)


def make_constructor(
    max_stock_weight: float = 0.10,
    max_positions: int = 10,
    min_position_weight: float = 0.005,
    cash_buffer: float = 0.02,
) -> PortfolioConstructor:
    cfg = BacktestConfig(
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
            max_stock_weight=max_stock_weight,
            max_positions=max_positions,
            min_position_weight=min_position_weight,
            cash_buffer=cash_buffer,
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
    return PortfolioConstructor(cfg)


class TestConstraints:
    def test_max_weight_clamps_high_weights(self):
        # Use 10 stocks — 10 × 10% cap = 100% ≥ 98% investable (satisfiable)
        constructor = make_constructor(max_stock_weight=0.10)
        raw = {f"STOCK{i}": 0.8 if i == 0 else 0.1 for i in range(10)}
        result = constructor.apply_constraints(raw)
        for w in result.values():
            assert w <= 0.10 + 1e-6

    def test_max_positions_limits_count(self):
        constructor = make_constructor(max_positions=3)
        raw = {f"STOCK{i}": 0.1 for i in range(10)}
        result = constructor.apply_constraints(raw)
        assert len(result) <= 3

    def test_min_position_filter(self):
        constructor = make_constructor(min_position_weight=0.02)
        raw = {"A": 0.001, "B": 0.05, "C": 0.05}  # A is below min
        result = constructor.apply_constraints(raw)
        assert "A" not in result

    def test_cash_buffer_reserved(self):
        constructor = make_constructor(cash_buffer=0.05)
        # Use enough stocks so the cap is satisfiable
        raw = {f"S{i}": 0.5 for i in range(20)}
        result = constructor.apply_constraints(raw)
        total = sum(result.values())
        assert abs(total - 0.95) < 1e-4, f"Expected 0.95 investable, got {total:.6f}"

    def test_missing_price_symbols_excluded(self):
        constructor = make_constructor()
        raw = {"A": 0.3, "B": 0.3, "C": 0.4}
        prices = {"A": 100.0, "B": 0.0, "C": float("nan")}  # B and C invalid
        result = constructor.apply_constraints(raw, prices=prices)
        assert "B" not in result
        assert "C" not in result
        assert "A" in result

    def test_turnover_calculation(self):
        constructor = make_constructor()
        old = {"A": 0.5, "B": 0.5}
        new = {"A": 0.3, "C": 0.7}  # Sold B, bought C, reduced A
        turnover = constructor.compute_turnover(old, new)
        # |0.3-0.5| + |0-0.5| + |0.7-0| = 0.2 + 0.5 + 0.7 = 1.4 / 2 = 0.7
        assert abs(turnover - 0.7) < 1e-6

    def test_empty_input_returns_empty(self):
        constructor = make_constructor()
        assert constructor.apply_constraints({}) == {}

    def test_all_below_min_weight_returns_empty(self):
        constructor = make_constructor(min_position_weight=0.05)
        raw = {"A": 0.001, "B": 0.002}  # All below min
        result = constructor.apply_constraints(raw)
        assert result == {}
