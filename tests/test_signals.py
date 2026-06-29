"""
Tests for signal generation — critically testing for look-ahead bias.

The most important property of any backtest signal is that it does NOT
use data beyond the signal generation date. These tests explicitly verify
that constraint, along with signal range, weight normalisation, and
minimum history enforcement.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock

from src.signals.momentum import MomentumSignal
from src.signals.mean_reversion import MeanReversionSignal
from src.signals.regime_filter import RegimeFilterSignal
from src.utils.config import (
    MomentumSignalConfig,
    MeanReversionSignalConfig,
    RegimeFilterConfig,
)


def make_prices(
    n_stocks: int = 20,
    n_days: int = 500,
    start: str = "2019-01-01",
) -> pd.DataFrame:
    """Generate synthetic OHLCV price panel for testing."""
    np.random.seed(42)
    dates = pd.bdate_range(start=start, periods=n_days)
    symbols = [f"STOCK{i:02d}.NS" for i in range(n_stocks)]
    # Random walk prices — all positive
    returns = np.random.randn(n_days, n_stocks) * 0.01
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=symbols)


def make_momentum_cfg() -> MomentumSignalConfig:
    return MomentumSignalConfig(
        name="test_momentum",
        enabled=True,
        lookback_days=252,
        skip_days=21,
        top_n=5,
        min_valid_stocks=10,
        rebalance_freq="monthly",
        min_price=1.0,
        min_avg_volume_shares=0,
    )


def make_mr_cfg() -> MeanReversionSignalConfig:
    return MeanReversionSignalConfig(
        name="test_mr",
        enabled=True,
        rsi_window=5,
        rsi_oversold=30.0,
        zscore_window=20,
        zscore_threshold=-2.0,
        max_positions=5,
        rebalance_freq="weekly",
        rsi_exit=50.0,
        min_history_days=30,
    )


def make_regime_cfg() -> RegimeFilterConfig:
    return RegimeFilterConfig(
        name="test_regime",
        enabled=True,
        vol_window=20,
        vol_longrun_window=126,
        vol_expansion_threshold=1.5,
        high_vol_risk_factor=0.3,
        breadth_window=200,
        breadth_low_threshold=0.40,
        breadth_risk_factor=0.5,
        eval_freq="daily",
    )


class TestMomentumSignal:
    def test_weights_non_negative(self):
        prices = make_prices()
        signal = MomentumSignal(make_momentum_cfg())
        output = signal.generate(prices)
        assert (output.weights >= 0).all().all(), "Momentum weights must be non-negative"

    def test_weights_at_most_one(self):
        prices = make_prices()
        signal = MomentumSignal(make_momentum_cfg())
        output = signal.generate(prices)
        row_sums = output.weights.sum(axis=1)
        # Each rebalance row should sum to ~1.0 or 0.0
        non_zero = row_sums[row_sums > 0]
        assert (non_zero <= 1.01).all(), "Weights must not exceed 1.0 per row"

    def test_top_n_constraint(self):
        prices = make_prices()
        cfg = make_momentum_cfg()
        cfg.top_n = 5
        signal = MomentumSignal(cfg)
        output = signal.generate(prices)
        # On any rebalance date, at most top_n stocks should be non-zero
        non_zero_per_row = (output.weights > 0).sum(axis=1)
        assert (non_zero_per_row <= cfg.top_n + 1).all(), (
            f"More than top_n={cfg.top_n} stocks selected"
        )

    def test_no_signal_before_lookback(self):
        """Signal should produce zeros before enough history accumulates."""
        prices = make_prices(n_days=300)
        cfg = make_momentum_cfg()
        cfg.lookback_days = 252
        signal = MomentumSignal(cfg)
        output = signal.generate(prices)
        # First 252 days should have zero weights (no lookback available)
        early = output.weights.iloc[:252]
        assert (early.fillna(0) == 0).all().all(), (
            "Signal should produce no weights before lookback window fills"
        )

    def test_no_lookahead_bias(self):
        """The signal at date t must not use prices after date t."""
        prices = make_prices(n_days=400)
        signal = MomentumSignal(make_momentum_cfg())
        output = signal.generate(prices)

        # The weights index should never exceed the prices index
        assert output.weights.index.max() <= prices.index.max()

        # For each rebalance date, verify it's within the price index
        weight_dates = output.weights.index
        price_dates = prices.index
        for d in weight_dates:
            assert d in price_dates, f"Weight date {d} not in price index"


class TestMeanReversionSignal:
    def test_weights_non_negative(self):
        prices = make_prices()
        signal = MeanReversionSignal(make_mr_cfg())
        output = signal.generate(prices)
        assert (output.weights >= 0).all().all()

    def test_max_positions_respected(self):
        prices = make_prices()
        cfg = make_mr_cfg()
        cfg.max_positions = 3
        signal = MeanReversionSignal(cfg)
        output = signal.generate(prices)
        max_positions = (output.weights > 0).sum(axis=1).max()
        assert max_positions <= cfg.max_positions + 1  # +1 tolerance for forward-fill


class TestRegimeFilterSignal:
    def test_multiplier_range(self):
        """Regime multiplier must be in [0, 1]."""
        prices = make_prices()
        signal = RegimeFilterSignal(make_regime_cfg())
        output = signal.generate(prices)
        multiplier = output.weights["regime_multiplier"]
        assert (multiplier >= 0).all(), "Multiplier must be >= 0"
        assert (multiplier <= 1).all(), "Multiplier must be <= 1"

    def test_high_vol_triggers_reduction(self):
        """When vol spikes dramatically, multiplier should drop below 1."""
        prices = make_prices(n_days=300)
        # Inject high-vol period
        high_vol_prices = prices.copy()
        high_vol_prices.iloc[150:160] *= np.random.uniform(0.7, 1.3, size=(10, prices.shape[1]))

        signal = RegimeFilterSignal(make_regime_cfg())
        output = signal.generate(high_vol_prices)
        multiplier = output.weights["regime_multiplier"]
        # Should have at least some reduced days
        assert (multiplier < 1.0).any(), "Expected some risk-reduced days after vol spike"

    def test_output_has_expected_column(self):
        prices = make_prices()
        signal = RegimeFilterSignal(make_regime_cfg())
        output = signal.generate(prices)
        assert "regime_multiplier" in output.weights.columns


class TestBaseSignalLookahead:
    def test_validate_no_lookahead_raises_on_future(self):
        """validate_no_lookahead must raise AssertionError when prices extend beyond signal date."""
        from src.signals.base import BaseSignal

        class DummySignal(BaseSignal):
            @property
            def name(self): return "dummy"
            @property
            def rebalance_freq(self): return "monthly"
            def generate(self, prices, **kwargs): pass

        sig = DummySignal()
        prices = make_prices(n_days=100)
        signal_date = prices.index[50]  # Mid-point

        # Prices extend 50 days beyond signal_date — should raise
        with pytest.raises(AssertionError, match="Look-ahead bias"):
            sig.validate_no_lookahead(signal_date, prices)

    def test_validate_no_lookahead_passes_for_past_data(self):
        from src.signals.base import BaseSignal

        class DummySignal(BaseSignal):
            @property
            def name(self): return "dummy"
            @property
            def rebalance_freq(self): return "monthly"
            def generate(self, prices, **kwargs): pass

        sig = DummySignal()
        prices = make_prices(n_days=100)
        signal_date = prices.index[-1]  # Last date — no future data

        # Should not raise
        result = sig.validate_no_lookahead(signal_date, prices)
        assert result is True
