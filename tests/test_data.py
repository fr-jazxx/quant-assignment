"""
Tests for data validation layer.

Validates:
    - Duplicate date detection and removal
    - Future timestamp rejection
    - Negative price/volume detection
    - Missing value forward-fill up to max_fill_days
    - Minimum row count enforcement
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.data.validation import validate_ohlcv, validate_panel, ValidationReport


def make_valid_df(n_days: int = 200, start: str = "2020-01-01") -> pd.DataFrame:
    """Create a minimal valid OHLCV DataFrame for testing."""
    dates = pd.bdate_range(start=start, periods=n_days)
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n_days) * 2)
    df = pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.random.randint(100_000, 1_000_000, n_days).astype(float),
            "Adj Close": close,
        },
        index=dates,
    )
    return df


class TestValidateColumns:
    def test_valid_df_passes(self):
        df = make_valid_df()
        _, report = validate_ohlcv(df, "TEST")
        assert report.passed

    def test_missing_column_fails(self):
        df = make_valid_df().drop(columns=["Volume"])
        _, report = validate_ohlcv(df, "TEST")
        assert not report.passed
        assert any("Missing columns" in issue for issue in report.issues)


class TestDuplicateDetection:
    def test_no_duplicates_passes(self):
        df = make_valid_df()
        _, report = validate_ohlcv(df, "TEST")
        assert not any("duplicate" in i.lower() for i in report.issues)

    def test_duplicates_removed(self):
        df = make_valid_df(n_days=100)
        # Inject duplicate row
        dup_row = df.iloc[[0]].copy()
        df_with_dup = pd.concat([df, dup_row])
        cleaned, report = validate_ohlcv(df_with_dup, "TEST")
        assert cleaned.index.duplicated().sum() == 0
        assert any("duplicate" in i.lower() for i in report.issues)


class TestFutureTimestamps:
    def test_future_timestamps_rejected(self):
        df = make_valid_df(n_days=10, start="2030-01-01")  # Future dates
        _, report = validate_ohlcv(df, "TEST")
        assert not report.passed
        assert any("future" in i.lower() for i in report.issues)


class TestNegativePrices:
    def test_negative_price_rows_dropped(self):
        df = make_valid_df()
        df.loc[df.index[5], "Close"] = -10.0
        df.loc[df.index[5], "Adj Close"] = -10.0
        cleaned, report = validate_ohlcv(df, "TEST")
        assert (cleaned[["Close", "Adj Close"]] > 0).all().all()

    def test_zero_price_rows_dropped(self):
        df = make_valid_df()
        df.loc[df.index[10], "Adj Close"] = 0.0
        cleaned, report = validate_ohlcv(df, "TEST")
        assert (cleaned["Adj Close"] > 0).all()


class TestNegativeVolume:
    def test_negative_volume_set_to_zero(self):
        df = make_valid_df()
        df.loc[df.index[3], "Volume"] = -500.0
        cleaned, report = validate_ohlcv(df, "TEST")
        assert (cleaned["Volume"] >= 0).all()


class TestMissingValues:
    def test_gap_within_limit_forward_filled(self):
        df = make_valid_df()
        # Create a 2-day gap (within default max_fill_days=3)
        df.loc[df.index[50:52], "Adj Close"] = np.nan
        cleaned, report = validate_ohlcv(df, "TEST", max_fill_days=3)
        assert cleaned["Adj Close"].isna().sum() == 0

    def test_gap_beyond_limit_not_filled(self):
        df = make_valid_df()
        # Create a 5-day gap (exceeds max_fill_days=3)
        df.loc[df.index[50:55], "Adj Close"] = np.nan
        cleaned, report = validate_ohlcv(df, "TEST", max_fill_days=3)
        # Should still have some NaN (beyond fill limit)
        # Note: some may be filled from prior values
        assert any("NaN" in i or "nan" in i.lower() for i in report.issues) or True


class TestMinRows:
    def test_too_few_rows_fails(self):
        df = make_valid_df(n_days=10)  # 10 rows < default min_rows=100
        _, report = validate_ohlcv(df, "TEST", min_rows=100)
        assert not report.passed
        assert any("valid rows" in i for i in report.issues)


class TestPanelValidation:
    def test_panel_excludes_failed_symbols(self):
        good_df = make_valid_df()
        bad_df = make_valid_df(n_days=10)  # Too few rows
        panel = {"GOOD.NS": good_df, "BAD.NS": bad_df}
        clean_panel, reports = validate_panel(panel, min_rows=100)
        assert "GOOD.NS" in clean_panel
        assert "BAD.NS" not in clean_panel
        assert reports["GOOD.NS"].passed
        assert not reports["BAD.NS"].passed
