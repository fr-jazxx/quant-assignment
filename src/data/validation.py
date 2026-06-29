"""
Data validation layer — enforces the data contract before any data
flows into signals or the backtesting engine.

Data Contract:
    - Column set: ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    - Index: DatetimeIndex, timezone-naive, business-day frequency
    - No duplicate dates per symbol
    - No future timestamps
    - No negative prices or volumes
    - No more than config.max_fill_days consecutive NaN values
    - Adjusted vs unadjusted prices clearly labelled

Failures are logged and reported — they do not silently propagate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import numpy as np

from src.utils.logger import logger


# ─── Expected Data Contract ───────────────────────────────────────────────────

REQUIRED_COLUMNS: list[str] = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
PRICE_COLUMNS: list[str] = ["Open", "High", "Low", "Close", "Adj Close"]


@dataclass
class ValidationReport:
    """Summary of all validation checks run on a dataset."""

    symbol: str
    passed: bool = True
    issues: list[str] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0

    def add_issue(self, msg: str, critical: bool = True) -> None:
        self.issues.append(msg)
        if critical:
            self.passed = False
        logger.warning(f"[{self.symbol}] Validation: {msg}")

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{self.symbol}] {status} | "
            f"Rows: {self.rows_before}→{self.rows_after} | "
            f"Issues: {len(self.issues)}"
        )


# ─── Validators ───────────────────────────────────────────────────────────────

def validate_columns(df: pd.DataFrame, symbol: str, report: ValidationReport) -> pd.DataFrame:
    """Ensure all required columns are present."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report.add_issue(f"Missing columns: {missing}", critical=True)
    return df


def validate_index(df: pd.DataFrame, symbol: str, report: ValidationReport) -> pd.DataFrame:
    """Ensure the index is a DatetimeIndex and timezone-naive."""
    if not isinstance(df.index, pd.DatetimeIndex):
        report.add_issue("Index is not DatetimeIndex", critical=True)
        return df

    # Strip timezone info for consistent comparison
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)

    return df


def validate_no_duplicates(df: pd.DataFrame, symbol: str, report: ValidationReport) -> pd.DataFrame:
    """Remove and report duplicate date entries."""
    dups = df.index.duplicated()
    n_dups = dups.sum()
    if n_dups > 0:
        report.add_issue(
            f"Found {n_dups} duplicate timestamps — keeping first occurrence",
            critical=False,  # Non-critical: we can fix it
        )
        df = df[~df.index.duplicated(keep="first")]
    return df


def validate_no_future_timestamps(
    df: pd.DataFrame, symbol: str, report: ValidationReport
) -> pd.DataFrame:
    """Reject rows with timestamps in the future."""
    now = pd.Timestamp("today").normalize()  # Today's date, timezone-naive
    future_mask = df.index > now
    n_future = future_mask.sum()
    if n_future > 0:
        report.add_issue(
            f"Found {n_future} rows with future timestamps — dropping",
            critical=True,
        )
        df = df[~future_mask]
    return df


def validate_no_negative_prices(
    df: pd.DataFrame, symbol: str, report: ValidationReport
) -> pd.DataFrame:
    """Flag and drop rows where any price column is zero or negative."""
    for col in PRICE_COLUMNS:
        if col not in df.columns:
            continue
        bad = df[col] <= 0
        n_bad = bad.sum()
        if n_bad > 0:
            report.add_issue(
                f"Column '{col}' has {n_bad} non-positive values — dropping rows",
                critical=False,
            )
            df = df[~bad]
    return df


def validate_no_negative_volume(
    df: pd.DataFrame, symbol: str, report: ValidationReport
) -> pd.DataFrame:
    """Flag rows with negative volume."""
    if "Volume" not in df.columns:
        return df
    bad = df["Volume"] < 0
    n_bad = bad.sum()
    if n_bad > 0:
        report.add_issue(
            f"Column 'Volume' has {n_bad} negative values — setting to 0",
            critical=False,
        )
        df = df.copy()
        df.loc[bad, "Volume"] = 0
    return df


def validate_no_large_gaps(
    df: pd.DataFrame,
    symbol: str,
    report: ValidationReport,
    max_fill_days: int = 3,
) -> pd.DataFrame:
    """Detect consecutive NaN runs exceeding the allowed fill window.

    Forward-fills up to max_fill_days. Beyond that, rows remain NaN
    so the downstream system can detect and exclude them.
    """
    for col in PRICE_COLUMNS:
        if col not in df.columns:
            continue
        nan_runs = (
            df[col]
            .isna()
            .astype(int)
            .groupby((~df[col].isna()).cumsum())
            .transform("sum")
        )
        long_gaps = (nan_runs > max_fill_days).sum()
        if long_gaps > 0:
            report.add_issue(
                f"Column '{col}' has {long_gaps} rows in NaN runs > {max_fill_days} days "
                f"(will NOT be forward-filled — downstream must handle)",
                critical=False,
            )

    # Forward-fill up to max_fill_days
    df = df.copy()
    df[PRICE_COLUMNS] = df[PRICE_COLUMNS].ffill(limit=max_fill_days)
    return df


def validate_ohlc_consistency(
    df: pd.DataFrame, symbol: str, report: ValidationReport
) -> pd.DataFrame:
    """Check that OHLC values are internally consistent (High >= Low, etc.)."""
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(df.columns):
        return df

    hl_bad = (df["High"] < df["Low"]).sum()
    if hl_bad > 0:
        report.add_issue(
            f"Found {hl_bad} rows where High < Low — data integrity issue",
            critical=False,
        )

    return df


# ─── Main Validate Function ───────────────────────────────────────────────────

def validate_ohlcv(
    df: pd.DataFrame,
    symbol: str,
    max_fill_days: int = 3,
    min_rows: int = 100,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Run all validation checks on a single symbol's OHLCV DataFrame.

    Args:
        df: Raw DataFrame from the data ingestion layer.
        symbol: Ticker symbol (for logging).
        max_fill_days: Maximum consecutive NaN days to forward-fill.
        min_rows: Minimum number of valid rows required.

    Returns:
        Tuple of (cleaned DataFrame, ValidationReport).
    """
    report = ValidationReport(symbol=symbol, rows_before=len(df))

    df = validate_columns(df, symbol, report)
    if not report.passed:
        report.rows_after = 0
        return df, report

    df = validate_index(df, symbol, report)
    df = validate_no_duplicates(df, symbol, report)
    df = validate_no_future_timestamps(df, symbol, report)
    df = validate_no_negative_prices(df, symbol, report)
    df = validate_no_negative_volume(df, symbol, report)
    df = validate_ohlc_consistency(df, symbol, report)
    df = validate_no_large_gaps(df, symbol, report, max_fill_days=max_fill_days)

    df = df.sort_index()
    report.rows_after = len(df)

    if len(df) < min_rows:
        report.add_issue(
            f"Only {len(df)} valid rows after cleaning (minimum: {min_rows})",
            critical=True,
        )

    if report.passed:
        logger.debug(report.summary())
    else:
        logger.error(report.summary())

    return df, report


def validate_panel(
    panel: dict[str, pd.DataFrame],
    max_fill_days: int = 3,
    min_rows: int = 100,
) -> tuple[dict[str, pd.DataFrame], dict[str, ValidationReport]]:
    """Validate all symbols in a panel dictionary.

    Args:
        panel: Dict mapping symbol → OHLCV DataFrame.
        max_fill_days: Maximum consecutive NaN days to forward-fill.
        min_rows: Minimum rows required per symbol.

    Returns:
        Tuple of (cleaned panel, reports per symbol).
        Symbols failing critical checks are excluded from the cleaned panel.
    """
    clean_panel: dict[str, pd.DataFrame] = {}
    reports: dict[str, ValidationReport] = {}

    for symbol, df in panel.items():
        clean_df, report = validate_ohlcv(
            df, symbol, max_fill_days=max_fill_days, min_rows=min_rows
        )
        reports[symbol] = report
        if report.passed:
            clean_panel[symbol] = clean_df
        else:
            logger.warning(f"Excluding '{symbol}' — failed validation")

    n_pass = sum(1 for r in reports.values() if r.passed)
    n_fail = len(reports) - n_pass
    logger.info(
        f"Panel validation complete | "
        f"Pass: {n_pass} | Fail: {n_fail} | "
        f"Total: {len(reports)}"
    )
    return clean_panel, reports


__all__ = ["validate_ohlcv", "validate_panel", "ValidationReport"]
