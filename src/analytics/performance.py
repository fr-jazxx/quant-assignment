"""
Performance report generation — compiles all metrics into structured
output formats for the research report.

Produces:
    - Tabular summary comparison (strategy vs benchmark)
    - Period breakdown (train / validation / OOS)
    - Gross vs net performance table (shows cost impact)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.risk.metrics import (
    PerformanceMetrics,
    compute_all_metrics,
    compute_monthly_return_table,
)
from src.utils.logger import logger


def build_comparison_table(
    strategy_metrics: PerformanceMetrics,
    benchmark_curve: pd.Series,
    risk_free_rate: float = 0.065,
) -> pd.DataFrame:
    """Build a side-by-side comparison table: strategy vs NIFTY 50.

    Args:
        strategy_metrics: Pre-computed strategy PerformanceMetrics.
        benchmark_curve: NIFTY 50 benchmark equity curve.
        risk_free_rate: Annualised risk-free rate for Sharpe computation.

    Returns:
        DataFrame with metrics as rows, strategy/benchmark as columns.
    """
    bench_metrics = compute_all_metrics(
        equity_curve=benchmark_curve,
        risk_free_rate=risk_free_rate,
    )

    strategy_dict = strategy_metrics.to_dict()
    bench_dict = bench_metrics.to_dict()

    comparison = pd.DataFrame(
        {
            "Strategy": strategy_dict,
            "NIFTY 50 Benchmark": bench_dict,
        }
    )
    return comparison


def build_period_breakdown(
    equity_curve: pd.Series,
    blotter: pd.DataFrame | None,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    oos_start: pd.Timestamp,
    risk_free_rate: float = 0.065,
) -> pd.DataFrame:
    """Compute performance metrics for each evaluation period separately.

    Args:
        equity_curve: Full portfolio equity curve.
        blotter: Trade blotter for turnover calculation.
        train_end: End of training period.
        validation_end: End of validation period.
        oos_start: Start of OOS period.
        risk_free_rate: Annualised risk-free rate.

    Returns:
        DataFrame with periods as columns and metrics as rows.
    """
    periods = {
        "Training (2015–2017)": equity_curve[equity_curve.index <= train_end],
        "Validation (2018–2019)": equity_curve[
            (equity_curve.index > train_end) & (equity_curve.index <= validation_end)
        ],
        "OOS / Stress Test (2020)": equity_curve[equity_curve.index >= oos_start],
    }

    rows: dict[str, dict] = {}
    for period_name, curve in periods.items():
        if len(curve) < 5:
            continue
        period_blotter = None
        if blotter is not None and not blotter.empty:
            period_blotter = blotter[
                (blotter["date"] >= curve.index[0]) & (blotter["date"] <= curve.index[-1])
            ]
        m = compute_all_metrics(curve, period_blotter, risk_free_rate)
        rows[period_name] = m.to_dict()

    return pd.DataFrame(rows)


def build_gross_vs_net_table(
    gross_curve: pd.Series,
    net_curve: pd.Series,
    risk_free_rate: float = 0.065,
) -> pd.DataFrame:
    """Compare gross vs net performance to show cost impact.

    Args:
        gross_curve: Equity curve before transaction costs.
        net_curve: Equity curve after all costs.
        risk_free_rate: Annualised risk-free rate.

    Returns:
        DataFrame showing the cost drag on each metric.
    """
    gross_m = compute_all_metrics(gross_curve, risk_free_rate=risk_free_rate)
    net_m = compute_all_metrics(net_curve, risk_free_rate=risk_free_rate)

    return pd.DataFrame({
        "Gross (Before Costs)": gross_m.to_dict(),
        "Net (After Costs)": net_m.to_dict(),
    })


def save_performance_outputs(
    equity_curve: pd.Series,
    strategy_metrics: PerformanceMetrics,
    benchmark_curve: pd.Series,
    blotter: pd.DataFrame | None,
    meta_weights: pd.DataFrame,
    holdings_history: pd.DataFrame,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    oos_start: pd.Timestamp,
    output_dir: str | Path = "outputs",
    risk_free_rate: float = 0.065,
) -> None:
    """Save all performance tables to CSV files.

    Args:
        equity_curve: Full portfolio equity curve.
        strategy_metrics: Pre-computed PerformanceMetrics.
        benchmark_curve: Benchmark equity curve.
        blotter: Trade blotter.
        meta_weights: Meta-allocator weight history.
        holdings_history: Portfolio holdings at each snapshot.
        train_end: End of training period.
        validation_end: End of validation period.
        oos_start: Start of OOS period.
        output_dir: Directory to write CSV files.
        risk_free_rate: Annualised risk-free rate.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Strategy vs benchmark comparison
    comparison = build_comparison_table(
        strategy_metrics, benchmark_curve, risk_free_rate
    )
    comparison.to_csv(out / "performance_comparison.csv")
    logger.info(f"Saved: {out}/performance_comparison.csv")

    # Period breakdown
    period_breakdown = build_period_breakdown(
        equity_curve, blotter,
        train_end, validation_end, oos_start, risk_free_rate,
    )
    period_breakdown.to_csv(out / "period_breakdown.csv")
    logger.info(f"Saved: {out}/period_breakdown.csv")

    # Monthly returns table
    monthly_table = compute_monthly_return_table(equity_curve)
    monthly_table.to_csv(out / "monthly_returns_table.csv")
    logger.info(f"Saved: {out}/monthly_returns_table.csv")

    # Meta-allocation summary
    meta_summary = meta_weights.describe()
    meta_summary.to_csv(out / "meta_allocation_summary.csv")
    logger.info(f"Saved: {out}/meta_allocation_summary.csv")


__all__ = [
    "build_comparison_table",
    "build_period_breakdown",
    "build_gross_vs_net_table",
    "save_performance_outputs",
]
