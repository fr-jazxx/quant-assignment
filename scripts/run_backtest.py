"""
Main CLI entry point for running the full backtest pipeline.

Usage:
    python scripts/run_backtest.py --config-dir config/
    python scripts/run_backtest.py --config-dir config/ --force-refresh
    python scripts/run_backtest.py --config-dir config/ --period train

The script:
    1. Loads and validates all configuration.
    2. Fetches and caches market data.
    3. Generates signals (momentum, mean reversion, regime filter).
    4. Computes strategy-level returns for meta-allocation.
    5. Runs the meta-allocator to get dynamic weights.
    6. Executes the full backtest simulation.
    7. Computes all performance metrics.
    8. Generates charts and exports output files.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import pandas as pd
import numpy as np

from src.utils.config import AppConfig
from src.utils.logger import logger, setup_logger
from src.data.universe import Universe
from src.data.ingestion import fetch_panel, panel_to_wide
from src.signals.momentum import MomentumSignal
from src.signals.mean_reversion import MeanReversionSignal
from src.signals.regime_filter import RegimeFilterSignal
from src.portfolio.construction import PortfolioConstructor
from src.backtest.engine import BacktestEngine
from src.backtest.costs import CostModel
from src.meta_allocator.allocator import MetaAllocator
from src.risk.metrics import compute_all_metrics
from src.analytics.charts import generate_all_charts


@click.command()
@click.option(
    "--config-dir",
    default="config",
    help="Directory containing universe.yaml, strategy.yaml, backtest.yaml",
    show_default=True,
)
@click.option(
    "--cache-dir",
    default="data_cache",
    help="Directory for local Parquet data cache",
    show_default=True,
)
@click.option(
    "--output-dir",
    default="outputs",
    help="Directory for output files (charts, blotter, etc.)",
    show_default=True,
)
@click.option(
    "--force-refresh",
    is_flag=True,
    default=False,
    help="Force re-download of market data (bypass cache)",
)
@click.option(
    "--period",
    default="full",
    type=click.Choice(["full", "train", "validation", "oos"]),
    help="Which period to run the backtest over",
    show_default=True,
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging verbosity",
    show_default=True,
)
def main(
    config_dir: str,
    cache_dir: str,
    output_dir: str,
    force_refresh: bool,
    period: str,
    log_level: str,
) -> None:
    """Run the Indian Equity Quant Research and Backtesting Engine."""

    setup_logger(log_dir="logs", level=log_level)
    logger.info("=" * 70)
    logger.info("IndiQuant Backtesting Engine")
    logger.info("=" * 70)

    # ── 1. Load configuration ──────────────────────────────────────────────
    cfg = AppConfig.from_dir(config_dir)
    np.random.seed(cfg.backtest.random_seed)

    # ── 2. Build universe ──────────────────────────────────────────────────
    universe = Universe.from_config(cfg.universe)

    # ── 3. Fetch market data ───────────────────────────────────────────────
    logger.info("Fetching market data...")
    panel, benchmark_df = fetch_panel(
        universe=universe,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        max_fill_days=cfg.backtest.data.max_fill_days,
        min_rows=cfg.universe.min_trading_days,
    )

    if not panel:
        logger.error("No data fetched — cannot continue.")
        sys.exit(1)

    # Build wide-format price panels
    adj_close = panel_to_wide(panel, column="Adj Close")
    volume = panel_to_wide(panel, column="Volume")
    open_prices = panel_to_wide(panel, column="Open")

    # Benchmark adjusted close
    bench_close = (
        benchmark_df["Adj Close"]
        if benchmark_df is not None and not benchmark_df.empty
        else adj_close.mean(axis=1)
    )

    # ── 4. Apply period mask ───────────────────────────────────────────────
    if period != "full":
        mask = universe.get_period_mask(adj_close.index, period=period)
        adj_close = adj_close[mask]
        volume = volume.reindex(adj_close.index)
        open_prices = open_prices.reindex(adj_close.index)
        bench_close = bench_close.reindex(adj_close.index)
        logger.info(
            f"Period filter applied: '{period}' | "
            f"Dates: {adj_close.index[0].date()} → {adj_close.index[-1].date()}"
        )

    # ── 5. Generate signals ────────────────────────────────────────────────
    logger.info("Generating trading signals...")

    mom_signal = MomentumSignal(cfg.strategy.signals.momentum)
    mr_signal = MeanReversionSignal(cfg.strategy.signals.mean_reversion)
    regime_signal = RegimeFilterSignal(cfg.strategy.signals.regime_filter)

    momentum_output = mom_signal.generate(adj_close, volume=volume)
    mr_output = mr_signal.generate(adj_close)
    regime_output = regime_signal.generate(adj_close, index_prices=bench_close)

    # Extract regime multiplier (scalar per day)
    regime_multiplier = regime_output.weights["regime_multiplier"]

    signal_outputs = {
        momentum_output.name: momentum_output,
        mr_output.name: mr_output,
    }

    # ── 6. Compute per-signal strategy returns (for meta-allocator) ────────
    logger.info("Computing strategy returns for meta-allocator...")
    daily_returns = adj_close.pct_change()

    strategy_returns = pd.DataFrame()
    for name, sig_out in signal_outputs.items():
        sr = MetaAllocator.compute_strategy_returns(sig_out.weights, daily_returns)
        strategy_returns[name] = sr

    # ── 7. Run meta-allocator ──────────────────────────────────────────────
    logger.info("Running meta-allocator...")
    meta_allocator = MetaAllocator(cfg.strategy.meta_allocator)

    # Get monthly rebalance dates for meta-allocation
    from src.signals.momentum import MomentumSignal as _MS
    meta_rebalance_dates = _MS._get_monthly_rebalance_dates(adj_close.index)

    meta_weights = meta_allocator.compute(
        strategy_returns=strategy_returns,
        rebalance_dates=meta_rebalance_dates,
    )

    # ── 8. Run backtest simulation ─────────────────────────────────────────
    logger.info("Running backtest simulation...")
    engine = BacktestEngine(config=cfg)
    result = engine.run(
        prices=adj_close,
        signal_outputs=signal_outputs,
        meta_weights=meta_weights,
        regime_multiplier=regime_multiplier,
        benchmark_prices=bench_close,
        open_prices=open_prices,
    )

    # ── 9. Compute performance metrics ─────────────────────────────────────
    logger.info("Computing performance metrics...")
    metrics = compute_all_metrics(
        equity_curve=result.equity_curve,
        blotter=result.blotter,
        risk_free_rate=cfg.strategy.meta_allocator.risk_free_rate,
        rolling_window=cfg.strategy.meta_allocator.rolling_sharpe_window,
    )

    # ── 10. Print summary ──────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 70)
    for k, v in metrics.to_dict().items():
        logger.info(f"  {k:<35} {v}")

    # ── 11. Save outputs ───────────────────────────────────────────────────
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Equity curve
    result.equity_curve.to_csv(output_path / "equity_curve.csv", header=True)

    # Trade blotter
    if not result.blotter.empty:
        result.blotter.to_csv(output_path / "trade_blotter.csv", index=False)

    # Holdings history
    if not result.holdings_history.empty:
        result.holdings_history.to_csv(output_path / "holdings_history.csv")

    # Meta weights
    result.meta_weights.to_csv(output_path / "meta_allocation_history.csv")

    # Monthly return table
    from src.risk.metrics import compute_monthly_return_table
    monthly_table = compute_monthly_return_table(result.equity_curve)
    monthly_table.to_csv(output_path / "monthly_returns.csv")

    logger.info(f"Output files saved → {output_path.resolve()}")

    # ── 12. Generate charts ────────────────────────────────────────────────
    logger.info("Generating charts...")
    generate_all_charts(result, metrics, output_dir=output_path)

    logger.info("=" * 70)
    logger.info("Backtest complete. Review outputs/ for all generated files.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
