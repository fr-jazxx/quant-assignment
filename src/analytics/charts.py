"""
Plotly-based chart generation for the research report.

All charts are saved as interactive HTML and static PNG files to outputs/.
Charts are designed to be clear, professional, and self-explanatory
for inclusion in the research report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.risk.metrics import PerformanceMetrics, compute_monthly_return_table
from src.utils.logger import logger


def _save(fig: go.Figure, name: str, output_dir: Path) -> None:
    """Save chart as both HTML and PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{name}.html"
    png_path = output_dir / f"{name}.png"
    fig.write_html(str(html_path))
    try:
        fig.write_image(str(png_path), width=1400, height=700)
        logger.info(f"Saved chart: {png_path}")
    except Exception as e:
        logger.warning(f"PNG export failed (kaleido issue?): {e}. HTML saved: {html_path}")


COLORS = {
    "strategy": "#00D4AA",
    "benchmark": "#FF6B6B",
    "drawdown": "#FF4757",
    "momentum": "#3742FA",
    "mean_reversion": "#FFA502",
    "regime": "#2ED573",
    "grid": "rgba(255,255,255,0.1)",
    "bg": "#0D1117",
    "paper": "#161B22",
    "text": "#E6EDF3",
}

LAYOUT = dict(
    plot_bgcolor=COLORS["bg"],
    paper_bgcolor=COLORS["paper"],
    font=dict(color=COLORS["text"], family="Inter, sans-serif", size=12),
    legend=dict(
        bgcolor="rgba(22,27,34,0.8)",
        bordercolor="rgba(255,255,255,0.2)",
        borderwidth=1,
    ),
    margin=dict(l=60, r=40, t=60, b=60),
)


def plot_equity_curve(
    strategy_curve: pd.Series,
    benchmark_curve: pd.Series,
    metrics: PerformanceMetrics,
    output_dir: Path,
    title: str = "Portfolio Equity Curve vs NIFTY 50",
) -> None:
    """Plot normalised equity curves for strategy and benchmark."""
    # Normalise to 100
    s_norm = strategy_curve / strategy_curve.iloc[0] * 100
    b_norm = benchmark_curve.reindex(strategy_curve.index).ffill()
    b_norm = b_norm / b_norm.iloc[0] * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s_norm.index, y=s_norm.values,
        name=f"Strategy (CAGR: {metrics.cagr:.1%}, SR: {metrics.sharpe_ratio:.2f})",
        line=dict(color=COLORS["strategy"], width=2),
        fill="none",
    ))
    fig.add_trace(go.Scatter(
        x=b_norm.index, y=b_norm.values,
        name="NIFTY 50 Benchmark",
        line=dict(color=COLORS["benchmark"], width=2, dash="dash"),
    ))

    # Add training/validation/OOS shading labels
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(title="Date", gridcolor=COLORS["grid"]),
        yaxis=dict(title="Portfolio Value (Base = 100)", gridcolor=COLORS["grid"]),
        **LAYOUT,
    )
    _save(fig, "equity_curve", output_dir)


def plot_drawdown_curve(
    drawdown_series: pd.Series,
    output_dir: Path,
    title: str = "Portfolio Drawdown from Peak",
) -> None:
    """Plot drawdown curve with key periods annotated."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown_series.index,
        y=(drawdown_series * 100).values,
        name="Drawdown (%)",
        fill="tozeroy",
        line=dict(color=COLORS["drawdown"], width=1),
        fillcolor="rgba(255,71,87,0.3)",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(title="Date", gridcolor=COLORS["grid"]),
        yaxis=dict(title="Drawdown (%)", gridcolor=COLORS["grid"]),
        **LAYOUT,
    )
    _save(fig, "drawdown_curve", output_dir)


def plot_monthly_heatmap(
    equity_curve: pd.Series,
    output_dir: Path,
    title: str = "Monthly Returns Heatmap",
) -> None:
    """Plot monthly return heatmap (year × month)."""
    table = compute_monthly_return_table(equity_curve)
    z_vals = table.values * 100  # Convert to %

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=table.columns.tolist(),
        y=table.index.tolist(),
        colorscale=[
            [0.0, "#FF4757"],    # Deep red for large losses
            [0.4, "#FF6B6B"],    # Light red
            [0.5, "#2D2D2D"],    # Neutral grey
            [0.6, "#2ED573"],    # Light green
            [1.0, "#00B34A"],    # Deep green for large gains
        ],
        text=np.round(z_vals, 1).astype(str),
        texttemplate="%{text}%",
        textfont=dict(size=10),
        zmid=0,
        colorbar=dict(title="Return %"),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(title="Month"),
        yaxis=dict(title="Year"),
        **LAYOUT,
    )
    _save(fig, "monthly_heatmap", output_dir)


def plot_rolling_metrics(
    rolling_sharpe: pd.Series,
    rolling_vol: pd.Series,
    output_dir: Path,
    title: str = "Rolling 60-Day Sharpe and Volatility",
) -> None:
    """Plot rolling Sharpe and volatility on a dual-axis chart."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            name="Rolling Sharpe (60d)",
            line=dict(color=COLORS["strategy"], width=2),
        ),
        secondary_y=False,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")

    fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=(rolling_vol * 100).values,
            name="Rolling Vol % (60d)",
            line=dict(color=COLORS["benchmark"], width=1.5, dash="dot"),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(gridcolor=COLORS["grid"]),
        **LAYOUT,
    )
    fig.update_yaxes(title_text="Sharpe Ratio", secondary_y=False, gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="Volatility (%)", secondary_y=True)
    _save(fig, "rolling_metrics", output_dir)


def plot_meta_allocation(
    meta_weights: pd.DataFrame,
    output_dir: Path,
    title: str = "Meta-Allocator: Strategy Weights Over Time",
) -> None:
    """Stacked area chart of meta-allocation weights over time."""
    strategy_colors = {
        "momentum_12_1": COLORS["momentum"],
        "mean_reversion_rsi": COLORS["mean_reversion"],
        "regime_filter_vol_breadth": COLORS["regime"],
    }

    fig = go.Figure()
    for col in meta_weights.columns:
        fig.add_trace(go.Scatter(
            x=meta_weights.index,
            y=(meta_weights[col] * 100).values,
            name=col.replace("_", " ").title(),
            mode="lines",
            stackgroup="one",
            line=dict(color=strategy_colors.get(col, "#AAAAAA"), width=0.5),
            fillcolor=strategy_colors.get(col, "#AAAAAA"),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(title="Date", gridcolor=COLORS["grid"]),
        yaxis=dict(title="Allocation Weight (%)", gridcolor=COLORS["grid"]),
        **LAYOUT,
    )
    _save(fig, "meta_allocation", output_dir)


def plot_signal_comparison(
    signal_metrics: dict[str, PerformanceMetrics],
    output_dir: Path,
    title: str = "Individual Signal Performance Comparison",
) -> None:
    """Bar chart comparing key metrics across all signals."""
    names = list(signal_metrics.keys())
    cagrs = [m.cagr * 100 for m in signal_metrics.values()]
    sharpes = [m.sharpe_ratio for m in signal_metrics.values()]
    max_dds = [abs(m.max_drawdown) * 100 for m in signal_metrics.values()]

    fig = make_subplots(rows=1, cols=3, subplot_titles=["CAGR (%)", "Sharpe Ratio", "Max Drawdown (%)"])
    colors = [COLORS["momentum"], COLORS["mean_reversion"], COLORS["regime"], COLORS["strategy"]]

    for i, (name, cagr) in enumerate(zip(names, cagrs)):
        fig.add_trace(go.Bar(x=[name], y=[cagr], marker_color=colors[i % len(colors)], name=name, showlegend=False), row=1, col=1)
    for i, (name, sr) in enumerate(zip(names, sharpes)):
        fig.add_trace(go.Bar(x=[name], y=[sr], marker_color=colors[i % len(colors)], name=name, showlegend=False), row=1, col=2)
    for i, (name, dd) in enumerate(zip(names, max_dds)):
        fig.add_trace(go.Bar(x=[name], y=[dd], marker_color=colors[i % len(colors)], name=name, showlegend=False), row=1, col=3)

    fig.update_layout(title=dict(text=title, font=dict(size=18)), **LAYOUT)
    _save(fig, "signal_comparison", output_dir)


def plot_regime_overlay(
    equity_curve: pd.Series,
    regime_multiplier: pd.Series,
    output_dir: Path,
    title: str = "Equity Curve with Regime Overlay",
) -> None:
    """Equity curve with regime risk reduction periods highlighted."""
    s_norm = equity_curve / equity_curve.iloc[0] * 100
    low_risk = regime_multiplier < 1.0

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=s_norm.index, y=s_norm.values,
            name="Portfolio Value (Base=100)",
            line=dict(color=COLORS["strategy"], width=2),
        ), secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=regime_multiplier.index, y=regime_multiplier.values,
            name="Regime Multiplier",
            line=dict(color=COLORS["benchmark"], width=1, dash="dot"),
        ), secondary_y=True,
    )

    fig.update_layout(title=dict(text=title, font=dict(size=18)), xaxis=dict(gridcolor=COLORS["grid"]), **LAYOUT)
    fig.update_yaxes(title_text="Portfolio Value", secondary_y=False, gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="Regime Multiplier", secondary_y=True, range=[0, 1.5])
    _save(fig, "regime_overlay", output_dir)


def generate_all_charts(
    result,  # BacktestResult
    metrics: PerformanceMetrics,
    output_dir: str | Path = "outputs",
) -> None:
    """Generate all standard charts and save to output_dir."""
    output_dir = Path(output_dir)
    logger.info(f"Generating charts → {output_dir}")

    plot_equity_curve(
        result.equity_curve,
        result.benchmark,
        metrics,
        output_dir,
    )
    plot_drawdown_curve(metrics.drawdown_series, output_dir)
    plot_monthly_heatmap(result.equity_curve, output_dir)
    plot_rolling_metrics(metrics.rolling_sharpe, metrics.rolling_vol, output_dir)
    plot_meta_allocation(result.meta_weights, output_dir)
    plot_regime_overlay(result.equity_curve, result.regime_multiplier, output_dir)

    logger.info("All charts generated successfully.")


__all__ = ["generate_all_charts", "plot_equity_curve", "plot_drawdown_curve", "plot_monthly_heatmap"]
