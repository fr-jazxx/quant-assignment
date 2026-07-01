"""
Risk and performance metrics for portfolio backtest analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import logger


ANNUALISATION = 252
DEFAULT_RISK_FREE = 0.065  # 6.5% annualised


@dataclass
class PerformanceMetrics:
    """Complete set of performance metrics for a strategy."""

    # Return metrics
    total_return: float
    cagr: float
    annualised_volatility: float

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown metrics
    max_drawdown: float
    max_drawdown_duration_days: int
    avg_drawdown: float

    # Win/loss metrics
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

    # Turnover / trading
    annualised_turnover: float
    total_trades: int

    # Period info
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    n_days: int

    # Monthly return stats
    monthly_returns: pd.Series = field(default_factory=pd.Series)
    rolling_sharpe: pd.Series = field(default_factory=pd.Series)
    rolling_vol: pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)

    def to_dict(self) -> dict[str, Any]:
        """Return scalar metrics as dict (excludes Series fields)."""
        return {
            "Total Return": f"{self.total_return:.2%}",
            "CAGR": f"{self.cagr:.2%}",
            "Annualised Volatility": f"{self.annualised_volatility:.2%}",
            "Sharpe Ratio": f"{self.sharpe_ratio:.3f}",
            "Sortino Ratio": f"{self.sortino_ratio:.3f}",
            "Calmar Ratio": f"{self.calmar_ratio:.3f}",
            "Max Drawdown": f"{self.max_drawdown:.2%}",
            "Max Drawdown Duration (days)": str(self.max_drawdown_duration_days),
            "Win Rate": f"{self.win_rate:.2%}",
            "Avg Win": f"{self.avg_win:.4%}",
            "Avg Loss": f"{self.avg_loss:.4%}",
            "Profit Factor": f"{self.profit_factor:.3f}",
            "Annualised Turnover": f"{self.annualised_turnover:.1%}",
            "Total Trades": str(self.total_trades),
            "Start Date": str(self.start_date.date()),
            "End Date": str(self.end_date.date()),
            "Days": str(self.n_days),
        }


def compute_cagr(returns: pd.Series, annualisation: int = ANNUALISATION) -> float:
    """Compound Annual Growth Rate."""
    cumulative = (1 + returns).prod()
    n_years = len(returns) / annualisation
    if n_years <= 0 or cumulative <= 0:
        return float("nan")
    return cumulative ** (1 / n_years) - 1


def compute_sharpe(
    returns: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE,
    annualisation: int = ANNUALISATION,
) -> float:
    """Annualised Sharpe ratio."""
    if len(returns) < 2:
        return float("nan")
    excess = returns - risk_free_rate / annualisation
    std = excess.std()
    if std == 0:
        return float("nan")
    return (excess.mean() / std) * np.sqrt(annualisation)


def compute_sortino(
    returns: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE,
    annualisation: int = ANNUALISATION,
) -> float:
    """Annualised Sortino ratio (uses downside deviation)."""
    if len(returns) < 2:
        return float("nan")
    excess = returns - risk_free_rate / annualisation
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = downside.std()
    if downside_std == 0:
        return float("nan")
    return (excess.mean() / downside_std) * np.sqrt(annualisation)


def compute_max_drawdown(equity_curve: pd.Series) -> tuple[float, int]:
    """Maximum peak-to-trough drawdown and its duration in days.

    Returns:
        Tuple of (max_drawdown as fraction, duration in calendar days).
    """
    if equity_curve.empty:
        return (0.0, 0)
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Drawdown duration
    in_drawdown = drawdown < 0
    if not in_drawdown.any():
        return (max_dd, 0)

    # Find the longest continuous drawdown period
    groups = (in_drawdown != in_drawdown.shift()).cumsum()
    dd_groups = drawdown[in_drawdown].groupby(groups[in_drawdown])
    durations = []
    for _, group in dd_groups:
        if not group.empty:
            duration = (group.index[-1] - group.index[0]).days
            durations.append(duration)
    max_duration = max(durations) if durations else 0
    return (float(max_dd), max_duration)


def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Drawdown at each point in time."""
    rolling_max = equity_curve.cummax()
    return (equity_curve - rolling_max) / rolling_max


def compute_calmar(
    cagr: float,
    max_drawdown: float,
) -> float:
    """Calmar ratio: CAGR / |Max Drawdown|."""
    if max_drawdown == 0 or np.isnan(max_drawdown):
        return float("nan")
    return cagr / abs(max_drawdown)


def compute_win_loss(returns: pd.Series) -> tuple[float, float, float, float]:
    """Win rate, avg win, avg loss, profit factor.

    Returns:
        Tuple of (win_rate, avg_win, avg_loss, profit_factor).
    """
    if len(returns) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / len(returns)
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    total_gains = wins.sum()
    total_losses = abs(losses.sum())
    profit_factor = total_gains / total_losses if total_losses > 0 else float("inf")
    return (win_rate, avg_win, avg_loss, profit_factor)


def compute_rolling_sharpe(
    returns: pd.Series,
    window: int = 60,
    risk_free_rate: float = DEFAULT_RISK_FREE,
    annualisation: int = ANNUALISATION,
) -> pd.Series:
    """Rolling Sharpe ratio over a given window."""
    excess = returns - risk_free_rate / annualisation
    rolling_mean = excess.rolling(window=window, min_periods=window // 2).mean()
    rolling_std = excess.rolling(window=window, min_periods=window // 2).std()
    return (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(annualisation)


def compute_rolling_vol(
    returns: pd.Series,
    window: int = 60,
    annualisation: int = ANNUALISATION,
) -> pd.Series:
    """Rolling annualised volatility."""
    return returns.rolling(window=window, min_periods=window // 2).std() * np.sqrt(annualisation)


def compute_monthly_returns(equity_curve: pd.Series) -> pd.Series:
    """Resample daily equity curve to monthly returns."""
    monthly = equity_curve.resample("ME").last()
    return monthly.pct_change().dropna()


def compute_monthly_return_table(equity_curve: pd.Series) -> pd.DataFrame:
    """Create a year × month table of monthly returns."""
    monthly = compute_monthly_returns(equity_curve)
    df = monthly.to_frame("return")
    df["year"] = df.index.year
    df["month"] = df.index.month
    table = df.pivot(index="year", columns="month", values="return")
    MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    table.columns = [MONTH_NAMES[col - 1] for col in table.columns]
    return table


def compute_all_metrics(
    equity_curve: pd.Series,
    blotter: pd.DataFrame | None = None,
    risk_free_rate: float = DEFAULT_RISK_FREE,
    rolling_window: int = 60,
) -> PerformanceMetrics:
    """Compute the full set of performance metrics.

    Args:
        equity_curve: Daily portfolio value series.
        blotter: Trade blotter DataFrame (for turnover and trade count).
        risk_free_rate: Annualised risk-free rate.
        rolling_window: Window for rolling Sharpe and vol.

    Returns:
        PerformanceMetrics with all computed values.
    """
    returns = equity_curve.pct_change().dropna()

    if len(returns) < 10:
        logger.warning("Too few return observations for reliable metrics")

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    cagr = compute_cagr(returns)
    ann_vol = returns.std() * np.sqrt(ANNUALISATION)
    sharpe = compute_sharpe(returns, risk_free_rate)
    sortino = compute_sortino(returns, risk_free_rate)
    max_dd, dd_duration = compute_max_drawdown(equity_curve)
    calmar = compute_calmar(cagr, max_dd)
    win_rate, avg_win, avg_loss, profit_factor = compute_win_loss(returns)

    # Drawdown series
    dd_series = compute_drawdown_series(equity_curve)
    avg_drawdown = float(dd_series[dd_series < 0].mean()) if (dd_series < 0).any() else 0.0

    # Rolling metrics
    rolling_sharpe = compute_rolling_sharpe(returns, rolling_window, risk_free_rate)
    rolling_vol = compute_rolling_vol(returns, rolling_window)

    # Monthly returns
    monthly_returns = compute_monthly_returns(equity_curve)

    # Trade stats
    total_trades = len(blotter) if blotter is not None and not blotter.empty else 0
    if blotter is not None and not blotter.empty and "gross_value" in blotter.columns:
        total_turnover = blotter["gross_value"].sum()
        n_years = len(returns) / ANNUALISATION
        avg_portfolio_value = equity_curve.mean()
        ann_turnover = (total_turnover / avg_portfolio_value) / n_years if n_years > 0 else 0.0
    else:
        ann_turnover = float("nan")

    metrics = PerformanceMetrics(
        total_return=float(total_return),
        cagr=float(cagr),
        annualised_volatility=float(ann_vol),
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        calmar_ratio=float(calmar),
        max_drawdown=float(max_dd),
        max_drawdown_duration_days=int(dd_duration),
        avg_drawdown=float(avg_drawdown),
        win_rate=float(win_rate),
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
        profit_factor=float(profit_factor),
        annualised_turnover=float(ann_turnover),
        total_trades=int(total_trades),
        start_date=equity_curve.index[0],
        end_date=equity_curve.index[-1],
        n_days=len(returns),
        monthly_returns=monthly_returns,
        rolling_sharpe=rolling_sharpe,
        rolling_vol=rolling_vol,
        drawdown_series=dd_series,
    )

    logger.info(
        f"Performance | CAGR: {cagr:.2%} | Sharpe: {sharpe:.3f} | "
        f"MaxDD: {max_dd:.2%} | Calmar: {calmar:.3f}"
    )
    return metrics


__all__ = [
    "PerformanceMetrics",
    "compute_all_metrics",
    "compute_cagr",
    "compute_sharpe",
    "compute_sortino",
    "compute_max_drawdown",
    "compute_drawdown_series",
    "compute_monthly_return_table",
    "compute_rolling_sharpe",
]
