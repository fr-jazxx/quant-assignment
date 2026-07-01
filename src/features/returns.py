"""
Feature engineering functions (returns, volatility, RSI, Z-score, and market breadth).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import logger


# Return Calculations

def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns from adjusted close prices."""
    returns = prices.pct_change()
    logger.debug(f"Daily returns: shape={returns.shape}")
    return returns


def compute_rolling_returns(
    prices: pd.DataFrame,
    window: int,
    skip_days: int = 0,
) -> pd.DataFrame:
    """Compute rolling N-day returns, optionally skipping the most recent days."""
    if skip_days > 0:
        recent_price = prices.shift(skip_days)
        past_price = prices.shift(window)
        rolling_ret = (recent_price / past_price) - 1.0
    else:
        rolling_ret = prices.pct_change(window)

    logger.debug(
        f"Rolling returns: window={window}, skip={skip_days}, shape={rolling_ret.shape}"
    )
    return rolling_ret


def compute_rolling_volatility(
    returns: pd.DataFrame,
    window: int,
    annualisation_factor: int = 252,
) -> pd.DataFrame:
    """Compute rolling annualised volatility."""
    vol = returns.rolling(window=window, min_periods=window // 2).std()
    vol_annualised = vol * np.sqrt(annualisation_factor)
    logger.debug(f"Rolling vol: window={window}, shape={vol_annualised.shape}")
    return vol_annualised


# Cross-Sectional Rank

def cross_sectional_rank(df: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    """Rank each column within each row (cross-sectional ranking)."""
    return df.rank(axis=1, ascending=ascending, na_option="keep", pct=True)


# RSI

def compute_rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute Relative Strength Index (RSI) using Wilder's smoothing."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=window - 1, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    logger.debug(f"RSI({window}): shape={rsi.shape}")
    return rsi


# Z-Score

def compute_zscore(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute rolling z-score of price relative to its rolling mean and std."""
    rolling_mean = prices.rolling(window=window, min_periods=window // 2).mean()
    rolling_std = prices.rolling(window=window, min_periods=window // 2).std()
    zscore = (prices - rolling_mean) / rolling_std.replace(0, np.nan)
    logger.debug(f"Z-score({window}): shape={zscore.shape}")
    return zscore


# Market Breadth

def compute_market_breadth(
    prices: pd.DataFrame,
    sma_window: int = 200,
) -> pd.Series:
    """Compute market breadth: fraction of stocks trading above their SMA."""
    sma = prices.rolling(window=sma_window, min_periods=sma_window // 2).mean()
    above_sma = (prices > sma).sum(axis=1)
    total_valid = prices.notna().sum(axis=1)
    breadth = above_sma / total_valid.replace(0, np.nan)
    logger.debug(f"Market breadth({sma_window}): shape={breadth.shape}")
    return breadth


# Index / Benchmark Volatility

def compute_index_realized_vol(
    index_prices: pd.Series,
    short_window: int = 20,
    long_window: int = 126,
    annualisation_factor: int = 252,
) -> pd.DataFrame:
    """Compute short and long-run realised volatility for the benchmark index."""
    returns = index_prices.pct_change()
    short_vol = (
        returns.rolling(window=short_window, min_periods=short_window // 2)
        .std()
        * np.sqrt(annualisation_factor)
    )
    long_vol = (
        returns.rolling(window=long_window, min_periods=long_window // 2)
        .std()
        * np.sqrt(annualisation_factor)
    )
    vol_ratio = short_vol / long_vol.replace(0, np.nan)

    result = pd.DataFrame(
        {"short_vol": short_vol, "long_vol": long_vol, "vol_ratio": vol_ratio}
    )
    logger.debug(
        f"Index vol ({short_window}/{long_window}): shape={result.shape}"
    )
    return result


__all__ = [
    "compute_daily_returns",
    "compute_rolling_returns",
    "compute_rolling_volatility",
    "cross_sectional_rank",
    "compute_rsi",
    "compute_zscore",
    "compute_market_breadth",
    "compute_index_realized_vol",
]
