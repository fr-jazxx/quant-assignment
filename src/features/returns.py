"""
Feature engineering — return calculations, rolling statistics,
RSI, z-scores, and volatility measures.

All functions operate on wide-format DataFrames (DatetimeIndex × symbols).
No future data is used — all rolling operations are strictly backward-looking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import logger


# ─── Return Calculations ──────────────────────────────────────────────────────

def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns from adjusted close prices.

    Args:
        prices: Wide DataFrame of adjusted close prices (Date × Symbol).

    Returns:
        DataFrame of daily returns. First row is NaN.
    """
    returns = prices.pct_change()
    logger.debug(f"Daily returns: shape={returns.shape}")
    return returns


def compute_rolling_returns(
    prices: pd.DataFrame,
    window: int,
    skip_days: int = 0,
) -> pd.DataFrame:
    """Compute rolling N-day returns, optionally skipping the most recent days.

    This is used for momentum signals: 12-month return excluding last month
    is computed as rolling(252, skip=21).

    Strictly backward-looking: on date t, uses prices from t-window to t-skip.
    No look-ahead bias introduced.

    Args:
        prices: Wide DataFrame of adjusted close prices.
        window: Total lookback window in trading days.
        skip_days: Number of most-recent days to exclude from return calculation.
                   Set to 21 for 12-1 momentum (Jegadeesh & Titman, 1993).

    Returns:
        DataFrame of rolling returns.
    """
    if skip_days > 0:
        # Return from (t - window) to (t - skip_days)
        # Shift prices by skip_days to get the "recent" denominator price
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
    """Compute rolling annualised volatility.

    Args:
        returns: DataFrame of daily returns.
        window: Rolling window in trading days.
        annualisation_factor: Trading days per year (252 for equity).

    Returns:
        DataFrame of annualised rolling volatility.
    """
    vol = returns.rolling(window=window, min_periods=window // 2).std()
    vol_annualised = vol * np.sqrt(annualisation_factor)
    logger.debug(f"Rolling vol: window={window}, shape={vol_annualised.shape}")
    return vol_annualised


# ─── Cross-Sectional Rank ─────────────────────────────────────────────────────

def cross_sectional_rank(df: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    """Rank each column within each row (cross-sectional ranking).

    Used for momentum signal: rank stocks by their rolling return score
    on each date independently.

    Args:
        df: DataFrame of scores (Date × Symbol).
        ascending: If True, rank from lowest to highest (1 = lowest score).

    Returns:
        DataFrame of ranks. NaN stocks are excluded from ranking.
    """
    return df.rank(axis=1, ascending=ascending, na_option="keep", pct=True)


# ─── RSI ─────────────────────────────────────────────────────────────────────

def compute_rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute Relative Strength Index (RSI) using Wilder's smoothing.

    RSI formula:
        RS = AvgGain / AvgLoss  (over window)
        RSI = 100 - (100 / (1 + RS))

    Args:
        prices: Wide DataFrame of adjusted close prices.
        window: RSI lookback window (typically 5 for short-term, 14 for classic).

    Returns:
        DataFrame of RSI values in [0, 100].
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=window - 1, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    logger.debug(f"RSI({window}): shape={rsi.shape}")
    return rsi


# ─── Z-Score ─────────────────────────────────────────────────────────────────

def compute_zscore(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute rolling z-score of price relative to its rolling mean and std.

    z_t = (Price_t - Mean(Price_{t-window:t})) / Std(Price_{t-window:t})

    Values < -2 indicate a stock is unusually cheap relative to recent history.

    Args:
        prices: Wide DataFrame of adjusted close prices.
        window: Rolling window in trading days.

    Returns:
        DataFrame of rolling z-scores.
    """
    rolling_mean = prices.rolling(window=window, min_periods=window // 2).mean()
    rolling_std = prices.rolling(window=window, min_periods=window // 2).std()
    zscore = (prices - rolling_mean) / rolling_std.replace(0, np.nan)
    logger.debug(f"Z-score({window}): shape={zscore.shape}")
    return zscore


# ─── Market Breadth ──────────────────────────────────────────────────────────

def compute_market_breadth(
    prices: pd.DataFrame,
    sma_window: int = 200,
) -> pd.Series:
    """Compute market breadth: fraction of stocks trading above their SMA.

    This is a proxy for broad market health. A breadth reading below 40%
    suggests a deteriorating market environment.

    Args:
        prices: Wide DataFrame of adjusted close prices.
        sma_window: Simple moving average window (200 for long-term trend).

    Returns:
        Series indexed by date with values in [0, 1].
    """
    sma = prices.rolling(window=sma_window, min_periods=sma_window // 2).mean()
    above_sma = (prices > sma).sum(axis=1)
    total_valid = prices.notna().sum(axis=1)
    breadth = above_sma / total_valid.replace(0, np.nan)
    logger.debug(f"Market breadth({sma_window}): shape={breadth.shape}")
    return breadth


# ─── Index / Benchmark Volatility ────────────────────────────────────────────

def compute_index_realized_vol(
    index_prices: pd.Series,
    short_window: int = 20,
    long_window: int = 126,
    annualisation_factor: int = 252,
) -> pd.DataFrame:
    """Compute short and long-run realised volatility for the benchmark index.

    Used by the regime filter to identify elevated volatility environments.

    Args:
        index_prices: Series of benchmark index adjusted close prices.
        short_window: Recent volatility window (e.g. 20 days).
        long_window: Long-run volatility window (e.g. 126 days ≈ 6 months).
        annualisation_factor: Trading days per year.

    Returns:
        DataFrame with columns ['short_vol', 'long_vol', 'vol_ratio'].
    """
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
