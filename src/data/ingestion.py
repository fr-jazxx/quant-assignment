"""
Data ingestion layer — fetches historical OHLCV data from yfinance,
caches locally as Parquet files, and returns clean DataFrames.

Design Decisions & Limitations:
    - Data source: yfinance (Yahoo Finance)
    - Prices: Adjusted Close is used as the primary price for returns.
      Unadjusted OHLCV is retained for volume-based filters.
    - Corporate actions: yfinance applies split/dividend adjustments
      retroactively to ALL historical data. This means the adjusted
      price history changes over time. This is NOT point-in-time accurate.
    - Survivorship bias: The symbol list is static (as of Dec 2023).
      Delisted companies that exited NIFTY 100 before this date are
      excluded. This introduces upward bias in backtested returns.
    - Intraday data: Not available. All signals use End-of-Day (EOD) data.
    - All limitations are documented in the research report.

Cache:
    - First fetch is stored as Parquet in data_cache/<symbol>.parquet.
    - Subsequent runs load from cache unless force_refresh=True.
    - Cache key is: symbol + date range (stale cache is detected by range).
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.data.validation import validate_panel
from src.data.universe import Universe
from src.utils.config import UniverseConfig, BacktestConfig
from src.utils.logger import logger


# ─── Cache Utilities ──────────────────────────────────────────────────────────

def _cache_path(cache_dir: Path, symbol: str) -> Path:
    """Return the Parquet file path for a given symbol."""
    safe_symbol = symbol.replace(".", "_").replace("^", "IDX_")
    return cache_dir / f"{safe_symbol}.parquet"


def _is_cache_valid(
    path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> bool:
    """Check whether the cached file covers the required date range.

    A cache is considered valid if:
        1. The file exists.
        2. The cached data's index covers [start, end].
    """
    if not path.exists():
        return False
    try:
        cached = pd.read_parquet(path, columns=["Close"])
        if cached.empty:
            return False
        cache_start = cached.index.min()
        cache_end = cached.index.max()
        return (cache_start <= start) and (cache_end >= end)
    except Exception as exc:
        logger.warning(f"Cache read failed for {path}: {exc}")
        return False


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to Parquet cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy")
    logger.debug(f"Cached → {path}")


def _load_cache(path: Path) -> pd.DataFrame:
    """Load a DataFrame from Parquet cache."""
    df = pd.read_parquet(path)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


# ─── Single Symbol Fetch ──────────────────────────────────────────────────────

def fetch_symbol(
    symbol: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    cache_dir: Path,
    force_refresh: bool = False,
    retry_count: int = 3,
    retry_delay: float = 2.0,
) -> pd.DataFrame | None:
    """Fetch OHLCV data for a single symbol.

    Uses local Parquet cache to avoid redundant network calls.
    Returns None if the symbol is unavailable after retries.

    Args:
        symbol: Ticker with exchange suffix (e.g. 'TCS.NS').
        start: Start date string or Timestamp.
        end: End date string or Timestamp.
        cache_dir: Directory to read/write Parquet cache.
        force_refresh: If True, bypass cache and re-fetch from yfinance.
        retry_count: Number of retry attempts on network failure.
        retry_delay: Seconds to wait between retries.

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume, Adj Close]
        or None if fetching failed.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    cache_file = _cache_path(cache_dir, symbol)

    # ── Try cache first ──
    if not force_refresh and _is_cache_valid(cache_file, start_ts, end_ts):
        logger.debug(f"Cache hit: {symbol}")
        df = _load_cache(cache_file)
        # Slice to requested range
        return df.loc[start_ts:end_ts]

    # ── Fetch from yfinance ──
    logger.debug(f"Fetching from yfinance: {symbol} [{start} → {end}]")
    last_exc: Exception | None = None

    for attempt in range(1, retry_count + 1):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=str(start_ts.date()),
                end=str((end_ts + pd.Timedelta(days=1)).date()),
                auto_adjust=False,   # Keep both adjusted and unadjusted
                actions=False,       # We don't use dividends/splits directly
            )

            if df is None or df.empty:
                logger.warning(f"No data returned for {symbol} (attempt {attempt})")
                if attempt < retry_count:
                    time.sleep(retry_delay)
                continue

            # yfinance column normalisation
            df = df.rename(columns={"Adj Close": "Adj Close"})
            if "Adj Close" not in df.columns and "Close" in df.columns:
                logger.warning(
                    f"{symbol}: 'Adj Close' not available — using 'Close' as proxy"
                )
                df["Adj Close"] = df["Close"]

            # Strip timezone
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            _save_cache(df, cache_file)
            return df.loc[start_ts:end_ts]

        except Exception as exc:
            last_exc = exc
            logger.warning(
                f"{symbol}: Fetch error (attempt {attempt}/{retry_count}): {exc}"
            )
            if attempt < retry_count:
                time.sleep(retry_delay)

    logger.error(f"Failed to fetch {symbol} after {retry_count} attempts: {last_exc}")
    return None


# ─── Panel Fetch ──────────────────────────────────────────────────────────────

def fetch_panel(
    universe: Universe,
    cache_dir: str | Path = "data_cache",
    force_refresh: bool = False,
    max_fill_days: int = 3,
    min_rows: int = 252,
    batch_delay: float = 0.5,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Fetch and validate OHLCV data for all universe symbols.

    Also fetches the benchmark index separately.

    Args:
        universe: Universe object with symbols and date range.
        cache_dir: Root directory for Parquet cache.
        force_refresh: Force re-download from yfinance.
        max_fill_days: Max consecutive NaN days to forward-fill.
        min_rows: Minimum valid rows to keep a symbol.
        batch_delay: Seconds to sleep between symbol fetches (rate-limiting).

    Returns:
        Tuple of:
            - panel: Dict of symbol → validated OHLCV DataFrame.
            - benchmark_df: Benchmark OHLCV DataFrame.
    """
    cache_dir = Path(cache_dir)
    start = str(universe.start_date.date())
    end = str(universe.end_date.date())

    raw_panel: dict[str, pd.DataFrame] = {}
    failed_symbols: list[str] = []

    # ── Fetch all symbols ──
    logger.info(f"Fetching {len(universe.symbols)} symbols from yfinance...")
    for symbol in tqdm(universe.symbols, desc="Downloading market data"):
        df = fetch_symbol(
            symbol=symbol,
            start=start,
            end=end,
            cache_dir=cache_dir,
            force_refresh=force_refresh,
        )
        if df is not None and not df.empty:
            raw_panel[symbol] = df
        else:
            failed_symbols.append(symbol)
        time.sleep(batch_delay)

    if failed_symbols:
        logger.warning(
            f"{len(failed_symbols)} symbols failed to fetch: {failed_symbols}"
        )

    # ── Validate all data ──
    logger.info("Running data validation on fetched panel...")
    clean_panel, reports = validate_panel(
        raw_panel,
        max_fill_days=max_fill_days,
        min_rows=min_rows,
    )

    # Update universe with only successfully fetched + validated symbols
    universe.set_active_symbols(list(clean_panel.keys()))

    # ── Fetch benchmark ──
    logger.info(f"Fetching benchmark: {universe.benchmark}")
    benchmark_df = fetch_symbol(
        symbol=universe.benchmark,
        start=start,
        end=end,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
    )
    if benchmark_df is None:
        logger.error(f"Benchmark {universe.benchmark} could not be fetched!")
        benchmark_df = pd.DataFrame()

    logger.info(
        f"Data layer ready | "
        f"Active symbols: {len(clean_panel)} | "
        f"Failed: {len(failed_symbols)} | "
        f"Benchmark rows: {len(benchmark_df)}"
    )
    return clean_panel, benchmark_df


# ─── Panel to MultiIndex DataFrame ───────────────────────────────────────────

def panel_to_wide(
    panel: dict[str, pd.DataFrame],
    column: str = "Adj Close",
) -> pd.DataFrame:
    """Convert a panel dict to a wide-format DataFrame.

    Args:
        panel: Dict of symbol → OHLCV DataFrame.
        column: Column to extract (e.g. 'Adj Close', 'Volume', 'Close').

    Returns:
        Wide DataFrame with DatetimeIndex and symbols as columns.
    """
    frames = {symbol: df[column] for symbol, df in panel.items() if column in df.columns}
    wide = pd.DataFrame(frames)
    wide.index = pd.DatetimeIndex(wide.index)
    wide = wide.sort_index()
    logger.debug(
        f"Wide panel '{column}': shape={wide.shape}, "
        f"NaN fraction={wide.isna().mean().mean():.3f}"
    )
    return wide


__all__ = ["fetch_symbol", "fetch_panel", "panel_to_wide"]
