"""
Structured logging configuration using loguru.

All modules should import the logger from here to maintain consistent
formatting across the entire system.
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_dir: str | Path = "logs", level: str = "INFO") -> None:
    """Configure loguru with console and file sinks.

    Args:
        log_dir: Directory to write log files into.
        level: Minimum log level. Use "DEBUG" for verbose output.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default sink
    logger.remove()

    # Console sink — human-readable
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        level=level,
        colorize=True,
    )

    # File sink — structured for debugging
    logger.add(
        log_path / "quant_engine.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )


# Initialise with defaults — callers can override via setup_logger()
setup_logger()

__all__ = ["logger", "setup_logger"]
