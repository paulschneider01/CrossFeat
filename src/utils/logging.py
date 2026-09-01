"""Logging utilities for CrossFeat."""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(output_dir: Path, name: str = "train") -> logging.Logger:
    """Setup logging with file and console handlers.

    Args:
        output_dir: Directory to save log files
        name: Logger name

    Returns:
        Configured logger instance
    """
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Include microseconds to avoid filename collisions in fast consecutive calls.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = log_dir / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # Replace existing handlers (and close them) to avoid duplicate logs and FD leaks.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    # File handler (detailed)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(fh)

    # Console handler (info and above)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(ch)

    logger.info(f"Logging to: {log_file}")

    return logger
