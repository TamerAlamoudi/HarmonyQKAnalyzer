# utils/logging_utils.py
"""
Utility functions for logging configuration.
"""
import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Logger name
        log_file: Path to log file. If None, no file logging will be set up.
        level: Logging level
        log_to_console: Whether to log to console

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add file handler if log_file is provided
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_timestamp() -> str:
    """Get current timestamp in a filename-friendly format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
