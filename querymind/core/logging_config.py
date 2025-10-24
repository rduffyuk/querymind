"""
QueryMind Logging Configuration

Provides centralized logging setup for all QueryMind components.
Supports environment-based configuration and structured formatting.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Configure logging for QueryMind components.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                   Defaults to environment variable LOG_LEVEL or INFO
        log_file: Optional log file path. If provided, logs to both console and file
        log_to_console: Whether to log to console (default: True)

    Returns:
        Configured root logger

    Example:
        >>> logger = setup_logging(log_level="DEBUG", log_file="/tmp/querymind.log")
        >>> logger.info("QueryMind started")
    """
    # Get log level from environment or parameter
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Validate log level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_levels:
        log_level = "INFO"

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Add console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Add file handler if log_file specified
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing query")
    """
    return logging.getLogger(name)


# Initialize logging on module import with sensible defaults
# Can be reconfigured later with setup_logging()
if not logging.getLogger().handlers:
    setup_logging()
