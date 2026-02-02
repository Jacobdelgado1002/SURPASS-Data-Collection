"""
Centralized logging configuration for the project.

This module provides a reusable logger factory that supports:
    - Consistent formatting
    - Project-wide root logger namespace
    - Per-module log levels
    - Optional file logging
    - Safe repeated imports (no duplicate handlers)

Example:
    from logger_config import get_logger
    import logging

    logger = get_logger(__name__)
    logger.info("Starting pipeline")

    debug_logger = get_logger(__name__, level=logging.DEBUG)

    file_logger = get_logger(__name__, log_file="run.log")


If this file is used inside a script, no special CLI steps are required.
Logging initializes automatically on first import.
"""
import logging
from pathlib import Path
from typing import Optional, Union


# ---------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------

LOG_FORMAT: str = (
    "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(name)s - %(message)s"
)
DEFAULT_LEVEL: int = logging.INFO
ROOT_LOGGER_NAME: str = "data_collection"


# ---------------------------------------------------------------------
# Logger configuration
# ---------------------------------------------------------------------


def _configure_root_logger(level: int = DEFAULT_LEVEL) -> logging.Logger:
    """
    Configure and return the project-wide root logger.

    This function ensures that:
        - Logging configuration is applied exactly once
        - Duplicate handlers are not added during repeated imports
        - All child loggers inherit the same formatting

    Args:
        level: Logging severity level (e.g., logging.INFO, logging.DEBUG).
            Must be a valid logging level constant.

    Returns:
        Configured root logger instance for the project namespace.

    Notes:
        - Should only be called internally by this module
        - Idempotent: safe to call multiple times (checks for existing handlers)
        - Uses StreamHandler for console output to stderr
        - Sets propagate=False to prevent double logging through root logger

    Examples:
        >>> root = _configure_root_logger(logging.DEBUG)
        >>> root.name
        'data_collection'
        >>> root.level
        10  # DEBUG level
    """
    logger: logging.Logger = logging.getLogger(ROOT_LOGGER_NAME)

    # Prevent duplicate handlers if module is imported multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler: logging.StreamHandler = logging.StreamHandler()
    formatter: logging.Formatter = logging.Formatter(LOG_FORMAT)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False  # Avoid double logging through root logger

    return logger


# Initialize once at import time
_root_logger: logging.Logger = _configure_root_logger()


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def get_logger(
    name: Optional[str] = None,
    *,
    level: Optional[int] = None,
    log_file: Optional[Union[str, Path]] = None,  # ← IMPROVED: Added Union type hint
) -> logging.Logger:
    """
    Create or retrieve a project logger with optional customization.

    This function should be the only way modules obtain loggers. It provides
    a consistent interface for creating loggers with optional level overrides
    and file output.

    Args:
        name: Logger name. Pass `__name__` for per-module loggers.
            If None, returns the project root logger.
            Example: get_logger(__name__) creates "data_collection.my_module"

        level: Optional log level override for this specific logger
            (e.g., logging.DEBUG, logging.INFO). If None, inherits root level.
            Does not affect parent or sibling loggers.

        log_file: Optional path to a file where logs should also be written.
            Accepts either string path or pathlib.Path object.
            Adds a FileHandler only once per file to avoid duplicates.
            Creates parent directories if they don't exist.

    Returns:
        Configured logger instance. The logger will:
            - Inherit project-wide formatting
            - Use specified or default log level
            - Output to console (always)
            - Output to file (if log_file specified)

    Raises:
        OSError: If log_file directory cannot be created or file cannot be written.

    Behavior:
        - Child loggers inherit the project formatter
        - Level overrides affect only this logger (not parent/children)
        - File handlers are attached only once per unique path
        - Safe for repeated calls (idempotent for same arguments)

    Examples:
        # Basic usage - inherits INFO level
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")

        # Debug level for specific module
        >>> logger = get_logger(__name__, level=logging.DEBUG)
        >>> logger.debug("Detailed information")

        # Log to file in addition to console
        >>> logger = get_logger(__name__, log_file="train.log")
        >>> logger.info("This goes to both console and train.log")

        # Combine level override and file logging
        >>> logger = get_logger(__name__, level=logging.DEBUG, log_file="debug.log")

    Notes:
        - Multiple calls with same parameters return the same logger instance
        - File handlers use the same format as console handlers
        - File paths are resolved to absolute paths for duplicate detection
        - If a file handler already exists for a path, it won't be added again
    """
    logger: logging.Logger

    if name is None:
        logger = _root_logger
    else:
        logger = _root_logger.getChild(name)

    # -------------------------------------------------------------
    # Optional per-logger level override
    # -------------------------------------------------------------
    if level is not None:
        logger.setLevel(level)

    # -------------------------------------------------------------
    # Optional file logging
    # -------------------------------------------------------------
    if log_file is not None:
        log_path: Path = Path(log_file)

        # Avoid attaching duplicate file handlers
        # Check if this exact file path already has a handler
        existing_files = {
            getattr(handler, "baseFilename", None)
            for handler in logger.handlers
            if isinstance(handler, logging.FileHandler)
        }

        # Resolve to absolute path for accurate comparison
        if str(log_path.resolve()) not in existing_files:
            # Create parent directories if they don't exist
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Create and configure file handler
            formatter: logging.Formatter = logging.Formatter(LOG_FORMAT)
            file_handler: logging.FileHandler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger