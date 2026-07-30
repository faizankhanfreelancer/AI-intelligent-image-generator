"""
utils/logger.py

Application-wide logger. Uses Rich for readable, colorized console output
and a rotating file handler so history/logs/ retains a durable record of
startup events, generation requests, durations, and errors.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from rich.logging import RichHandler

from config import settings


def get_logger(name: str = "image_studio") -> logging.Logger:
    """Return a configured logger, creating handlers only once."""

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(settings.log_level.upper())
    logger.propagate = False

    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
        markup=True,
    )
    console_handler.setLevel(settings.log_level.upper())

    log_file = settings.paths.logs_dir / "app.log"
    file_handler = RotatingFileHandler(
        log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = get_logger()
