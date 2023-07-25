"""Logging setup."""

import logging


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    for level in logging_levels:
        setattr(logger, level, getattr(logger, level))

    return logger
