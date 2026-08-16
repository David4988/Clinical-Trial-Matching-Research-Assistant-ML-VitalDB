"""Console + file logging for the audit pipeline."""

import logging
import sys

from vitaldb_audit import config

LOGGER_NAME = "vitaldb_audit"
_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure and return the package logger.

    Logs to stderr always, and to results/audit.log so a run leaves a trail.
    Safe to call more than once: existing handlers are cleared first.
    """
    config.ensure_dirs()
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    fh = logging.FileHandler(config.RESULTS_DIR / "audit.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
