"""
Application-wide logging utility configured with standardized formatting.
"""

import logging
import sys
from config import LOG_LEVEL

def setup_logger(name: str = "DiabetesAdaptiveRAG") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = getattr(logging, LOG_LEVEL, logging.INFO)
        logger.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | [%(name)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger