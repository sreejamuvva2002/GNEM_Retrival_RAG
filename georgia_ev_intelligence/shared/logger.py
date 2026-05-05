"""
Shared logger. Every module uses this — consistent format across all phases.
Pattern adopted from ev_data_LLM_comparsions/src/utils/logger.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_FMT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_root_configured = False


def _configure_root() -> None:
    global _root_configured
    if _root_configured:
        return
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler — INFO and above
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_FMT, _DATE_FMT))
    root.addHandler(console)

    # File handler — DEBUG and above (full detail for debugging)
    log_file = _LOG_DIR / "georgia_ev.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FMT, _DATE_FMT))
    root.addHandler(file_handler)

    _root_configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a named logger. Call once per module at module level."""
    _configure_root()
    return logging.getLogger(name)
