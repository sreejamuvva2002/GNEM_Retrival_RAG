"""Display helpers for vocabulary and schema labels."""
from __future__ import annotations

import re

from .constants import SOURCE_COLUMN_DISPLAY, _CANONICAL_ACRONYMS


def display_source_column(source_column: str) -> str:
    return SOURCE_COLUMN_DISPLAY.get(source_column, source_column.replace("_", " ").title())


def to_canonical_value(normalized: str) -> str:
    """Convert a normalized vocabulary value to a display canonical form."""
    if not normalized:
        return ""
    parts = normalized.split()
    out: list[str] = []
    for part in parts:
        if part.lower() in _CANONICAL_ACRONYMS:
            out.append(part.upper() if len(part) <= 3 else part.title())
        elif re.fullmatch(r"[a-z]+/\w+", part, re.IGNORECASE):
            left, _, right = part.partition("/")
            out.append(f"{left.title()}/{right}")
        elif re.fullmatch(r"\d+/\d+", part):
            out.append(part)
        else:
            out.append(part.title())
    return " ".join(out)
