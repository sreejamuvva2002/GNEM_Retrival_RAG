"""Light search-safe normalization for vocabulary terms.

This normalizer applies minimal transformations on top of the already-normalized
values from loader.py. It does NOT duplicate loader normalization.

Transformations:
- lowercase
- trim leading/trailing whitespace
- collapse repeated internal whitespace
- preserve meaningful symbols: /, -, &
"""
from __future__ import annotations

import re


def normalize_term(value: str) -> str:
    """Apply light search-safe normalization to a term value.

    The input is already normalized by loader.py. This adds only:
    - lowercase
    - trim spaces
    - collapse repeated whitespace
    """
    if not value:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def is_multiple_words(normalized: str) -> bool:
    """Determine if a normalized term contains multiple meaningful tokens.

    Returns True for:
    - "battery cell" (space-separated)
    - "tier 1/2" (slash with digits = phrase-like)
    - "original equipment manufacturer" (multi-word phrase)

    Returns False for:
    - "hyundai" (single token)
    - "yes" (single token)
    """
    if not normalized:
        return False
    # Contains a space -> multiple words
    if " " in normalized.strip():
        return True
    # Pattern like "tier1/2" or "1/2" where slash connects meaningful parts
    if re.search(r"\w/\w", normalized):
        return True
    return False
