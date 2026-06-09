"""Text & value normalization for routing.

Two layers:

* **Question-level** — ``normalize_question`` (README §10) prepares the user
  question and extracts quoted phrases (treated as intentional, exact text).
* **Value-level** — ``normalize_for_field`` routes a user-supplied value through
  the *same* loader helper that produced the stored column, so resolution
  compares like-for-like against ``ColumnMeta.unique_values``.

The per-column dispatch is correct *by construction*: it mirrors
``loader.normalize_dataframe`` — any column not listed falls back to
``clean_text``, exactly as the loader does. This is why slash handling is safe
even though normalization is column-specific and asymmetric:

    category               "Tier 2/3"               -> "Tier 2/3"   (slash kept, no space)
    primary_facility_type  "Engineering/Manufacturing" -> "Engineering/ Manufacturing"
    primary_oems           "Hyundai/Kia"            -> "Hyundai/ Kia"
    product_service        "A/B"                    -> "A / B"
"""
from __future__ import annotations

import re

from georgia_ev_intelligence.shared.data import loader as L

# Map real KB column -> the exact loader normalizer used to build its stored
# values. Mirrors loader.normalize_dataframe; unlisted columns fall back to
# clean_text (matching the loader's own fallback).
COLUMN_NORMALIZERS = {
    "company": L.clean_company,
    "category": L.clean_category,                            # "Tier 2/3" -> "Tier 2/3"
    "primary_facility_type": L.clean_primary_facility_type,  # "A/B" -> "A/ B"
    "primary_oems": L.normalize_separators,                  # "A/B" -> "A/ B"
    "oem_footprint": L.clean_oem_footprint,
    "product_service": L.clean_product_service,              # "A/B" -> "A / B"
    "updated_location": L.clean_missing_only,                # "City, County" verbatim
}


def normalize_for_field(field: str, value) -> str:
    """Normalize ``value`` the way column ``field`` was normalized at load time.

    Callers must pass a scalar (str/number), not a list — list inputs are split
    upstream in value resolution.
    """
    normalizer = COLUMN_NORMALIZERS.get(field, L.clean_text)
    return str(normalizer(value))


_QUOTED = re.compile(r'"([^"]+)"')


def extract_quoted_phrases(text: str) -> list[str]:
    """Return quoted substrings, preserved exactly (README §10)."""
    return [m.strip() for m in _QUOTED.findall(text or "") if m.strip()]


def normalize_question(question: str) -> dict:
    """Prepare a question for routing (README §10).

    Returns the ``original``, a whitespace-collapsed ``normalized`` form, a
    ``lowercase`` form, and any ``quoted_phrases`` (intentional exact text the
    validator should prioritise during value resolution).
    """
    original = (question or "").strip()
    normalized = re.sub(r"\s+", " ", original).strip()
    return {
        "original": original,
        "normalized": normalized,
        "lowercase": normalized.lower(),
        "quoted_phrases": extract_quoted_phrases(original),
    }
