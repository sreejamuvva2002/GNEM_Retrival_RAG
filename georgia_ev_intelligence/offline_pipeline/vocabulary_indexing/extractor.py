"""Extract vocabulary terms from the normalized KB DataFrame."""
from __future__ import annotations

import logging
import re
from collections import defaultdict

import pandas as pd

from .config import (
    COLUMN_TERM_TYPE_MAP,
    MULTI_VALUE_COLUMNS,
    NUMERIC_COLUMNS,
    SEPARATOR_PATTERN,
    SKIP_VALUES,
)
from .models import VocabularyTerm
from .normalizer import is_multiple_words, normalize_term
from .term_classifier import classify

logger = logging.getLogger(__name__)


def extract_terms(df: pd.DataFrame) -> list[VocabularyTerm]:
    """Extract vocabulary terms from the normalized KB DataFrame.

    Iterates over configured columns, extracts and normalizes cell values,
    aggregates frequency and row_ids, and returns deduplicated VocabularyTerm list.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized DataFrame from loader.load(). Must contain '_row_id' column.

    Returns
    -------
    list[VocabularyTerm]
        Deduplicated vocabulary terms with frequency and row_id data.
    """
    if "_row_id" not in df.columns:
        raise ValueError("DataFrame must contain '_row_id' column (from loader.load())")

    # Accumulator: (normalized_value, source_column) -> {frequency, row_ids}
    term_data: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"frequency": 0, "row_ids": set()}
    )

    columns_processed = 0

    for column, term_type in COLUMN_TERM_TYPE_MAP.items():
        if column not in df.columns:
            logger.warning("Column '%s' not found in DataFrame, skipping.", column)
            continue

        columns_processed += 1

        if column in NUMERIC_COLUMNS:
            _extract_numeric_column(df, column, term_data)
        elif column in MULTI_VALUE_COLUMNS:
            _extract_multi_value_column(df, column, term_data)
        else:
            _extract_text_column(df, column, term_data)

    # Build VocabularyTerm objects
    terms: list[VocabularyTerm] = []
    for (normalized_value, source_column), data in term_data.items():
        row_ids = sorted(data["row_ids"])
        terms.append(
            VocabularyTerm(
                normalized_value=normalized_value,
                term_frequency=data["frequency"],
                row_ids=row_ids,
                multiple_words=is_multiple_words(normalized_value),
                term_type=classify(source_column),
                source_column=source_column,
            )
        )

    logger.info(
        "Extracted %d unique terms from %d columns (%d rows).",
        len(terms),
        columns_processed,
        len(df),
    )
    return terms


def _extract_text_column(
    df: pd.DataFrame,
    column: str,
    term_data: dict[tuple[str, str], dict],
) -> None:
    """Extract full cell values from a text column."""
    for _, row in df.iterrows():
        raw_value = row[column]
        normalized = normalize_term(str(raw_value))
        if _should_skip(normalized):
            continue
        row_id = int(row["_row_id"])
        key = (normalized, column)
        term_data[key]["frequency"] += 1
        term_data[key]["row_ids"].add(row_id)


def _extract_multi_value_column(
    df: pd.DataFrame,
    column: str,
    term_data: dict[tuple[str, str], dict],
) -> None:
    """Extract and split multi-value cells (comma, semicolon, pipe, newline)."""
    for _, row in df.iterrows():
        raw_value = str(row[column])
        parts = _split_multi_value(raw_value)
        row_id = int(row["_row_id"])

        for part in parts:
            normalized = normalize_term(part)
            if _should_skip(normalized):
                continue
            key = (normalized, column)
            term_data[key]["frequency"] += 1
            term_data[key]["row_ids"].add(row_id)


def _extract_numeric_column(
    df: pd.DataFrame,
    column: str,
    term_data: dict[tuple[str, str], dict],
) -> None:
    """Create controlled vocabulary entries for numeric columns.

    Instead of indexing every numeric value, creates a single entry
    for the column with row_ids pointing to rows that have valid numeric data.
    """
    valid_row_ids: set[int] = set()
    for _, row in df.iterrows():
        value = str(row[column]).strip().lower()
        if value in SKIP_VALUES:
            continue
        # Check if it's actually numeric
        try:
            float(value)
            valid_row_ids.add(int(row["_row_id"]))
        except (ValueError, TypeError):
            continue

    if valid_row_ids:
        # Use the column name itself as the controlled term
        key = (column, column)
        term_data[key]["frequency"] = len(valid_row_ids)
        term_data[key]["row_ids"] = valid_row_ids


def _split_multi_value(cell_value: str) -> list[str]:
    """Split on comma, semicolon, pipe, newline. NOT on slash.

    Preserves values like "tier 1/2", "ev / battery relevant".
    """
    parts = re.split(SEPARATOR_PATTERN, cell_value)
    return [part.strip() for part in parts if part.strip()]


def _should_skip(value: str) -> bool:
    """Return True if the value should not be indexed."""
    return value.strip().lower() in SKIP_VALUES or not value.strip()
