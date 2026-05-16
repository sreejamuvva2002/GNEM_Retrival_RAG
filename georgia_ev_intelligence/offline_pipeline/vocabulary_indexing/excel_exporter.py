"""Export vocabulary index to Excel for reference."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .models import VocabularyTerm

logger = logging.getLogger(__name__)


def export_to_excel(
    terms: list[VocabularyTerm], output_path: Path | str
) -> Path:
    """Export vocabulary terms to an Excel file.

    Includes readable fields. Excludes full vectors (not useful in a spreadsheet).
    Adds has_vector, token_count, and row_count columns.

    Parameters
    ----------
    terms : list[VocabularyTerm]
        Vocabulary terms to export.
    output_path : Path or str
        Full path for the output Excel file.

    Returns
    -------
    Path
        The resolved output path.
    """
    rows = [
        {
            "normalized_value": t.normalized_value,
            "term_frequency": t.term_frequency,
            "row_ids": str(t.row_ids),
            "multiple_words": t.multiple_words,
            "term_type": t.term_type,
            "source_column": t.source_column,
            "has_vector": t.term_vector is not None,
            "token_count": len(t.normalized_value.split()),
            "row_count": len(t.row_ids),
        }
        for t in terms
    ]

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["term_type", "term_frequency"], ascending=[True, False]
    ).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)

    logger.info("Exported %d vocabulary terms to %s", len(terms), output_path)
    return output_path
