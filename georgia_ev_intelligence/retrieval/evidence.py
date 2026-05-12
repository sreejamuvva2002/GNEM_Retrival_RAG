"""
Formats retrieved rows into evidence strings passed to the synthesizer.
"""
from __future__ import annotations
import pandas as pd
from .. import config


def format_evidence(rows: pd.DataFrame) -> list[str]:
    """Convert DataFrame rows into human-readable evidence strings."""
    skip_cols = {"_row_id", "latitude", "longitude", "address"}
    evidence = []
    for _, row in rows.head(config.MAX_EVIDENCE_ROWS).iterrows():
        parts = []
        for col, val in row.items():
            if col in skip_cols or pd.isna(val):
                continue
            label = col.replace("_", " ").title()
            parts.append(f"{label}: {val}")
        evidence.append(" | ".join(parts))
    return evidence


def select(rows: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (top rows, formatted evidence strings)."""
    evidence_strings = format_evidence(rows)
    return rows, evidence_strings
