"""CSV/TSV → Markdown (README §16).

Parses the CSV into a table preview and reports row/column counts. Very large CSVs
are previewed (not fully inlined) with a note recommending structured handling.
"""
from __future__ import annotations

import io

from georgia_ev_intelligence.kb_builder.extractors import csv_extractor

from .base import BaseConverter, ConversionResult, dataframe_to_markdown_table

# Rows beyond this are previewed rather than fully rendered.
_PREVIEW_ROWS = 200


class CsvConverter(BaseConverter):
    extraction_tool = "pandas-csv"

    def convert(self, raw_bytes: bytes, source_name: str) -> ConversionResult:
        heading = source_name.rsplit("/", 1)[-1]
        warnings: list[str] = []
        try:
            import pandas as pd
        except ImportError:
            warnings.append("pandas unavailable, fell back to text extractor")
            _, body = csv_extractor.extract(raw_bytes)
            return ConversionResult(
                markdown_body=f"# {heading}\n\n```\n{body}\n```",
                title=heading,
                warnings=warnings,
                quality_status="needs_review",
            )

        # Decode with encoding fallback.
        text = None
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                text = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            return ConversionResult(
                markdown_body=f"# {heading}",
                title=heading,
                warnings=["Could not decode CSV bytes"],
                quality_status="failed",
            )

        sep = "\t" if source_name.lower().endswith(".tsv") else None
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
        except Exception as exc:
            warnings.append(f"CSV parse failed: {exc}")
            return ConversionResult(
                markdown_body=f"# {heading}\n\n```\n{text.strip()}\n```",
                title=heading,
                warnings=warnings,
                quality_status="needs_review",
            )

        # Drop fully-empty rows/columns.
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        n_rows, n_cols = df.shape

        large = n_rows > _PREVIEW_ROWS
        table = dataframe_to_markdown_table(df, max_rows=_PREVIEW_ROWS if large else None)

        parts = [f"# {heading}", "", "## Table Preview", "", table, "", "## Notes", ""]
        parts.append(f"- Total rows: {n_rows:,}")
        parts.append(f"- Total columns: {n_cols}")
        if large:
            parts.append(
                f"- Large CSV: only the first {_PREVIEW_ROWS} rows are previewed above; "
                "treat the full file as structured database input."
            )
            warnings.append("Large CSV previewed, not fully inlined")

        return ConversionResult(
            markdown_body="\n".join(parts),
            title=heading,
            metadata={"row_count": int(n_rows), "column_count": int(n_cols)},
            warnings=warnings,
        )
