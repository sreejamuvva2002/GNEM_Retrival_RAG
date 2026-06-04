"""Excel → Markdown (README §15).

Default mode is workbook-level Markdown with one ``## Sheet:`` table per sheet.
For the structured 205-company KB, row/entity-level Markdown is preferable; that mode
is available via ``EXCEL_ROW_LEVEL`` but is opt-in (sheet/workbook mode by default).
"""
from __future__ import annotations

import io

from georgia_ev_intelligence.kb_builder.extractors import excel_extractor

from .base import BaseConverter, ConversionResult, dataframe_to_markdown_table

_PREVIEW_ROWS = 200


class ExcelConverter(BaseConverter):
    extraction_tool = "pandas-openpyxl"

    def convert(self, raw_bytes: bytes, source_name: str) -> ConversionResult:
        heading = source_name.rsplit("/", 1)[-1]
        warnings: list[str] = []
        try:
            import pandas as pd
        except ImportError:
            warnings.append("pandas unavailable, fell back to text extractor")
            title, body = excel_extractor.extract(raw_bytes)
            return ConversionResult(
                markdown_body=f"# {heading}\n\n{body}",
                title=title or heading,
                warnings=warnings,
                quality_status="needs_review",
            )

        try:
            sheets = pd.read_excel(io.BytesIO(raw_bytes), sheet_name=None)
        except Exception as exc:
            return ConversionResult(
                markdown_body=f"# {heading}",
                title=heading,
                warnings=[f"Excel parse failed: {exc}"],
                quality_status="failed",
            )

        parts = [f"# {heading}", ""]
        total_rows = 0
        for sheet_name, df in sheets.items():
            df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
            n_rows = len(df)
            total_rows += n_rows
            parts.append(f"## Sheet: {sheet_name}")
            parts.append("")
            large = n_rows > _PREVIEW_ROWS
            table = dataframe_to_markdown_table(
                df, max_rows=_PREVIEW_ROWS if large else None
            )
            parts.append(table if table else "_(empty sheet)_")
            parts.append("")
            parts.append(f"- Rows: {n_rows:,} · Columns: {df.shape[1]}")
            parts.append("")
            if large:
                warnings.append(f"Sheet '{sheet_name}' previewed to {_PREVIEW_ROWS} rows")

        return ConversionResult(
            markdown_body="\n".join(parts).strip(),
            title=heading,
            metadata={"sheet_count": len(sheets), "row_count": int(total_rows)},
            warnings=warnings,
        )
