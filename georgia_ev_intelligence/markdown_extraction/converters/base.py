"""Converter interface (README §39) shared by every file type.

Each converter wraps the matching ``kb_builder.extractors`` extractor for parsing and
returns a ``ConversionResult`` holding the Markdown *body only* — the YAML front matter
is added later by ``metadata.build_markdown``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversionResult:
    """Output of a converter — body content plus extraction signals."""

    markdown_body: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    requires_ocr: bool = False
    quality_status: str = "pass"


class BaseConverter:
    """All converters implement ``convert(raw_bytes, source_name) -> ConversionResult``."""

    # Human-readable tool name recorded in metadata / manifests.
    extraction_tool: str = "base"

    def convert(self, raw_bytes: bytes, source_name: str) -> ConversionResult:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Markdown rendering helpers shared across converters
# --------------------------------------------------------------------------- #

def _cell(value: Any) -> str:
    """Render a single table cell: stringify, escape pipes, collapse newlines."""
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip()


def rows_to_markdown_table(rows: list[list[Any]], header: list[Any] | None = None) -> str:
    """Render rows (and optional header) as a GitHub-flavored Markdown table.

    If ``header`` is None, the first row is used as the header.
    Returns "" when there is nothing to render.
    """
    if header is None:
        if not rows:
            return ""
        header, body_rows = rows[0], rows[1:]
    else:
        body_rows = rows

    if not header:
        return ""

    ncols = len(header)
    head_line = "| " + " | ".join(_cell(c) for c in header) + " |"
    sep_line = "| " + " | ".join("---" for _ in range(ncols)) + " |"
    lines = [head_line, sep_line]
    for row in body_rows:
        cells = [_cell(c) for c in row[:ncols]]
        cells += [""] * (ncols - len(cells))  # pad short rows
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def dataframe_to_markdown_table(df, max_rows: int | None = None) -> str:
    """Render a pandas DataFrame as a Markdown table (header + rows)."""
    header = [str(c) for c in df.columns]
    truncated = False
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
        truncated = True
    rows = df.astype(object).where(df.notna(), "").values.tolist()
    table = rows_to_markdown_table(rows, header=header)
    if truncated:
        table += f"\n\n_… table truncated to first {max_rows} rows._"
    return table
