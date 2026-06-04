"""DOCX → Markdown (README §14).

Re-parses the document with python-docx to preserve heading hierarchy, paragraphs,
lists, and tables (rendered as Markdown tables). Falls back to the plain
``docx_extractor`` if structured parsing fails.
"""
from __future__ import annotations

import io

from georgia_ev_intelligence.kb_builder.extractors import docx_extractor

from .base import BaseConverter, ConversionResult, rows_to_markdown_table


def _heading_level(style_name: str) -> int | None:
    """Map a paragraph style name to a Markdown heading level (1-6), or None."""
    name = (style_name or "").lower()
    if name.startswith("title"):
        return 1
    if name.startswith("heading"):
        digits = "".join(ch for ch in name if ch.isdigit())
        if digits:
            return min(int(digits) + 1, 6)  # Heading 1 -> ##, keep top-level # for title
        return 2
    return None


class DocxConverter(BaseConverter):
    extraction_tool = "python-docx"

    def convert(self, raw_bytes: bytes, source_name: str) -> ConversionResult:
        heading = source_name.rsplit("/", 1)[-1]
        warnings: list[str] = []
        try:
            from docx import Document  # type: ignore
        except ImportError:
            warnings.append("python-docx unavailable, fell back to text extractor")
            title, body = docx_extractor.extract(raw_bytes)
            return ConversionResult(
                markdown_body=f"# {title or heading}\n\n{body}",
                title=title or heading,
                warnings=warnings,
                quality_status="needs_review",
            )

        try:
            doc = Document(io.BytesIO(raw_bytes))
        except Exception as exc:
            return ConversionResult(
                markdown_body=f"# {heading}",
                title=heading,
                warnings=[f"DOCX parse failed: {exc}"],
                quality_status="failed",
            )

        title = ""
        lines: list[str] = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            style_name = para.style.name if para.style else ""
            level = _heading_level(style_name)
            if level == 1 and not title:
                title = text
                lines.append(f"# {text}")
            elif level:
                lines.append(f"{'#' * level} {text}")
            elif "list" in (style_name or "").lower():
                lines.append(f"- {text}")
            else:
                lines.append(text)
            lines.append("")

        n_tables = 0
        if doc.tables:
            lines.append("## Tables")
            lines.append("")
            for i, table in enumerate(doc.tables, start=1):
                rows = [[cell.text for cell in row.cells] for row in table.rows]
                md_table = rows_to_markdown_table(rows)
                if md_table:
                    n_tables += 1
                    lines.append(f"### Table {i}")
                    lines.append("")
                    lines.append(md_table)
                    lines.append("")

        if not title:
            title = heading

        body = "\n".join(lines).strip()
        if not body:
            warnings.append("No extractable content in DOCX")

        return ConversionResult(
            markdown_body=body or f"# {title}",
            title=title,
            metadata={"num_tables": n_tables},
            warnings=warnings,
        )
