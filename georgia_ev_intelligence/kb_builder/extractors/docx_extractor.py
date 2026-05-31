"""DOCX → clean body text extractor using python-docx."""
from __future__ import annotations

import io


def extract(docx_bytes: bytes) -> tuple[str, str]:
    """Extract (title, body_text) from raw DOCX bytes.

    Title is the first non-empty paragraph styled as a heading or the first
    paragraph overall.  Body is every paragraph's text joined with newlines.

    Returns ("", "") on failure.
    """
    try:
        from docx import Document  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "python-docx is required: pip install python-docx>=1.1"
        ) from exc

    try:
        doc = Document(io.BytesIO(docx_bytes))

        title = ""
        paragraphs: list[str] = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            # Use the first heading as the title
            if not title and para.style and "heading" in para.style.name.lower():
                title = text
            paragraphs.append(text)

        # If no heading found, use first paragraph as title
        if not title and paragraphs:
            title = paragraphs[0]

        body = "\n".join(paragraphs).strip()
        return title, body

    except Exception:
        return "", ""
