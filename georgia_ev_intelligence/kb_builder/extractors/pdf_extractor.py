"""PDF → clean body text extractor using pdfplumber."""
from __future__ import annotations

import io


def extract(pdf_bytes: bytes) -> tuple[str, str]:
    """Extract (title, body_text) from raw PDF bytes.

    Title is taken from the PDF metadata if available; body is concatenated
    page text with double-newline page breaks.

    Returns ("", "") on failure.
    """
    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pdfplumber is required: pip install pdfplumber>=0.11"
        ) from exc

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            # Extract metadata title
            meta = pdf.metadata or {}
            title = str(meta.get("Title") or meta.get("title") or "").strip()

            pages: list[str] = []
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                text = text.strip()
                if text:
                    pages.append(text)

            body = "\n\n".join(pages).strip()
            return title, body

    except Exception:
        return "", ""
