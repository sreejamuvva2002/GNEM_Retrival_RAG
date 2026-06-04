"""PDF → Markdown (README §12).

Re-opens the PDF with pdfplumber to preserve page boundaries (``## Page N`` markers)
and extract per-page tables. When a page yields little/no text, attempts OCR via
PyMuPDF + pytesseract if available; otherwise flags ``requires_ocr`` / ``needs_review``.
"""
from __future__ import annotations

import io
import logging

from georgia_ev_intelligence.kb_builder.extractors import pdf_extractor

from .base import BaseConverter, ConversionResult, rows_to_markdown_table

logger = logging.getLogger(__name__)

# A page with fewer than this many characters is considered "empty" (candidate for OCR).
_MIN_PAGE_CHARS = 20


def _ocr_page(pdf_bytes: bytes, page_index: int) -> str:
    """Render one page to an image and OCR it. Returns "" if tooling unavailable."""
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img).strip()
    except Exception as exc:
        logger.debug("OCR failed on page %d: %s", page_index, exc)
        return ""


class PdfConverter(BaseConverter):
    extraction_tool = "pdfplumber"

    def convert(self, raw_bytes: bytes, source_name: str) -> ConversionResult:
        heading = source_name.rsplit("/", 1)[-1]
        warnings: list[str] = []
        try:
            import pdfplumber  # type: ignore
        except ImportError:
            warnings.append("pdfplumber unavailable, fell back to text extractor")
            title, body = pdf_extractor.extract(raw_bytes)
            return ConversionResult(
                markdown_body=f"# {title or heading}\n\n{body}",
                title=title or heading,
                warnings=warnings,
                quality_status="needs_review",
            )

        try:
            pdf = pdfplumber.open(io.BytesIO(raw_bytes))
        except Exception as exc:
            return ConversionResult(
                markdown_body=f"# {heading}",
                title=heading,
                warnings=[f"PDF open failed: {exc}"],
                quality_status="failed",
            )

        title = ""
        page_sections: list[str] = []
        table_sections: list[str] = []
        empty_pages = 0
        ocr_pages = 0
        requires_ocr = False

        with pdf:
            meta = pdf.metadata or {}
            title = str(meta.get("Title") or meta.get("title") or "").strip()
            total_pages = len(pdf.pages)

            for idx, page in enumerate(pdf.pages):
                text = (page.extract_text(x_tolerance=2, y_tolerance=2) or "").strip()
                if len(text) < _MIN_PAGE_CHARS:
                    ocr_text = _ocr_page(raw_bytes, idx)
                    if ocr_text:
                        text = ocr_text
                        ocr_pages += 1
                        requires_ocr = True
                    else:
                        empty_pages += 1
                        requires_ocr = True

                page_sections.append(f"## Page {idx + 1}\n\n{text or '_(no extractable text)_'}")

                # Tables on this page.
                try:
                    tables = page.extract_tables() or []
                except Exception:
                    tables = []
                for t_i, table in enumerate(tables, start=1):
                    md_table = rows_to_markdown_table([list(r) for r in table])
                    if md_table:
                        table_sections.append(
                            f"### Table {len(table_sections) + 1} — Page {idx + 1}\n\n{md_table}"
                        )

        if not title:
            title = heading

        parts = [f"# {title}", ""]
        parts.extend(s + "\n" for s in page_sections)
        if table_sections:
            parts.append("## Extracted Tables\n")
            parts.extend(s + "\n" for s in table_sections)

        body_text_len = sum(len(s) for s in page_sections)
        quality_status = "pass"
        if total_pages and empty_pages >= total_pages:
            quality_status = "failed"
            warnings.append("All pages empty after extraction/OCR")
        elif empty_pages:
            quality_status = "needs_review"
            warnings.append(f"{empty_pages}/{total_pages} pages had no extractable text")
        if ocr_pages:
            warnings.append(f"OCR used on {ocr_pages} page(s)")

        return ConversionResult(
            markdown_body="\n".join(parts).strip(),
            title=title,
            metadata={
                "page_count": total_pages,
                "empty_pages": empty_pages,
                "ocr_pages": ocr_pages,
                "num_tables": len(table_sections),
                "text_length": body_text_len,
            },
            warnings=warnings,
            requires_ocr=requires_ocr,
            quality_status=quality_status,
        )
