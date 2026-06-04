"""Image → Markdown (README §20).

Runs OCR (via the existing ``image_extractor``) and records image dimensions.
Images with no useful text are flagged ``low_value`` (the pipeline may skip them).
Vision description is intentionally a non-default fallback.
"""
from __future__ import annotations

import io
import logging

from georgia_ev_intelligence.kb_builder.extractors import image_extractor

from .base import BaseConverter, ConversionResult

logger = logging.getLogger(__name__)

# OCR text shorter than this means the image carries no useful text.
_MIN_OCR_CHARS = 10


def _dimensions(raw_bytes: bytes) -> tuple[int | None, int | None]:
    try:
        from PIL import Image
        with Image.open(io.BytesIO(raw_bytes)) as img:
            return img.width, img.height
    except Exception:
        return None, None


class ImageConverter(BaseConverter):
    extraction_tool = "pytesseract-ocr"

    def convert(self, raw_bytes: bytes, source_name: str) -> ConversionResult:
        heading = source_name.rsplit("/", 1)[-1]
        warnings: list[str] = []

        _, ocr_text = image_extractor.extract(raw_bytes)
        ocr_text = (ocr_text or "").strip()
        width, height = _dimensions(raw_bytes)

        parts = [f"# {heading}", ""]
        if ocr_text:
            parts += ["## OCR Text", "", ocr_text]
            quality_status = "pass"
        else:
            parts += ["## OCR Text", "", "_(no text detected)_"]
            quality_status = "low_value"
            warnings.append("No OCR text detected — low-value image")

        return ConversionResult(
            markdown_body="\n".join(parts),
            title=heading,
            metadata={
                "extraction_method": "ocr",
                "image_width": width,
                "image_height": height,
                "ocr_char_count": len(ocr_text),
            },
            warnings=warnings,
            requires_ocr=True,
            quality_status=quality_status,
        )
