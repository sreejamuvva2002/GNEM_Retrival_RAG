"""Plain text → Markdown (README §19)."""
from __future__ import annotations

from georgia_ev_intelligence.kb_builder.extractors import text_extractor

from .base import BaseConverter, ConversionResult


class TextConverter(BaseConverter):
    extraction_tool = "text-decode"

    def convert(self, raw_bytes: bytes, source_name: str) -> ConversionResult:
        # Strip null bytes before decoding to avoid breaking downstream tools.
        cleaned = raw_bytes.replace(b"\x00", b"")
        title, body = text_extractor.extract(cleaned)
        # Normalize line endings.
        body = body.replace("\r\n", "\n").replace("\r", "\n").strip()
        warnings: list[str] = []
        if not body:
            warnings.append("Empty text content after decoding")

        heading = source_name.rsplit("/", 1)[-1]
        md = f"# {heading}\n\n{body}" if body else f"# {heading}"
        return ConversionResult(
            markdown_body=md,
            title=title or heading,
            metadata={"text_length": len(body)},
            warnings=warnings,
        )
