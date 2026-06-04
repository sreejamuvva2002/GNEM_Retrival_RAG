"""HTML → Markdown (README §13).

Prefers trafilatura for main-article extraction (drops nav/footer/boilerplate) and
falls back to the existing BeautifulSoup ``html_extractor`` (which already emits
``[text](href)`` links and ``[Image: alt](src)`` markers).
"""
from __future__ import annotations

from georgia_ev_intelligence.kb_builder.extractors import html_extractor

from .base import BaseConverter, ConversionResult

# Below this body length the page is treated as low-value (mostly nav/boilerplate).
_LOW_VALUE_CHARS = 200


def _trafilatura_markdown(raw_bytes: bytes) -> tuple[str | None, str | None]:
    """Return (title, markdown_body) via trafilatura, or (None, None) if unavailable."""
    try:
        import trafilatura  # type: ignore
    except ImportError:
        return None, None
    try:
        html = None
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                html = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if html is None:
            return None, None
        md = trafilatura.extract(
            html,
            output_format="markdown",
            include_links=True,
            include_tables=True,
            favor_recall=True,
        )
        if not md or not md.strip():
            return None, None
        meta = trafilatura.extract_metadata(html)
        title = getattr(meta, "title", None) if meta else None
        return title, md.strip()
    except Exception:
        return None, None


class HtmlConverter(BaseConverter):
    extraction_tool = "trafilatura+bs4"

    def convert(self, raw_bytes: bytes, source_name: str) -> ConversionResult:
        warnings: list[str] = []

        title, body = _trafilatura_markdown(raw_bytes)
        used_tool = "trafilatura"
        if body is None:
            used_tool = "beautifulsoup"
            t2, body = html_extractor.extract(raw_bytes)
            title = title or t2

        body = (body or "").strip()
        if not title:
            title = source_name.rsplit("/", 1)[-1]

        quality_status = "pass"
        if len(body) < _LOW_VALUE_CHARS:
            quality_status = "low_value"
            warnings.append("Little body content — likely nav/boilerplate-only page")

        md = f"# {title}\n\n{body}" if body else f"# {title}"
        return ConversionResult(
            markdown_body=md,
            title=title,
            metadata={"html_extractor": used_tool, "html_title": title},
            warnings=warnings,
            quality_status=quality_status,
        )
