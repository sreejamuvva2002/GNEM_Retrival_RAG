"""HTML → clean body text extractor using trafilatura.

trafilatura is purpose-built for article extraction and outperforms
BeautifulSoup for removing boilerplate (navbars, footers, ads).
"""
from __future__ import annotations

from typing import Optional


def extract(html_bytes: bytes, url: str = "") -> tuple[str, str]:
    """Extract (title, body_text) from raw HTML bytes.

    Returns ("", "") if extraction fails or yields empty content.
    """
    try:
        import trafilatura  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "trafilatura is required: pip install trafilatura>=1.9"
        ) from exc

    html_str = _decode(html_bytes)
    if not html_str:
        return "", ""

    result = trafilatura.extract(
        html_str,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
        url=url or None,
        output_format="txt",
    )

    title = _extract_title(html_str)
    body = (result or "").strip()
    return title, body


def _extract_title(html: str) -> str:
    """Best-effort <title> tag extraction without a full DOM parse."""
    import re
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try og:title
    m2 = re.search(
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\'](.*?)["\']',
        html,
        re.IGNORECASE,
    )
    if m2:
        return m2.group(1).strip()
    return ""


def _decode(raw: bytes) -> Optional[str]:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return None
