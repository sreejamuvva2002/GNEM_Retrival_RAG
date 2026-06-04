"""HTML → clean body text extractor using BeautifulSoup.

Extracts text while preserving links and images in the body.
"""
from __future__ import annotations

from typing import Optional


def extract(html_bytes: bytes, url: str = "") -> tuple[str, str]:
    """Extract (title, body_text) from raw HTML bytes.

    Returns ("", "") if extraction fails or yields empty content.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "beautifulsoup4 is required: pip install beautifulsoup4>=4.12"
        ) from exc

    html_str = _decode(html_bytes)
    if not html_str:
        return "", ""

    soup = BeautifulSoup(html_str, "html.parser")

    # Remove script and style elements
    for script_or_style in soup(["script", "style", "noscript", "meta", "header", "footer"]):
        script_or_style.decompose()

    # Convert <a> tags to [text](href)
    for a in soup.find_all("a"):
        href = a.get("href")
        text = a.get_text(strip=True)
        if href and text:
            new_nav_string = soup.new_string(f"[{text}]({href})")
            a.replace_with(new_nav_string)
        elif href:
            new_nav_string = soup.new_string(f"[Link]({href})")
            a.replace_with(new_nav_string)

    # Convert <img> tags to [Image: alt](src)
    for img in soup.find_all("img"):
        src = img.get("src")
        alt = img.get("alt", "image").strip()
        if src:
            new_nav_string = soup.new_string(f"[Image: {alt}]({src})")
            img.replace_with(new_nav_string)

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    if not title:
        title = _extract_title(html_str)

    # Get text
    body = soup.get_text(separator="\n\n", strip=True)
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
