"""Text extractor using native python decoding."""
from __future__ import annotations

def extract(text_bytes: bytes) -> tuple[str, str]:
    """Extract (title, body_text) from raw text bytes.

    Returns ("", "") on failure.
    """
    content = None
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            content = text_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
            
    if not content:
        return "", ""
        
    body = content.strip()
    title = "Text Document"
    return title, body
