"""CSV/TSV extractor using pandas."""
from __future__ import annotations

import io

def extract(csv_bytes: bytes) -> tuple[str, str]:
    """Extract (title, body_text) from raw CSV/TSV bytes.

    Returns ("", "") on failure.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required: pip install pandas>=2.0") from exc

    try:
        # Try different encodings
        content = None
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                content = csv_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue
                
        if not content:
            return "", ""

        # Use pandas to parse and then output a clean text representation
        # It handles delimiters automatically better than standard csv module
        # but standard csv module or just raw text is also fine.
        # Returning the decoded text is often sufficient for CSV, 
        # but pandas can normalize it.
        # Actually, for CSV, returning the raw text is usually the most lossless way.
        
        body = content.strip()
        title = "CSV/TSV Document"
        return title, body
    except Exception:
        return "", ""
