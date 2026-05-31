"""JSON extractor using native python json."""
from __future__ import annotations

import json

def extract(json_bytes: bytes) -> tuple[str, str]:
    """Extract (title, body_text) from raw JSON bytes.

    Returns ("", "") on failure.
    """
    try:
        content = json_bytes.decode("utf-8", errors="replace")
        data = json.loads(content)
        
        # Pretty print the json to make it readable text
        body = json.dumps(data, indent=2)
        title = "JSON Document"
        return title, body
    except Exception:
        return "", ""
