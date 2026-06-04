"""XML extractor using native xml.etree.ElementTree."""
from __future__ import annotations

import xml.etree.ElementTree as ET

def extract(xml_bytes: bytes) -> tuple[str, str]:
    """Extract (title, body_text) from raw XML bytes.

    Returns ("", "") on failure.
    """
    try:
        root = ET.fromstring(xml_bytes)
        
        # Extract all text nodes
        texts = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                texts.append(elem.text.strip())
            if elem.tail and elem.tail.strip():
                texts.append(elem.tail.strip())
                
        body = "\n".join(texts).strip()
        title = "XML Document"
        return title, body
    except Exception:
        return "", ""
