"""Image extractor using pytesseract for OCR."""
from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)

def extract(image_bytes: bytes) -> tuple[str, str]:
    """Extract (title, body_text) from raw image bytes using OCR.

    Returns ("Image Document", "") on failure or empty text.
    """
    try:
        from PIL import Image
        import pytesseract
        import sys
        
        # Explicitly set the tesseract path for Windows, rely on PATH for Linux
        if sys.platform == 'win32':
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            
    except ImportError:
        logger.warning("Pillow and pytesseract are required for image OCR. Returning empty text.")
        return "Image Document", ""

    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        body = text.strip()
        title = "Image Document"
        return title, body
    except Exception as exc:
        logger.warning("Failed to extract text from image: %s", exc)
        return "Image Document", ""
