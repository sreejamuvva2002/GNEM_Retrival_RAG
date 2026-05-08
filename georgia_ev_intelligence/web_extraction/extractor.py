"""
Phase 1 -- Extractor
Two extraction paths:
  1. PDF -> PyMuPDF (fast, accurate, handles scanned docs)
  2. HTML/Web -> Tavily Extract (replaces trafilatura + requests)

Tavily Extract handles: JS-rendered pages, paywalls, boilerplate removal.
No filtering needed -- Tavily search already ranks by relevance.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from typing import Any

import httpx

from shared.config import Config
from shared.logger import get_logger

logger = get_logger("phase1.extractor")

# PDF mime types and extensions
_PDF_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}
_PDF_EXTENSIONS = {".pdf"}

# Minimum content length to consider a document useful
MIN_CONTENT_CHARS = 150


@dataclass
class ExtractedDocument:
    """Result of extracting text from one URL."""
    url: str
    company_name: str
    content_type: str           # "pdf" or "html"
    text: str                   # Extracted clean text
    title: str = ""
    word_count: int = 0
    char_count: int = 0
    content_hash: str = ""      # SHA-256 of text for deduplication
    raw_bytes: bytes = field(default_factory=bytes)  # Original bytes (PDF only)
    raw_bytes_size: int = 0
    extraction_method: str = "" # "pymupdf" or "tavily_extract"
    error: str = ""             # Non-empty if extraction failed


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _is_pdf_url(url: str) -> bool:
    """Heuristic: URL ends in .pdf or contains /pdf/"""
    url_lower = url.lower()
    return url_lower.endswith(".pdf") or "/pdf/" in url_lower


def _is_pdf_content_type(content_type: str) -> bool:
    return content_type.lower().split(";")[0].strip() in _PDF_CONTENT_TYPES


def extract_pdf_bytes(pdf_bytes: bytes, url: str) -> str:
    """
    Extract text from PDF bytes using PyMuPDF (fitz).
    Much faster than pdfminer, handles complex layouts.
    Returns cleaned text string.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages: list[str] = []
        for page in doc:
            text = page.get_text("text")
            if text and text.strip():
                pages.append(text.strip())
        doc.close()
        full_text = "\n\n".join(pages)
        return full_text
    except Exception as exc:
        logger.warning("PyMuPDF failed for %s: %s", url, exc)
        return ""


async def download_pdf(url: str, timeout: float = 30.0) -> bytes:
    """Download a PDF file and return raw bytes."""
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        return response.content


async def tavily_extract(url: str) -> dict[str, Any]:
    """
    Use Tavily Extract API to get clean text from a URL.
    Handles JS pages, paywalls, boilerplate removal.

    Returns: {"url": str, "raw_content": str, "images": [...]}
    """
    cfg = Config.get()
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.tavily.com/extract",
            json={
                "urls": [url],
                "api_key": cfg.tavily_api_key,
            },
        )
        response.raise_for_status()
        data = response.json()

    results = data.get("results", [])
    if results:
        return results[0]
    # Check failed_results
    failed = data.get("failed_results", [])
    if failed:
        raise ValueError(f"Tavily extract failed for {url}: {failed[0].get('error', 'unknown')}")
    raise ValueError(f"Tavily extract returned no results for {url}")


async def extract_document(
    url: str,
    company_name: str,
    force_pdf: bool = False,
) -> ExtractedDocument:
    """
    Main entry point: extract text from any URL.

    Decision logic:
      - URL ends in .pdf OR force_pdf=True -> download + PyMuPDF
      - Anything else -> Tavily Extract API

    Args:
        url: Source URL
        company_name: For tagging/logging
        force_pdf: Override detection and treat as PDF

    Returns:
        ExtractedDocument with all metadata
    """
    is_pdf = force_pdf or _is_pdf_url(url)

    if is_pdf:
        return await _extract_pdf(url, company_name)
    else:
        return await _extract_html(url, company_name)


async def _extract_pdf(url: str, company_name: str) -> ExtractedDocument:
    """Download PDF and extract via PyMuPDF."""
    logger.info("[PDF] %s", url[:100])
    try:
        raw_bytes = await download_pdf(url)

        # Verify it's actually a PDF
        if not raw_bytes.startswith(b"%PDF"):
            # Not a PDF -- try Tavily extract instead
            logger.info("URL %s not a real PDF (no %%PDF header), falling back to Tavily", url)
            return await _extract_html(url, company_name)

        text = extract_pdf_bytes(raw_bytes, url)
        if not text or len(text) < MIN_CONTENT_CHARS:
            logger.warning("  [FAIL] PDF empty text (len=%d): %s", len(text), url[:80])
            return ExtractedDocument(
                url=url,
                company_name=company_name,
                content_type="pdf",
                text="",
                error=f"PDF extracted empty text (len={len(text)})",
                extraction_method="pymupdf",
            )

        content_hash = _sha256(raw_bytes)
        logger.info("  [OK] PDF %d words extracted", len(text.split()))
        return ExtractedDocument(
            url=url,
            company_name=company_name,
            content_type="pdf",
            text=text,
            word_count=len(text.split()),
            char_count=len(text),
            content_hash=content_hash,
            raw_bytes=raw_bytes,
            raw_bytes_size=len(raw_bytes),
            extraction_method="pymupdf",
        )
    except httpx.HTTPStatusError as exc:
        return ExtractedDocument(
            url=url, company_name=company_name, content_type="pdf", text="",
            error=f"HTTP {exc.response.status_code} downloading PDF",
            extraction_method="pymupdf",
        )
    except Exception as exc:
        err = str(exc)[:200]
        logger.warning("  [FAIL] PDF: %s -- %s", url[:80], err)
        return ExtractedDocument(
            url=url, company_name=company_name, content_type="pdf", text="",
            error=err,
            extraction_method="pymupdf",
        )


async def _extract_html(url: str, company_name: str) -> ExtractedDocument:
    """Use Tavily Extract to get clean text from a web page."""
    logger.info("[HTML] %s", url[:100])
    try:
        result = await tavily_extract(url)
        text = result.get("raw_content", "").strip()

        if not text or len(text) < MIN_CONTENT_CHARS:
            logger.warning("  [FAIL] HTML short/empty (len=%d): %s", len(text), url[:80])
            return ExtractedDocument(
                url=url,
                company_name=company_name,
                content_type="html",
                text="",
                error=f"Tavily extract returned short/empty content (len={len(text)})",
                extraction_method="tavily_extract",
            )

        content_hash = _sha256(text.encode("utf-8"))
        logger.info("  [OK] HTML %d words extracted", len(text.split()))
        return ExtractedDocument(
            url=url,
            company_name=company_name,
            content_type="html",
            text=text,
            word_count=len(text.split()),
            char_count=len(text),
            content_hash=content_hash,
            extraction_method="tavily_extract",
        )
    except httpx.HTTPStatusError as exc:
        return ExtractedDocument(
            url=url, company_name=company_name, content_type="html", text="",
            error=f"Tavily extract HTTP {exc.response.status_code}",
            extraction_method="tavily_extract",
        )
    except Exception as exc:
        return ExtractedDocument(
            url=url, company_name=company_name, content_type="html", text="",
            error=str(exc)[:200],
            extraction_method="tavily_extract",
        )
