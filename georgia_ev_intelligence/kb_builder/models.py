"""RawDocument dataclass — the canonical unit written by the crawler."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class RawDocument:
    """A single scraped page/document before chunking."""

    url: str
    body_text: str
    source_type: str                     # company_site | gov_doc | news
    title: str = ""
    domain: str = ""
    http_status: int = 200
    language: str = "en"
    linked_company_id: Optional[str] = None
    crawled_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    file_type: str = "html"              # html | pdf | docx
    raw_binary: Optional[bytes] = None

    # Computed on creation
    content_hash: str = field(init=False)
    doc_id: str = field(init=False)

    def __post_init__(self) -> None:
        normalised = self.body_text.strip().lower()
        self.content_hash = hashlib.sha256(normalised.encode("utf-8")).hexdigest()
        self.doc_id = f"sha256:{self.content_hash}"

    def to_dict(self) -> dict:
        import base64
        raw_binary_str = None
        if self.raw_binary:
            raw_binary_str = base64.b64encode(self.raw_binary).decode("utf-8")

        return {
            "doc_id":            self.doc_id,
            "url":               self.url,
            "domain":            self.domain,
            "source_type":       self.source_type,
            "title":             self.title,
            "body_text":         self.body_text,
            "crawled_at":        self.crawled_at.isoformat(),
            "content_hash":      self.content_hash,
            "http_status":       self.http_status,
            "language":          self.language,
            "linked_company_id": self.linked_company_id,
            "ingestion_status":  "new",
            "file_type":         self.file_type,
            "raw_binary":        raw_binary_str,
        }
