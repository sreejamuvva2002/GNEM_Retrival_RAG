"""YAML front matter construction (README §26, §40).

``build_frontmatter`` assembles the required+optional metadata dict; ``build_markdown``
wraps a converter body with the YAML block. Both keep field order stable for readable
diffs across re-runs.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import yaml

from .converters import ConversionResult


def build_frontmatter(
    registry_record: dict,
    raw_hash: str,
    result: ConversionResult,
    quality: dict,
    markdown_key: str | None,
    extraction_version: str,
) -> dict:
    """Assemble the YAML front matter for one converted document.

    ``registry_record`` carries source provenance (key, url, mime, sizes, timestamps);
    ``result`` carries converter signals; ``quality`` carries the scored verdict.
    """
    md = result.metadata or {}

    front: dict[str, Any] = {
        "document_id": registry_record.get("document_id"),
        "source_bucket": registry_record.get("source_bucket"),
        "source_key": registry_record.get("source_key"),
        "markdown_key": markdown_key,
        "original_file_name": registry_record.get("original_file_name"),
        "file_type": registry_record.get("file_extension"),
        "mime_type": registry_record.get("mime_type"),
        "file_size_bytes": registry_record.get("file_size_bytes"),
        "raw_sha256": raw_hash,
        "source_url": registry_record.get("source_url"),
        "crawl_timestamp": registry_record.get("crawl_timestamp"),
        "extraction_tool": md.get("extraction_tool") or getattr(result, "extraction_tool", None),
        "extraction_version": extraction_version,
        "status": quality.get("quality_status", "success"),
        "quality_score": quality.get("quality_score"),
        "requires_ocr": result.requires_ocr,
    }

    # Optional metadata — only include when present.
    optional_keys = (
        "page_count", "sheet_count", "row_count", "column_count",
        "image_width", "image_height", "language", "num_tables",
    )
    for key in optional_keys:
        if md.get(key) is not None:
            front[key] = md[key]

    if result.warnings:
        front["warning_count"] = len(result.warnings)
        front["warnings"] = result.warnings

    # Drop keys that are None to keep the front matter clean.
    return {k: v for k, v in front.items() if v is not None}


def build_markdown(frontmatter: dict, body: str) -> str:
    """Wrap a Markdown body with a YAML front-matter block (README §40)."""
    fm = dict(frontmatter)
    fm["extracted_at"] = datetime.now(timezone.utc).isoformat()
    yaml_text = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True)
    return f"---\n{yaml_text}---\n\n{body.strip()}\n"
