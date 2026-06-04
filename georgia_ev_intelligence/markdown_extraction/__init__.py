"""Raw web documents → Markdown extraction pipeline.

Reads raw files already stored in Backblaze B2 by the kb_builder crawler, converts
each supported file type into YAML-fronted Markdown, runs quality checks, uploads the
Markdown to a processed/ prefix, and tracks everything with JSONL manifests.

The converters wrap the existing ``kb_builder.extractors`` for parsing and the B2 client
reuses ``kb_builder.b2_uploader`` for credentials — no duplicated parsing or boto3 logic.

CLI entry-point:
    python -m georgia_ev_intelligence.markdown_extraction --help
"""
from __future__ import annotations

__all__ = ["cli", "pipeline"]
