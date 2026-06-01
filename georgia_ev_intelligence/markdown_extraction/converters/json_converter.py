"""JSON → Markdown (README §17).

Pretty-prints the JSON in a fenced block and summarizes the top-level structure.
"""
from __future__ import annotations

import json

from .base import BaseConverter, ConversionResult


class JsonConverter(BaseConverter):
    extraction_tool = "json-stdlib"

    def convert(self, raw_bytes: bytes, source_name: str) -> ConversionResult:
        content = raw_bytes.decode("utf-8", errors="replace")
        warnings: list[str] = []
        try:
            data = json.loads(content)
        except Exception as exc:
            warnings.append(f"Invalid JSON: {exc}")
            heading = source_name.rsplit("/", 1)[-1]
            body = f"# {heading}\n\n## Raw Content\n\n```\n{content.strip()}\n```"
            return ConversionResult(
                markdown_body=body,
                title=heading,
                warnings=warnings,
                quality_status="needs_review",
            )

        if isinstance(data, dict):
            top_type = "object"
            top_keys = list(data.keys())
        elif isinstance(data, list):
            top_type = "array"
            top_keys = []
        else:
            top_type = type(data).__name__
            top_keys = []

        pretty = json.dumps(data, indent=2, ensure_ascii=False)
        heading = source_name.rsplit("/", 1)[-1]

        parts = [f"# {heading}", "", "## JSON Summary", ""]
        parts.append(f"- Top-level type: {top_type}")
        if top_keys:
            shown = ", ".join(str(k) for k in top_keys[:50])
            parts.append(f"- Top-level keys: {shown}")
        if top_type == "array":
            parts.append(f"- Record count: {len(data)}")
        parts += ["", "## Extracted Content", "", "```json", pretty, "```"]

        return ConversionResult(
            markdown_body="\n".join(parts),
            title=heading,
            metadata={
                "json_top_type": top_type,
                "json_top_keys": top_keys,
                "record_count": len(data) if isinstance(data, list) else None,
            },
        )
