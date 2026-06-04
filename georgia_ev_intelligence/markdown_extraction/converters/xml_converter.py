"""XML → Markdown (README §18).

Preserves element hierarchy, attributes, and text by emitting one section per
element path. Falls back to the simple text extractor if structured parsing fails.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

from georgia_ev_intelligence.kb_builder.extractors import xml_extractor

from .base import BaseConverter, ConversionResult

# Cap to keep huge XML files from producing unbounded Markdown.
_MAX_ELEMENTS = 2000


def _localname(tag: str) -> str:
    """Strip the ``{namespace}`` prefix from an ElementTree tag."""
    return tag.split("}", 1)[-1] if "}" in tag else tag


class XmlConverter(BaseConverter):
    extraction_tool = "xml-etree"

    def convert(self, raw_bytes: bytes, source_name: str) -> ConversionResult:
        heading = source_name.rsplit("/", 1)[-1]
        warnings: list[str] = []
        try:
            root = ET.fromstring(raw_bytes)
        except Exception as exc:
            warnings.append(f"XML parse failed, fell back to text: {exc}")
            _, body = xml_extractor.extract(raw_bytes)
            return ConversionResult(
                markdown_body=f"# {heading}\n\n{body}",
                title=heading,
                warnings=warnings,
                quality_status="needs_review",
            )

        namespaces = sorted({t.split("}", 1)[0][1:] for t in
                             (el.tag for el in root.iter()) if "}" in t})

        parts = [f"# {heading}", "", "## XML Summary", ""]
        parts.append(f"- Root element: `{_localname(root.tag)}`")
        if namespaces:
            parts.append("- Namespaces: " + ", ".join(f"`{ns}`" for ns in namespaces))
        parts += ["", "## Extracted Elements", ""]

        count = 0
        truncated = False

        def walk(elem, path: str) -> None:
            nonlocal count, truncated
            if count >= _MAX_ELEMENTS:
                truncated = True
                return
            name = _localname(elem.tag)
            here = f"{path}/{name}"
            text = (elem.text or "").strip()
            attrs = {_localname(k): v for k, v in elem.attrib.items()}
            if text or attrs:
                count += 1
                parts.append(f"### {here}")
                parts.append("")
                if attrs:
                    attr_str = ", ".join(f"`{k}`=\"{v}\"" for k, v in attrs.items())
                    parts.append(f"- Attributes: {attr_str}")
                if text:
                    parts.append(text)
                parts.append("")
            for child in list(elem):
                walk(child, here)

        walk(root, "")
        if truncated:
            warnings.append(f"XML truncated to first {_MAX_ELEMENTS} elements")
            parts.append(f"_… truncated to first {_MAX_ELEMENTS} elements._")

        return ConversionResult(
            markdown_body="\n".join(parts),
            title=heading,
            metadata={"root_element": _localname(root.tag), "namespaces": namespaces},
            warnings=warnings,
        )
