"""Extraction quality heuristic (README §28, §42).

Simple, fast signals over the rendered Markdown. Converter-supplied signals (e.g.
``low_value``/``needs_review`` from OCR or PDF page checks) take precedence over the
length-based default so format-specific knowledge is not overridden.
"""
from __future__ import annotations

from typing import Any

# Status precedence — higher wins when combining converter + heuristic verdicts.
_SEVERITY = {"pass": 0, "low_value": 1, "needs_review": 2, "failed": 3}


def _worst(a: str, b: str) -> str:
    return a if _SEVERITY.get(a, 0) >= _SEVERITY.get(b, 0) else b


def compute_quality(
    markdown_text: str,
    metadata: dict[str, Any] | None = None,
    converter_status: str = "pass",
) -> dict:
    """Return a quality report dict for one converted document."""
    text = markdown_text.strip()
    text_length = len(text)
    num_headings = text.count("\n#") + (1 if text.startswith("#") else 0)
    num_tables = text.count("|---")
    num_links = text.count("](")

    if text_length < 100:
        status = "failed"
        score = 0.0
    elif text_length < 500:
        status = "needs_review"
        score = 0.4
    else:
        status = "pass"
        score = 0.8

    # Reward structure on otherwise-passing documents.
    if status == "pass" and (num_tables or num_headings > 1):
        score = min(0.95, score + 0.1)

    status = _worst(status, converter_status)
    # If the converter flagged a problem, cap the score.
    if status == "failed":
        score = min(score, 0.0)
    elif status in ("needs_review", "low_value"):
        score = min(score, 0.4)

    return {
        "text_length": text_length,
        "num_headings": num_headings,
        "num_tables": num_tables,
        "num_links": num_links,
        "quality_score": round(score, 2),
        "quality_status": status,
    }
