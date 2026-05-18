"""Prepare RAGAS-compatible evaluation samples from pipeline results."""
from __future__ import annotations

from typing import Any

from ..schemas import RagResult


def prepare_sample(
    result: RagResult,
    reference_answer: str = "",
) -> dict[str, Any]:
    """Convert one RagResult into a RAGAS-compatible evaluation sample."""
    return {
        "question": result.question,
        "answer": result.answer,
        "contexts": result.trace.parent_chunk_texts,
        "ground_truth": reference_answer,
    }
