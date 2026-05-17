"""Prepare RAGAS-compatible evaluation datasets from pipeline results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...shared import config
from ..schemas import RagResult


def prepare_sample(
    result: RagResult,
    reference_answer: str = "",
) -> dict[str, Any]:
    """Convert one RagResult into a RAGAS-compatible evaluation sample.

    RAGAS expects:
    - question: the user question
    - answer: the generated answer
    - contexts: list of context strings used for generation
    - ground_truth: reference answer (optional)

    Uses the structured parent_chunk_texts from the trace (the actual
    parent record texts sent to the LLM), not a naive split of the
    formatted context string.
    """
    contexts: list[str] = result.trace.parent_chunk_texts

    return {
        "question": result.question,
        "answer": result.answer,
        "contexts": contexts,
        "ground_truth": reference_answer,
    }


def prepare_dataset(
    results: list[tuple[RagResult, str]],
) -> list[dict[str, Any]]:
    """Convert multiple (RagResult, reference_answer) pairs into a RAGAS dataset.

    Args:
        results: List of (RagResult, reference_answer) tuples.
                 reference_answer can be empty string if not available.

    Returns:
        List of RAGAS-compatible sample dicts.
    """
    return [prepare_sample(r, ref) for r, ref in results]


def save_dataset(
    dataset: list[dict[str, Any]],
    output_path: Path | None = None,
) -> Path:
    """Save RAGAS dataset to a JSON file for later evaluation."""
    out_path = output_path or (config.OUTPUTS_DIR / "ragas_dataset.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2, default=str)

    return out_path
