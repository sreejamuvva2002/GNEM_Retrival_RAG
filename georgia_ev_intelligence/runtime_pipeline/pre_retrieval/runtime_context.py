"""Pre-generation context package built before final answer generation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..ClarificationOptionGenerator.models import ResolvedQueryContext


@dataclass
class PreGenerationContextPackage:
    """Everything downstream generation needs after query analysis + retrieval."""

    original_query: str
    resolved_query_context: ResolvedQueryContext
    retrieval_results: list[Any] = field(default_factory=list)
    reranked_results: list[Any] | None = None
    parent_records: list[Any] = field(default_factory=list)
    citations: list[Any] = field(default_factory=list)
    retrieval_trace: dict[str, Any] = field(default_factory=dict)
    ready_for_generation: bool = False
