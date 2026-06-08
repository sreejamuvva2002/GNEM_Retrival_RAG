"""Shared helpers for the document-retrieval executors.

Keyword, vector and hybrid search all return child chunks, map them back to
parent records, and present the parent text as evidence. This module centralises
the query-text selection and the chunk -> parent -> evidence assembly so the
three executors stay thin.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import answer_formatter as fmt
from ..schemas import STATUS_SUCCESS, ExecutionResult

if TYPE_CHECKING:
    from georgia_ev_intelligence.runtime_pipeline.schemas import RetrievedChildChunk

_PREVIEW_PARENTS = 10


def query_text(final_route: dict[str, Any]) -> str:
    """Prefer the router's distilled ``query_focus``, else the raw question."""
    focus = (final_route.get("query_focus") or "").strip()
    if focus:
        return focus
    return (final_route.get("question") or "").strip()


def build_chunk_result(
    route: str,
    chunks: list["RetrievedChildChunk"],
    *,
    top_parents: int = _PREVIEW_PARENTS,
) -> ExecutionResult:
    """Map retrieved chunks to parents and assemble a document-evidence result."""
    from georgia_ev_intelligence.runtime_pipeline.retrieval.parent_fetcher import (
        fetch_parents,
    )

    if not chunks:
        return ExecutionResult(
            route=route,
            status=STATUS_SUCCESS,
            answer="No supporting evidence found.",
            evidence={"type": "document_chunks", "chunks": [], "parents": []},
        )

    ordered_parent_ids = [c.parent_record_id for c in chunks]
    contexts = fetch_parents(ordered_parent_ids)

    parents = [
        {
            "parent_record_id": ctx.record_id,
            "source_row_id": ctx.source_row_id,
            "text": ctx.parent_chunk_text,
        }
        for ctx in contexts
    ]
    previews = parents[:top_parents]

    chunk_refs = [
        {
            "chunk_id": c.chunk_id,
            "parent_record_id": c.parent_record_id,
            "chunk_type": c.chunk_type,
        }
        for c in chunks
    ]

    return ExecutionResult(
        route=route,
        status=STATUS_SUCCESS,
        answer=fmt.format_document_chunks(previews),
        evidence={"type": "document_chunks", "chunks": chunk_refs, "parents": parents},
    )
