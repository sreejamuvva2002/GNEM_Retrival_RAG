"""Map LLM answer citations back to parent records."""
from __future__ import annotations

import re

from ..schemas import Citation, CitationOutput, ParentContext


def format_citations(
    answer: str,
    citation_map: dict[str, ParentContext],
) -> CitationOutput:
    """Extract citations from the LLM answer and map them to source records.

    Returns both:
    - used_citations: only those citation IDs actually referenced in the answer
    - all_source_records: every parent that was sent to the LLM as context
    """
    # Build all source records (everything sent to the LLM)
    all_source_records: list[Citation] = []
    for cid, parent in citation_map.items():
        all_source_records.append(_citation_from_parent(cid, parent))

    # Extract citation IDs referenced in the answer (e.g., [S1], [S2], [S3])
    used_ids = set(re.findall(r"\[S(\d+)\]", answer))
    used_citations: list[Citation] = []
    for cid, parent in citation_map.items():
        # cid is like "S1", "S2"
        numeric = cid[1:]  # strip the "S" prefix
        if numeric in used_ids:
            used_citations.append(_citation_from_parent(cid, parent))

    return CitationOutput(
        used_citations=used_citations,
        all_source_records=all_source_records,
    )


def _citation_from_parent(citation_id: str, parent: ParentContext) -> Citation:
    """Build a Citation from a ParentContext."""
    return Citation(
        citation_id=citation_id,
        parent_record_id=parent.record_id,
        source_row_id=parent.source_row_id,
        company=parent.metadata.get("company", ""),
        retrieval_score=parent.combined_score,
    )
