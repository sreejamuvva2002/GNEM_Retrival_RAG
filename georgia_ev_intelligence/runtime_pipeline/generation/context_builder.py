"""Build structured LLM context from parent chunks with citation IDs."""
from __future__ import annotations

from ..schemas import ParentContext


def build_context(
    parents: list[ParentContext],
    max_context_chars: int | None = None,
    max_context_records: int | None = None,
) -> tuple[str, dict[str, ParentContext], list[ParentContext]]:
    """Build numbered context blocks for the LLM prompt.

    Each parent receives a citation ID [S1], [S2], etc.
    Uses parent_chunk_text and available metadata dynamically.
    Does not hardcode any KB values.

    Safety controls:
    - max_context_records: maximum number of parent records to include (default: 30)
    - max_context_chars: maximum total characters in the context string (default: 24000)

    Parents are already sorted by combined_score descending, so truncation
    removes the least relevant records while preserving citation ID mapping.

    Returns:
        (context_string, citation_map, included_parents)
        - context_string: the formatted context to send to the LLM
        - citation_map: maps "S1" -> ParentContext for included records
        - included_parents: the ParentContext objects actually included in the context
    """
    if not parents:
        return "No matching records found in the knowledge base.", {}, []

    limit_records = max_context_records if max_context_records is not None else 30
    limit_chars = max_context_chars if max_context_chars is not None else 24000

    blocks: list[str] = []
    citation_map: dict[str, ParentContext] = {}
    included_parents: list[ParentContext] = []
    total_chars = 0

    for idx, parent in enumerate(parents[:limit_records], start=1):
        citation_id = f"S{idx}"
        block = f"[{citation_id}]\n{parent.parent_chunk_text}"

        # Check character budget before adding
        block_size = len(block) + 2  # +2 for the "\n\n" separator
        if total_chars + block_size > limit_chars and blocks:
            break

        citation_map[citation_id] = parent
        included_parents.append(parent)
        blocks.append(block)
        total_chars += block_size

    context = "\n\n".join(blocks)
    return context, citation_map, included_parents
