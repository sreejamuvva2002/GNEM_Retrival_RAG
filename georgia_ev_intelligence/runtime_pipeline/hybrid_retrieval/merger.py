"""Result merging and deduplication for child chunk retrieval results."""
from __future__ import annotations

from collections.abc import Iterable

from georgia_ev_intelligence.runtime_pipeline.schemas import RetrievedChildChunk


class ChildResultMerger:
    """Merge child results from retrievers and deduplicate by chunk_id."""

    def merge(
        self,
        result_sets: Iterable[Iterable[RetrievedChildChunk]],
    ) -> list[RetrievedChildChunk]:
        merged: list[RetrievedChildChunk] = []
        seen_chunk_ids: set[str] = set()

        for result_set in result_sets:
            for child in result_set:
                if child.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(child.chunk_id)
                merged.append(child)

        return merged
