"""Map reranked child chunks back to deduplicated parent chunks."""
from __future__ import annotations

from georgia_ev_intelligence.runtime_pipeline.retrieval.parent_fetcher import fetch_parents
from georgia_ev_intelligence.runtime_pipeline.schemas import ParentContext

from .models import RerankedChildChunk


class ParentChildMapper:
    """Expand reranked children to their parent records using parent_record_id."""

    def map_to_parents(
        self,
        reranked_children: list[RerankedChildChunk],
    ) -> list[ParentContext]:
        if not reranked_children:
            return []

        return fetch_parents([
            child.parent_record_id for child in reranked_children
        ])
