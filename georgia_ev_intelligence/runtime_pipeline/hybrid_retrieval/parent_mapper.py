"""Map reranked child chunks back to deduplicated parent chunks."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from georgia_ev_intelligence.runtime_pipeline.retrieval.parent_fetcher import fetch_parents
from georgia_ev_intelligence.runtime_pipeline.schemas import ParentContext


class ChildWithParentRecordId(Protocol):
    """Child-like retrieval result carrying a parent record id."""

    parent_record_id: str


class ParentChildMapper:
    """Expand child hits to unique parent records using parent_record_id."""

    def map_to_parents(
        self,
        children: Sequence[ChildWithParentRecordId],
    ) -> list[ParentContext]:
        if not children:
            return []

        return fetch_parents([
            child.parent_record_id for child in children
        ])
