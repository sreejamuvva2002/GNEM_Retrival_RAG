"""Vocabulary-based parent retriever using exact row_id matching.

Given VocabularyMatches (already resolved row_ids), fetches matching
ParentContext records directly from the parent_chunks table. This is
an exact-match retriever -- no fuzzy search involved.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import psycopg2

from ...shared import config
from ..query_rewriting.models import TermMatch, VocabularyMatches
from ..schemas import ParentContext

logger = logging.getLogger(__name__)

_FETCH_BY_ROW_IDS_SQL = """
SELECT
    record_id, source_row_id, company, category,
    industry_group, updated_location, primary_facility_type,
    ev_supply_chain_role, primary_oems,
    supplier_or_affiliation_type, employment,
    product_service, ev_battery_relevant, parent_chunk_text
FROM parent_chunks
WHERE source_row_id = ANY(%s);
"""


class VocabularyFilterRetriever:
    """Retrieves ParentContext records directly from parent_chunks
    using row_ids resolved from kb_vocabulary_terms.

    Operates at parent level (unlike Dense and BM25 which are child-level).
    Assigns a high base score to ensure vocabulary matches appear at
    the top of the merged result list.
    """

    VOCABULARY_BASE_SCORE: float = 1.5

    def search(self, matches: VocabularyMatches) -> list[ParentContext]:
        """Fetch parent records for vocabulary-matched row_ids.

        Uses intersected_row_ids if non-empty, falls back to union_row_ids.
        Returns ParentContext list sorted by combined_score descending.
        """
        if not matches.has_matches:
            return []

        row_ids = matches.intersected_row_ids
        if not row_ids:
            row_ids = matches.union_row_ids
        if not row_ids:
            return []

        rows = self._fetch_by_row_ids(row_ids)
        if not rows:
            return []

        # Build ParentContext objects with vocabulary-based scoring
        contexts: list[ParentContext] = []
        for data in rows:
            score = self._compute_score(data["source_row_id"], matches)
            metadata = {
                k: v
                for k, v in data.items()
                if k not in ("record_id", "source_row_id", "parent_chunk_text")
            }
            contexts.append(
                ParentContext(
                    record_id=data["record_id"],
                    source_row_id=data["source_row_id"],
                    parent_chunk_text=data["parent_chunk_text"],
                    metadata=metadata,
                    max_rrf_score=0.0,
                    matched_child_ids=[],
                    matched_child_types=[],
                    dense_hit_count=0,
                    bm25_hit_count=0,
                    combined_score=score,
                )
            )

        contexts.sort(key=lambda p: p.combined_score, reverse=True)
        return contexts

    def _fetch_by_row_ids(self, row_ids: list[int]) -> list[dict[str, Any]]:
        """Fetch parent records from parent_chunks WHERE source_row_id = ANY(%s).

        Returns list of dicts built from cursor rows.
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(_FETCH_BY_ROW_IDS_SQL, (row_ids,))
                rows = cur.fetchall()
        finally:
            conn.close()

        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "record_id": row[0],
                    "source_row_id": int(row[1]) if row[1] is not None else 0,
                    "company": row[2] or "",
                    "category": row[3] or "",
                    "industry_group": row[4] or "",
                    "updated_location": row[5] or "",
                    "primary_facility_type": row[6] or "",
                    "ev_supply_chain_role": row[7] or "",
                    "primary_oems": row[8] or "",
                    "supplier_or_affiliation_type": row[9] or "",
                    "employment": row[10],
                    "product_service": row[11] or "",
                    "ev_battery_relevant": row[12] or "",
                    "parent_chunk_text": row[13] or "",
                }
            )
        return results

    def _compute_score(
        self, source_row_id: int, matches: VocabularyMatches
    ) -> float:
        """Compute the combined_score for a parent record.

        Score = VOCABULARY_BASE_SCORE + (normalised frequency bonus) * 0.1
        where frequency bonus = sum of term_frequency for all TermMatch
        whose row_ids include this source_row_id, normalised by the max
        term_frequency across all matches.
        """
        if not matches.term_matches:
            return self.VOCABULARY_BASE_SCORE

        max_freq = max(m.term_frequency for m in matches.term_matches)
        if max_freq == 0:
            return self.VOCABULARY_BASE_SCORE

        freq_sum = sum(
            m.term_frequency
            for m in matches.term_matches
            if source_row_id in m.row_ids
        )
        normalised_bonus = freq_sum / max_freq
        return self.VOCABULARY_BASE_SCORE + normalised_bonus * 0.1

    def _get_connection(self):
        """Create a PostgreSQL connection (same pattern as parent_fetcher.py)."""
        return psycopg2.connect(config.NEON_DATABASE_URL)
