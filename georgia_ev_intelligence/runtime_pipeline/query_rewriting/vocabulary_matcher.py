"""Vocabulary term matcher against kb_vocabulary_terms.

Given a StructuredQuery, queries the kb_vocabulary_terms table to find
matching terms and resolve them to parent row_ids. Uses exact, trigram,
and prefix matching in priority order.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import psycopg2

from ...shared import config
from .models import StructuredQuery, TermMatch, VocabularyMatches

logger = logging.getLogger(__name__)

_EXACT_SQL = """
SELECT normalized_value, term_type, source_column,
       row_ids, term_frequency
FROM kb_vocabulary_terms
WHERE normalized_value = %s AND term_type = %s
ORDER BY term_frequency DESC
LIMIT 10;
"""

_TRIGRAM_SQL = """
SELECT normalized_value, term_type, source_column,
       row_ids, term_frequency,
       similarity(normalized_value, %s) AS sim
FROM kb_vocabulary_terms
WHERE term_type = %s
  AND normalized_value %% %s
ORDER BY sim DESC, term_frequency DESC
LIMIT 5;
"""

_PREFIX_SQL = """
SELECT normalized_value, term_type, source_column,
       row_ids, term_frequency
FROM kb_vocabulary_terms
WHERE term_type = %s
  AND normalized_value ILIKE %s
ORDER BY term_frequency DESC
LIMIT 5;
"""


class VocabularyMatcher:
    """Queries kb_vocabulary_terms to resolve StructuredQuery filters
    into concrete parent row_ids.

    Uses the existing B-tree index for exact match, GIN trigram index
    for fuzzy matching, and ILIKE as a final fallback.
    """

    def match(self, structured_query: StructuredQuery) -> VocabularyMatches:
        """Match all terms in structured_query against kb_vocabulary_terms.

        For each (term, term_type) pair, calls _match_single_term().
        Aggregates into VocabularyMatches with intersection logic:
          - Group TermMatch objects by term_type
          - For each term_type, union the row_ids within that type
          - Then intersect across types
        If intersection is empty but union is non-empty,
        set intersected_row_ids=[] but union_row_ids=<all matched>.
        """
        all_terms = structured_query.all_terms()
        if not all_terms:
            return VocabularyMatches(structured_query=structured_query)

        all_matches: list[TermMatch] = []
        conn = self._get_connection()
        try:
            for term, term_type in all_terms:
                matches = self._match_single_term(term, term_type, conn)
                all_matches.extend(matches)
        finally:
            conn.close()

        if not all_matches:
            return VocabularyMatches(
                structured_query=structured_query,
                match_count=0,
                has_matches=False,
            )

        # Intersection logic: OR within same term_type, AND across types
        type_row_ids: dict[str, set[int]] = defaultdict(set)
        all_row_ids: set[int] = set()

        for m in all_matches:
            for rid in m.row_ids:
                type_row_ids[m.term_type].add(rid)
                all_row_ids.add(rid)

        # Intersect across term types
        type_sets = list(type_row_ids.values())
        intersected = type_sets[0].copy()
        for s in type_sets[1:]:
            intersected &= s

        union_ids = sorted(all_row_ids)
        intersected_ids = sorted(intersected)

        return VocabularyMatches(
            structured_query=structured_query,
            term_matches=all_matches,
            intersected_row_ids=intersected_ids,
            union_row_ids=union_ids,
            match_count=len(all_matches),
            has_matches=len(intersected_ids) > 0,
        )

    def _match_single_term(
        self,
        term: str,
        term_type: str,
        conn,
    ) -> list[TermMatch]:
        """Run three queries in priority order; stop at first success.

        Query 1: Exact match (B-tree index)
        Query 2: Trigram fuzzy match (GIN pg_trgm index), sim >= 0.3
        Query 3: Prefix/contains match (ILIKE fallback)
        """
        normalized = term.lower().strip()

        with conn.cursor() as cur:
            # Query 1: Exact match
            cur.execute(_EXACT_SQL, (normalized, term_type))
            rows = cur.fetchall()
            if rows:
                return [
                    TermMatch(
                        normalized_value=r[0],
                        term_type=r[1],
                        source_column=r[2],
                        row_ids=list(r[3]) if r[3] else [],
                        term_frequency=r[4],
                        match_type="exact",
                    )
                    for r in rows
                ]

            # Query 2: Trigram fuzzy match
            cur.execute(_TRIGRAM_SQL, (normalized, term_type, normalized))
            rows = cur.fetchall()
            if rows:
                results = []
                for r in rows:
                    sim = r[5]
                    if sim >= 0.3:
                        results.append(
                            TermMatch(
                                normalized_value=r[0],
                                term_type=r[1],
                                source_column=r[2],
                                row_ids=list(r[3]) if r[3] else [],
                                term_frequency=r[4],
                                match_type="trigram",
                            )
                        )
                if results:
                    return results

            # Query 3: Prefix/contains fallback
            cur.execute(_PREFIX_SQL, (term_type, f"%{normalized}%"))
            rows = cur.fetchall()
            if rows:
                return [
                    TermMatch(
                        normalized_value=r[0],
                        term_type=r[1],
                        source_column=r[2],
                        row_ids=list(r[3]) if r[3] else [],
                        term_frequency=r[4],
                        match_type="trigram",
                    )
                    for r in rows
                ]

        return []

    def _get_connection(self):
        """Create a PostgreSQL connection using the same pattern as parent_fetcher.py."""
        return psycopg2.connect(config.NEON_DATABASE_URL)
