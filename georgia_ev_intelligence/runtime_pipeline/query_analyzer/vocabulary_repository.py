"""Vocabulary loading abstractions."""
from __future__ import annotations

from typing import Protocol

import psycopg2

from ...shared import config
from .display import to_canonical_value
from .models import VocabularyTerm

_LOAD_TERMS_SQL = """
SELECT
    id,
    normalized_value,
    term_type,
    source_column,
    row_ids,
    term_frequency
FROM kb_vocabulary_terms
ORDER BY term_frequency DESC, LENGTH(normalized_value) DESC;
"""


class VocabularyRepository(Protocol):
    """Loads vocabulary terms for query analysis."""

    def load_terms(self) -> list[VocabularyTerm]:
        ...


class InMemoryVocabularyRepository:
    """Vocabulary repository backed by an in-memory term list."""

    def __init__(self, terms: list[VocabularyTerm]) -> None:
        self._terms = list(terms)

    def load_terms(self) -> list[VocabularyTerm]:
        return list(self._terms)


class PostgresVocabularyRepository:
    """Loads vocabulary terms from kb_vocabulary_terms."""

    def __init__(self, database_url: str | None = None) -> None:
        self._database_url = database_url or config.NEON_DATABASE_URL

    def load_terms(self) -> list[VocabularyTerm]:
        conn = psycopg2.connect(self._database_url)
        try:
            with conn.cursor() as cur:
                cur.execute(_LOAD_TERMS_SQL)
                rows = cur.fetchall()
        finally:
            conn.close()

        terms: list[VocabularyTerm] = []
        for row in rows:
            term_id, normalized_value, term_type, source_column, row_ids, frequency = row
            normalized = str(normalized_value or "").strip()
            if not normalized:
                continue
            terms.append(
                VocabularyTerm(
                    id=int(term_id),
                    canonical_value=to_canonical_value(normalized),
                    normalized_value=normalized,
                    source_column=str(source_column),
                    term_type=str(term_type) if term_type else None,
                    aliases=[],
                    row_ids=list(row_ids) if row_ids else [],
                    metadata={"term_frequency": int(frequency or 0)},
                )
            )
        return terms
