"""In-memory vocabulary lookup index."""
from __future__ import annotations

from dataclasses import dataclass

from .models import VocabularyTerm


@dataclass(frozen=True)
class IndexedPhrase:
    """Maps a searchable phrase to vocabulary term(s)."""

    phrase: str
    match_type: str  # canonical | alias | normalized
    terms: tuple[VocabularyTerm, ...]
    confidence: float


class VocabularyIndex:
    """Lookup index for normalized vocabulary phrases."""

    def __init__(self, terms: list[VocabularyTerm]) -> None:
        self._lookup: dict[str, list[IndexedPhrase]] = {}
        self._max_token_length = 1
        for term in terms:
            self._register_phrase(term.normalized_value, term, "canonical", 1.0)
            canon_norm = term.canonical_value.strip().lower()
            if canon_norm and canon_norm != term.normalized_value:
                self._register_phrase(canon_norm, term, "canonical", 1.0)
            for alias in term.aliases:
                alias_norm = alias.strip().lower()
                if alias_norm and alias_norm != term.normalized_value:
                    self._register_phrase(alias_norm, term, "alias", 0.9)

    def _register_phrase(
        self,
        phrase: str,
        term: VocabularyTerm,
        match_type: str,
        confidence: float,
    ) -> None:
        phrase = phrase.strip().lower()
        if not phrase:
            return
        token_count = len(phrase.split())
        self._max_token_length = max(self._max_token_length, token_count)
        bucket = self._lookup.setdefault(phrase, [])
        for existing in bucket:
            if existing.match_type == match_type and term in existing.terms:
                return
        bucket.append(
            IndexedPhrase(
                phrase=phrase,
                match_type=match_type,
                terms=(term,),
                confidence=confidence,
            )
        )

    @property
    def max_phrase_length(self) -> int:
        return self._max_token_length

    def lookup(self, phrase: str) -> list[VocabularyTerm]:
        """Return all vocabulary terms for an exact phrase key."""
        key = phrase.strip().lower()
        entries = self._lookup.get(key, [])
        seen: set[int] = set()
        results: list[VocabularyTerm] = []
        for entry in sorted(entries, key=lambda e: (-e.confidence, e.phrase)):
            for term in entry.terms:
                term_key = term.id if term.id is not None else hash(
                    (term.normalized_value, term.source_column)
                )
                if term_key in seen:
                    continue
                seen.add(term_key)
                results.append(term)
        return results

    def lookup_indexed(self, phrase: str) -> list[IndexedPhrase]:
        key = phrase.strip().lower()
        return list(self._lookup.get(key, []))

    @property
    def phrase_count(self) -> int:
        return len(self._lookup)
