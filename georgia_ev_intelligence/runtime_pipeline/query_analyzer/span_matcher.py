"""Longest non-overlapping vocabulary span matching."""
from __future__ import annotations

from dataclasses import dataclass

from .display import display_source_column, to_canonical_value
from .models import VocabularyMatch, VocabularyTerm
from .vocabulary_index import IndexedPhrase, VocabularyIndex

_MATCH_TYPE_RANK = {"canonical": 0, "normalized": 1, "alias": 2}


@dataclass(frozen=True)
class SpanMatchResult:
    """Output from longest-span matching."""

    matches: list[VocabularyMatch]
    occupied_token_indices: frozenset[int]
    unmatched_token_indices: tuple[int, ...]


class LongestSpanMatcher:
    """Greedy longest-span matcher over normalized query tokens."""

    def match(
        self,
        tokens: tuple[str, ...],
        index: VocabularyIndex,
    ) -> SpanMatchResult:
        if not tokens:
            return SpanMatchResult([], frozenset(), ())

        max_len = min(index.max_phrase_length, len(tokens))
        candidates: list[tuple[int, int, IndexedPhrase, VocabularyTerm]] = []

        for length in range(max_len, 0, -1):
            for start in range(0, len(tokens) - length + 1):
                end = start + length
                phrase = " ".join(tokens[start:end])
                for entry in index.lookup_indexed(phrase):
                    for term in entry.terms:
                        candidates.append((start, end, entry, term))

        candidates.sort(
            key=lambda c: (
                -(c[1] - c[0]),
                _MATCH_TYPE_RANK.get(c[2].match_type, 9),
                -c[2].confidence,
                c[0],
                c[3].normalized_value,
            )
        )

        occupied: set[int] = set()
        accepted: list[VocabularyMatch] = []

        for start, end, entry, term in candidates:
            span_indices = set(range(start, end))
            if span_indices & occupied:
                continue
            occupied |= span_indices
            matched_text = " ".join(tokens[start:end])
            accepted.append(
                VocabularyMatch(
                    matched_text=matched_text,
                    canonical_value=term.canonical_value or to_canonical_value(term.normalized_value),
                    source_column=display_source_column(term.source_column),
                    term_type=term.term_type,
                    match_type=entry.match_type,  # type: ignore[arg-type]
                    start_token=start,
                    end_token=end,
                    token_length=end - start,
                    confidence=entry.confidence,
                    vocabulary_term_id=term.id,
                )
            )

        accepted.sort(key=lambda m: m.start_token)
        unmatched = tuple(i for i in range(len(tokens)) if i not in occupied)
        return SpanMatchResult(
            matches=accepted,
            occupied_token_indices=frozenset(occupied),
            unmatched_token_indices=unmatched,
        )
