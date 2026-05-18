"""Extract ambiguous phrases from unmatched query tokens."""
from __future__ import annotations

from .constants import STOPWORDS_FOR_AMBIGUITY_EXTRACTION


class AmbiguousTermExtractor:
    """Group remaining meaningful tokens into ambiguous phrases."""

    def extract(
        self,
        tokens: tuple[str, ...],
        excluded_indices: frozenset[int],
    ) -> tuple[list[str], list[str]]:
        """Return (ambiguous_terms, ignored_tokens)."""
        ignored: list[str] = []
        candidate_indices: list[int] = []

        for idx, token in enumerate(tokens):
            if idx in excluded_indices:
                if token in STOPWORDS_FOR_AMBIGUITY_EXTRACTION:
                    ignored.append(token)
                continue
            if token in STOPWORDS_FOR_AMBIGUITY_EXTRACTION:
                ignored.append(token)
                continue
            if len(token) < 2 and not token.isdigit():
                ignored.append(token)
                continue
            candidate_indices.append(idx)

        ambiguous_terms: list[str] = []
        if not candidate_indices:
            return ambiguous_terms, ignored

        start = candidate_indices[0]
        prev = candidate_indices[0]
        group = [tokens[start]]

        for idx in candidate_indices[1:]:
            if idx == prev + 1:
                group.append(tokens[idx])
                prev = idx
            else:
                ambiguous_terms.append(" ".join(group))
                start = idx
                group = [tokens[start]]
                prev = idx
        ambiguous_terms.append(" ".join(group))

        return ambiguous_terms, ignored
