"""Query text normalization and tokenization."""
from __future__ import annotations

import re
from dataclasses import dataclass

# Word chars plus slash compounds (tier 1/2, 2/3)
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:/[a-z0-9]+)?", re.IGNORECASE)

# Hyphen between letters -> space (capacity-fragile)
_HYPHEN_PATTERN = re.compile(r"(?<=[a-z0-9])-(?=[a-z0-9])", re.IGNORECASE)


@dataclass(frozen=True)
class NormalizedQuery:
    """Normalized query text and positional tokens."""

    original: str
    normalized: str
    tokens: tuple[str, ...]


class QueryNormalizer:
    """Normalize and tokenize user queries for span matching."""

    def normalize(self, query: str) -> NormalizedQuery:
        original = query.strip()
        text = original.lower()
        text = _HYPHEN_PATTERN.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = tuple(_TOKEN_PATTERN.findall(text))
        return NormalizedQuery(original=original, normalized=text, tokens=tokens)

    def singularize_entity_token(self, token: str) -> str:
        """Safe singularization for target-entity detection only."""
        if len(token) > 3 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
        return token
