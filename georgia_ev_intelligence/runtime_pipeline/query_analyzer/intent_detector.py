"""Operation and target-entity detection from query tokens."""
from __future__ import annotations

from dataclasses import dataclass

from .constants import OPERATION_PHRASES, OPERATION_WORDS, TARGET_ENTITY_WORDS
from .normalizer import QueryNormalizer


@dataclass(frozen=True)
class IntentResult:
    operation: str | None
    target_entity: str | None
    operation_token_indices: frozenset[int]
    target_entity_token_indices: frozenset[int]


class IntentDetector:
    """Detect list/count/compare operations and target entity nouns."""

    def __init__(self, normalizer: QueryNormalizer | None = None) -> None:
        self._normalizer = normalizer or QueryNormalizer()

    def detect(
        self,
        normalized_text: str,
        tokens: tuple[str, ...],
    ) -> IntentResult:
        operation: str | None = None
        operation_indices: set[int] = set()
        target_entity: str | None = None
        target_indices: set[int] = set()

        for phrase, op in OPERATION_PHRASES:
            if normalized_text.startswith(phrase):
                operation = op
                phrase_tokens = phrase.split()
                operation_indices.update(range(len(phrase_tokens)))
                break

        if operation is None:
            for idx, token in enumerate(tokens):
                if token in OPERATION_WORDS:
                    operation = OPERATION_WORDS[token]
                    operation_indices.add(idx)
                    if token == "how" and idx + 1 < len(tokens) and tokens[idx + 1] == "many":
                        operation = "count"
                        operation_indices.add(idx + 1)
                    break

        for idx, token in enumerate(tokens):
            entity = TARGET_ENTITY_WORDS.get(token)
            if entity is None:
                singular = self._normalizer.singularize_entity_token(token)
                entity = TARGET_ENTITY_WORDS.get(singular)
            if entity:
                target_entity = entity
                target_indices.add(idx)

        return IntentResult(
            operation=operation,
            target_entity=target_entity,
            operation_token_indices=frozenset(operation_indices),
            target_entity_token_indices=frozenset(target_indices),
        )
