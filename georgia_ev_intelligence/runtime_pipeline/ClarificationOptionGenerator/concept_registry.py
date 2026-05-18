"""Deterministic concept registry for abstract ambiguous terms."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

from .exceptions import ConceptRegistryError
from .models import ClarificationOption, ConceptDefinition

_HYPHEN_PATTERN = re.compile(r"(?<=[a-z0-9])-(?=[a-z0-9])", re.IGNORECASE)


def normalize_term(term: str) -> str:
    text = term.strip().lower()
    text = _HYPHEN_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


class ConceptRegistryProtocol(Protocol):
    def find_concept(self, term: str) -> ConceptDefinition | None:
        ...


class InMemoryConceptRegistry:
    """Exact alias lookup against in-memory concept definitions."""

    def __init__(self, concepts: list[ConceptDefinition]) -> None:
        self._by_alias: dict[str, ConceptDefinition] = {}
        for concept in concepts:
            for alias in concept.aliases:
                key = normalize_term(alias)
                if not key:
                    continue
                self._by_alias[key] = concept

    def find_concept(self, term: str) -> ConceptDefinition | None:
        return self._by_alias.get(normalize_term(term))


class JsonConceptRegistry:
    """Load concepts from a JSON file and delegate to InMemoryConceptRegistry."""

    def __init__(self, path: Path | str) -> None:
        self._inner = InMemoryConceptRegistry(_load_concepts_from_json(Path(path)))

    def find_concept(self, term: str) -> ConceptDefinition | None:
        return self._inner.find_concept(term)


def default_concepts_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "concepts.json"


def _load_concepts_from_json(path: Path) -> list[ConceptDefinition]:
    if not path.is_file():
        raise ConceptRegistryError(f"Concept registry file not found: {path}")
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ConceptRegistryError("Concept registry root must be a JSON array.")
    concepts: list[ConceptDefinition] = []
    for entry in raw:
        concepts.append(_parse_concept_entry(entry))
    return concepts


def _parse_concept_entry(entry: dict) -> ConceptDefinition:
    try:
        options = [
            ClarificationOption(
                id=str(opt["id"]),
                label=str(opt["label"]),
                meaning=str(opt["meaning"]),
                requires_custom_text=bool(opt.get("requires_custom_text", False)),
                ignore_term=bool(opt.get("ignore_term", False)),
            )
            for opt in entry["clarification_options"]
        ]
        return ConceptDefinition(
            concept_id=str(entry["concept_id"]),
            aliases=[str(a) for a in entry["aliases"]],
            description=str(entry.get("description", "")),
            clarification_options=options,
        )
    except (KeyError, TypeError) as e:
        raise ConceptRegistryError(f"Invalid concept entry: {entry!r}") from e
