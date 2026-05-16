"""Data models for vocabulary terms."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class VocabularyTerm:
    """A single vocabulary term extracted from the normalized KB."""

    normalized_value: str
    term_frequency: int
    row_ids: list[int]
    multiple_words: bool
    term_type: str
    source_column: str
    term_vector: "np.ndarray | None" = field(default=None, repr=False)
