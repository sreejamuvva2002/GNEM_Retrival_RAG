"""Retriever protocol definitions for structural typing.

Defines the ChildRetriever protocol so new retrievers can be added
without modifying existing concrete classes. DensePgvectorRetriever
and BM25Retriever satisfy this protocol structurally.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..schemas import RetrievedChildChunk


@runtime_checkable
class ChildRetriever(Protocol):
    """Protocol for any retriever that returns child-level results.

    DensePgvectorRetriever and BM25Retriever satisfy this protocol
    structurally without modification (their search() signatures match).
    """

    def search(self, query: str, top_k: int = 100) -> list[RetrievedChildChunk]:
        """Search for child chunks matching the query."""
        ...
