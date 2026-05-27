"""Narrow interfaces used by the isolated hybrid retrieval orchestrator."""
from __future__ import annotations

from typing import Protocol

from georgia_ev_intelligence.runtime_pipeline.schemas import (
    ParentContext,
    RetrievedChildChunk,
)


class ChildRetriever(Protocol):
    """Retrieve child chunks for a query."""

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChildChunk]:
        """Return at most top_k child chunks."""


class ParentReranker(Protocol):
    """Rerank deduplicated parent chunks for a query."""

    def rerank_parents(
        self,
        query: str,
        parents: list[ParentContext],
        top_k: int,
    ) -> list[ParentContext]:
        """Return the top reranked parent chunks."""


class PromptBuilder(Protocol):
    """Build a prompt from a question and retrieved context."""

    def build(self, question: str, retrieved_context: str) -> str:
        """Return the prompt sent to the LLM."""


class NoContextPromptBuilder(Protocol):
    """Build a prompt from a question only (no retrieved context)."""

    def build(self, question: str) -> str:
        """Return the prompt sent to the LLM."""


class ContextualAnswerPipeline(Protocol):
    """Generate an answer from a question and retrieved context."""

    def answer(
        self,
        question: str,
        retrieved_context: str,
        timeout: int = 180,
    ) -> str:
        """Return the generated answer."""


class NonContextualAnswerPipeline(Protocol):
    """Generate an answer from only a question (no retrieved context)."""

    def answer(self, question: str, timeout: int = 180) -> str:
        """Return the generated answer."""
