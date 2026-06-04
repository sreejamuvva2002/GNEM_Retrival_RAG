"""Narrow interfaces used by the isolated hybrid retrieval orchestrator.

WHY THIS FILE EXISTS
--------------------
Defines ``typing.Protocol`` interfaces (structural sub-typing) that decouple
the orchestrator and pipeline classes from their concrete implementations.
This allows unit-testing with mocks, swapping retrievers/rerankers, and
makes the dependency graph explicit without tight coupling.

PROTOCOLS DEFINED
-----------------
``ChildRetriever``
    Anything that can ``retrieve(query, top_k) -> list[RetrievedChildChunk]``.
    Implemented by ``BM25ChildRetriever`` and ``DenseChildRetriever``.

``ParentReranker``
    Anything that can ``rerank_parents(query, parents, top_k) -> list[ParentContext]``.
    Implemented by ``CrossEncoderReranker``.

``PromptBuilder``
    Anything that builds a prompt from a question + retrieved context string.
    Used by ``RagOnlyAnswerPipeline`` and ``HybridRagAnswerPipeline``.

``NoContextPromptBuilder``
    Like ``PromptBuilder`` but no context argument — for pipelines that
    intentionally receive no retrieved text (``OnlyPretrainedAnswerPipeline``).

``ContextualAnswerPipeline``
    End-to-end pipeline that takes a question + retrieved_context and returns
    an answer string.  Covers ``rag_only`` and ``hybrid_rag`` pipelines.

``NonContextualAnswerPipeline``
    End-to-end pipeline that takes only a question and returns an answer.
    Covers the ``pretrained_only`` pipeline.

WHY PROTOCOLS INSTEAD OF ABSTRACT BASE CLASSES
-----------------------------------------------
Python Protocols support structural (duck-type) sub-typing, meaning classes do
not need to explicitly inherit from the protocol — they just need to implement
the matching method signatures.  This keeps concrete classes self-contained and
avoids deep inheritance trees.
"""
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
