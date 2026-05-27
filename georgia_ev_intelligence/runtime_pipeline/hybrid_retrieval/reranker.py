"""Cross-encoder reranking for parent chunk candidates.

WHY THIS FILE EXISTS
--------------------
After BM25+dense retrieval fetches many child chunks and they are mapped to
parent records, there are often more parent chunks than we want to pass to the
LLM (a large context degrades LLM accuracy).  This module uses a cross-encoder
model to score each parent chunk against the original query and keep only the
most relevant top-K.

TECHNIQUE: Cross-Encoder Reranking
------------------------------------
- A bi-encoder (like BERT/SBERT) embeds queries and documents independently;
  a cross-encoder instead feeds the query+document pair together in one forward
  pass, giving much more accurate relevance scores at the cost of speed.
- Model: ``cross-encoder/ms-marco-MiniLM-L12-v2`` (fast, passage-level reranking).
- Implemented via ``sentence_transformers.CrossEncoder``.

ACTIVE RERANKING PATH
---------------------
**Parent-level reranking is the active path.**  ``rerank_parents()`` scores
each parent chunk text against the original query and keeps ``top_k`` parents.
These are exactly the chunks whose text is concatenated and passed to the LLM.

``rerank()`` (child-level) exists for experimental use but is NOT called in the
standard pipeline.  Child-level reranking was disabled in favour of mapping all
deduped children to parents first, then reranking parents — this ensures the LLM
always sees complete company records rather than partial child snippets.

CORRECTNESS CONTRACT
--------------------
- The reranker is called with ``query = original_question`` (not a rewritten
  variant).  Reranking anchors to the user's exact question.
- ``_flatten_scores`` handles both scalar and sequence score outputs so the
  model can return numpy arrays, Python lists, or nested sequences.
- Lazy model loading: ``CrossEncoder`` is imported and instantiated on first use
  to avoid slow startup when running analysis-only commands.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from georgia_ev_intelligence.runtime_pipeline.retrieval.bm25_retriever import (
    _build_bm25_text,
)
from georgia_ev_intelligence.runtime_pipeline.schemas import (
    ParentContext,
    RetrievedChildChunk,
)

from .config import RERANKER_MODEL
from .models import RerankedChildChunk


class CrossEncoderReranker:
    """Score child or parent candidates with a sentence-transformers cross encoder."""

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        model: Any | None = None,
        child_text_builder: Callable[[RetrievedChildChunk], str] | None = None,
    ) -> None:
        self._model_name = model_name
        self._model = model or self._load_model(model_name)
        self._child_text_builder = child_text_builder or _default_child_text

    def rerank(
        self,
        query: str,
        children: list[RetrievedChildChunk],
        top_k: int,
    ) -> list[RerankedChildChunk]:
        if top_k <= 0 or not children:
            return []

        pairs = [(query, self._child_text_builder(child)) for child in children]
        raw_scores = self._model.predict(pairs, show_progress_bar=False)
        scores = _flatten_scores(raw_scores)

        scored_children = sorted(
            zip(children, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )

        reranked: list[RerankedChildChunk] = []
        for rank, (child, score) in enumerate(scored_children[:top_k], start=1):
            reranked.append(RerankedChildChunk(
                child=child,
                rerank_score=float(score),
                rank=rank,
            ))
        return reranked

    def rerank_parents(
        self,
        query: str,
        parents: list[ParentContext],
        top_k: int,
    ) -> list[ParentContext]:
        """Rerank unique parent chunks and return the final LLM context set.

        This is the active runtime path. Parent-level reranking is intentional:
        the LLM receives parent_chunk_text values, not child chunk metadata.
        """
        if top_k <= 0 or not parents:
            return []

        pairs = [(query, parent.parent_chunk_text) for parent in parents]
        raw_scores = self._model.predict(pairs, show_progress_bar=False)
        scores = _flatten_scores(raw_scores)

        scored_parents = sorted(
            zip(parents, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        return [parent for parent, _score in scored_parents[:top_k]]

    @staticmethod
    def _load_model(model_name: str) -> Any:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(model_name)


def _default_child_text(child: RetrievedChildChunk) -> str:
    return _build_bm25_text(child.chunk_type, child.metadata)


def _flatten_scores(raw_scores: Any) -> list[float]:
    if hasattr(raw_scores, "tolist"):
        values = raw_scores.tolist()
    else:
        values = list(raw_scores)

    return [_as_float(value) for value in values]


def _as_float(value: Any) -> float:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return float(value[0])
    return float(value)
