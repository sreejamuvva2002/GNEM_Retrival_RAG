"""Cross-encoder reranking for merged child chunk candidates."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from georgia_ev_intelligence.runtime_pipeline.debug_trace import record_step
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
        return [parent for parent, _score in self.score_parents(query, parents, top_k)]

    def score_parents(
        self,
        query: str,
        parents: list[ParentContext],
        top_k: int | None = None,
    ) -> list[tuple[ParentContext, float]]:
        """Rerank parents and return (parent, cross-encoder score) pairs.

        Same scoring as ``rerank_parents`` but exposes the relevance score so the
        self-healing loop can (a) re-rank the union of sub-query results against
        the original query and (b) surface a retrieval-confidence signal. When
        ``top_k`` is None all scored parents are returned.
        """
        if not parents:
            return []
        if top_k is not None and top_k <= 0:
            return []

        pairs = [(query, parent.parent_chunk_text) for parent in parents]
        raw_scores = self._model.predict(pairs, show_progress_bar=False)
        scores = _flatten_scores(raw_scores)

        scored_parents = sorted(
            zip(parents, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        if top_k is not None:
            scored_parents = scored_parents[:top_k]
        record_step(
            "rerank",
            status="ok",
            summary=(
                f"scored {len(parents)} parents, kept {len(scored_parents)} "
                f"(top_k={top_k}); top_score="
                f"{scored_parents[0][1] if scored_parents else 'n/a'}"
            ),
            details={
                "query": query,
                "top_k": top_k,
                "scored_count": len(parents),
                "kept_count": len(scored_parents),
                "scores": [
                    {"record_id": p.record_id, "score": round(float(s), 4)}
                    for p, s in scored_parents
                ],
            },
        )
        return scored_parents

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
