"""Qdrant-backed semantic retrieval over child chunks."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

from ...shared import config
from ...shared.embeddings import as_query_text, load_sentence_transformer
from ...shared.qdrant_client import build_client


class QdrantRetriever:
    """
    Search child chunks in Qdrant and return their parent KB rows.

    This implements the shared SemanticRetriever.search() interface used by
    every vector-search call site in the RAG pipeline.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        model_name: str,
        collection_name: str | None = None,
        client: "QdrantClient | None" = None,
    ) -> None:
        self._df = df.reset_index(drop=True)
        self._collection_name = collection_name or config.QDRANT_COLLECTION
        self._client = client or build_client()
        self._model = load_sentence_transformer(model_name)
        self._rows_by_id = {
            int(row["_row_id"]): row.to_dict()
            for _, row in self._df.iterrows()
            if "_row_id" in row and pd.notna(row["_row_id"])
        }

    def search(
        self,
        query: str,
        top_k: int = 15,
        threshold: float = 0.0,
    ) -> pd.DataFrame:
        query_vec = self._model.encode(
            [as_query_text(query)],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vec.astype(float).tolist(),
            limit=max(top_k * 4, top_k),
            with_payload=True,
            with_vectors=False,
            score_threshold=threshold if threshold > 0 else None,
        )

        rows: list[dict[str, Any]] = []
        seen: set[int] = set()

        for point in response.points:
            payload = point.payload or {}
            row_id = _to_int(payload.get("source_row_id"))
            if row_id is None or row_id in seen:
                continue
            row = self._rows_by_id.get(row_id)
            if row is None:
                continue
            scored = dict(row)
            scored["_score"] = float(point.score)
            rows.append(scored)
            seen.add(row_id)
            if len(rows) >= top_k:
                break

        if not rows:
            return self._df.iloc[0:0].copy()

        return pd.DataFrame(rows).reset_index(drop=True)


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
