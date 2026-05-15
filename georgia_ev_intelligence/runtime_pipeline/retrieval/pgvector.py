"""pgvector-backed semantic retrieval over child chunks."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import psycopg2

from ...shared import config
from ...shared.embeddings import as_query_text, load_sentence_transformer


_SEARCH_SQL = """
SELECT
    source_row_id,
    chunk_type,
    1 - (embedding <=> %s::vector) AS score
FROM child_chunks
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""


class PgVectorRetriever:
    """
    Search child chunks in pgvector and return their parent KB rows.

    Implements the same SemanticRetriever.search() interface as QdrantRetriever
    so it is a drop-in replacement throughout the pipeline.
    """

    def __init__(self, df: pd.DataFrame, model_name: str) -> None:
        self._df = df.reset_index(drop=True)
        self._model = load_sentence_transformer(model_name)
        self._db_url = config.NEON_DATABASE_URL
        self._rows_by_id: dict[int, dict[str, Any]] = {
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
        )[0].astype(float).tolist()

        # Over-fetch to allow for deduplication and threshold filtering
        fetch_limit = max(top_k * 4, top_k)

        conn = psycopg2.connect(self._db_url)
        try:
            with conn.cursor() as cur:
                cur.execute(_SEARCH_SQL, (query_vec, query_vec, fetch_limit))
                results = cur.fetchall()
        finally:
            conn.close()

        rows: list[dict[str, Any]] = []
        seen: set[int] = set()

        for source_row_id, chunk_type, score in results:
            if threshold > 0 and score < threshold:
                continue
            row_id = _to_int(source_row_id)
            if row_id is None or row_id in seen:
                continue
            row = self._rows_by_id.get(row_id)
            if row is None:
                continue
            scored = dict(row)
            scored["_score"] = float(score)
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
