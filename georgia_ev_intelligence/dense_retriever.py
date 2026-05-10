"""
In-memory dense retrieval over the KB DataFrame using sentence-transformers.

Each KB row is embedded as a single string built dynamically from all
non-skipped columns — no column names or data values are hardcoded here.
Cosine similarity is computed via dot product on L2-normalised vectors.
No vector database; everything lives in a numpy float32 matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .embedding_model import as_document_text, as_query_text, load_sentence_transformer


class DenseRetriever:
    """
    Parameters
    ----------
    df         : full KB DataFrame
    model_name : HuggingFace model ID, e.g. "sentence-transformers/all-MiniLM-L6-v2"
    skip_cols  : column names to exclude from row text (pass schema_index.SKIP_COLUMNS)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        model_name: str,
        skip_cols: set[str],
    ) -> None:
        self._df = df.reset_index(drop=True)
        self._skip_cols = skip_cols
        self._model = load_sentence_transformer(model_name)
        self._embeddings: np.ndarray = self._build_embeddings()

    # ── Row text construction ─────────────────────────────────────────────────

    def _row_to_text(self, row: pd.Series) -> str:
        """
        Convert one KB row to a plain-text string for embedding.

        Format: "col_name: value | col_name: value | ..."
        Only non-skipped, non-null columns are included.
        Column selection is fully dynamic — derived from df.columns at runtime.
        """
        parts: list[str] = []
        for col in self._df.columns:
            if col in self._skip_cols:
                continue
            val = row[col]
            if pd.isna(val) or str(val).strip() == "" or str(val).lower() == "nan":
                continue
            parts.append(f"{col}: {val}")
        return " | ".join(parts)

    def _build_embeddings(self) -> np.ndarray:
        """
        Encode all rows. Returns float32 L2-normalised matrix of shape (n_rows, dim).
        L2-normalisation means cosine similarity == dot product — no scipy needed.
        """
        texts = [as_document_text(self._row_to_text(row)) for _, row in self._df.iterrows()]
        embeddings = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    # ── Public search API ─────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 15,
        threshold: float = 0.0,
    ) -> pd.DataFrame:
        """
        Return up to top_k rows with cosine similarity >= threshold.

        Parameters
        ----------
        query     : natural-language query or fragment
        top_k     : maximum rows to return
        threshold : minimum cosine similarity (0–1). Pass 0.0 to skip filtering.

        Returns
        -------
        pd.DataFrame — subset of self._df sorted by similarity descending,
                       with an added '_score' column (float).
        """
        query_vec = self._model.encode(
            [as_query_text(query)],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)[0]

        scores: np.ndarray = self._embeddings @ query_vec

        # Over-fetch then threshold to avoid discarding good results
        n_candidates = min(top_k * 4, len(self._df))
        top_indices = np.argpartition(scores, -n_candidates)[-n_candidates:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        filtered_indices = [i for i in top_indices if scores[i] >= threshold][:top_k]

        if not filtered_indices:
            return self._df.iloc[0:0].copy()

        result = self._df.iloc[filtered_indices].copy()
        result["_score"] = [float(scores[i]) for i in filtered_indices]
        return result
