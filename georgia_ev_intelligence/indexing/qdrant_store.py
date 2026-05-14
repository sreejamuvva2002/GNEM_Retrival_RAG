"""Index parent-child KB chunks into Qdrant."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models

from .. import config
from .chunking import ChildChunk, build_parent_child_chunks
from .embeddings import as_document_text, load_sentence_transformer


@dataclass(frozen=True)
class QdrantIndexStats:
    collection_name: str
    chunks_indexed: int
    vector_size: int
    embedding_model: str


def build_client(
    url: str | None = None,
    path: str | None = None,
    api_key: str | None = None,
    timeout: int | None = None,
) -> QdrantClient:
    local_path = path if path is not None else config.QDRANT_PATH
    if local_path:
        return QdrantClient(path=local_path)

    return QdrantClient(
        url=url or config.QDRANT_URL,
        api_key=api_key or config.QDRANT_API_KEY or None,
        timeout=timeout or config.QDRANT_TIMEOUT,
    )


def index_kb_chunks(
    df: pd.DataFrame,
    *,
    collection_name: str | None = None,
    model_name: str | None = None,
    recreate: bool = False,
    batch_size: int | None = None,
    client: QdrantClient | None = None,
) -> QdrantIndexStats:
    model_id = model_name or config.EMBEDDING_MODEL
    collection = collection_name or config.QDRANT_COLLECTION
    size = batch_size or config.QDRANT_BATCH_SIZE

    model = load_sentence_transformer(model_id)
    if hasattr(model, "get_embedding_dimension"):
        vector_size = int(model.get_embedding_dimension())
    else:
        vector_size = int(model.get_sentence_embedding_dimension())
    qdrant = client or build_client()

    _ensure_collection(
        qdrant,
        collection_name=collection,
        vector_size=vector_size,
        recreate=recreate,
    )

    chunks = build_parent_child_chunks(df)
    for batch in _batched(chunks, size):
        texts = [as_document_text(chunk.embedding_text) for chunk in batch]
        vectors = model.encode(
            texts,
            batch_size=size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        points = [
            models.PointStruct(
                id=_point_id(chunk),
                vector=vectors[i].astype(float).tolist(),
                payload=chunk.payload(),
            )
            for i, chunk in enumerate(batch)
        ]
        qdrant.upsert(collection_name=collection, points=points, wait=True)

    return QdrantIndexStats(
        collection_name=collection,
        chunks_indexed=len(chunks),
        vector_size=vector_size,
        embedding_model=model_id,
    )


def _ensure_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    vector_size: int,
    recreate: bool,
) -> None:
    exists = client.collection_exists(collection_name)
    if recreate and exists:
        client.delete_collection(collection_name)
        exists = False

    if exists:
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


def _batched(chunks: list[ChildChunk], size: int) -> Iterable[list[ChildChunk]]:
    for start in range(0, len(chunks), size):
        yield chunks[start : start + size]


def _point_id(chunk: ChildChunk) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"georgia-ev-kb:{chunk.chunk_id}"))
