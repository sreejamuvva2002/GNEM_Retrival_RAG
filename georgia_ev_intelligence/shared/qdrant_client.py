"""Shared Qdrant client construction."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

from . import config


def build_client(
    url: str | None = None,
    path: str | None = None,
    api_key: str | None = None,
    timeout: int | None = None,
) -> "QdrantClient":
    from qdrant_client import QdrantClient

    local_path = path if path is not None else config.QDRANT_PATH
    if local_path:
        return QdrantClient(path=local_path)

    return QdrantClient(
        url=url or config.QDRANT_URL,
        api_key=api_key or config.QDRANT_API_KEY or None,
        timeout=timeout or config.QDRANT_TIMEOUT,
    )
