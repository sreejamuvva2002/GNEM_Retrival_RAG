"""Index-time chunking, embedding, and vector-store helpers."""

from .chunking import build_parent_child_chunks  # noqa: F401


def index_kb_chunks(*args, **kwargs):
    from .qdrant_store import index_kb_chunks as _index_kb_chunks

    return _index_kb_chunks(*args, **kwargs)
