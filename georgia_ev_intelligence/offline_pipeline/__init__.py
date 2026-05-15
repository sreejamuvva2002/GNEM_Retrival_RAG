"""Offline indexing pipeline."""

from importlib import import_module

_EXPORTS = {
    "build_parent_child_chunks": (
        "georgia_ev_intelligence.offline_pipeline.chunking.operations"
    ),
    "index_kb_children": "georgia_ev_intelligence.offline_pipeline.pgvector_store",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        module = import_module(_EXPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
