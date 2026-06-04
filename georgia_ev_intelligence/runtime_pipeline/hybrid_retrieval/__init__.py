"""Hybrid retrieval module.

All concrete classes require optional heavy dependencies (psycopg2,
sentence-transformers, rank-bm25).  Import them explicitly from their
submodules only when needed to avoid import-time failures in environments
where those packages are not installed.

    from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.factory import (
        build_default_pipeline,
    )
"""
