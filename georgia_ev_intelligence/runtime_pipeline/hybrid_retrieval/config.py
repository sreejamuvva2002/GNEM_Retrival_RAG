"""Configuration for the isolated hybrid retrieval module.

WHY THIS FILE EXISTS
--------------------
Centralises all retrieval and reranking hyper-parameters in one place so they
can be tuned or overridden via environment variables without changing code.
This separation keeps the orchestrator and individual components free of
hard-coded magic numbers.

CONFIGURABLE PARAMETERS
------------------------
``HYBRID_RETRIEVER_TOP_K``  (default 250)
    Maximum child chunks fetched per retriever (BM25 + dense) per query.
    For multi-query retrieval each individual query still uses this limit,
    but the parameter is currently not used in the multi-query path which
    accepts an explicit ``per_query_top_k=150`` argument.

``HYBRID_RERANKER_TOP_K``  (default 45)
    Maximum parent chunks kept after cross-encoder reranking.  These are
    exactly the chunks sent as context to the LLM.

``HYBRID_RERANKER_MODEL``  (default ``cross-encoder/ms-marco-MiniLM-L12-v2``)
    HuggingFace model ID for the sentence-transformers CrossEncoder.
    This model was trained on MS-MARCO passage retrieval and generalises
    well to Georgia EV supply chain domain queries.

USAGE
-----
``HybridRetrievalConfig`` is a frozen dataclass that bundles all three values
and is accepted by ``HybridRetrievalOrchestrator``.  To override at runtime::

    import os
    os.environ["HYBRID_RERANKER_TOP_K"] = "60"
    from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.config import HybridRetrievalConfig
    cfg = HybridRetrievalConfig()   # reranker_top_k == 60
"""
from __future__ import annotations

import os
from dataclasses import dataclass


RETRIEVER_TOP_K = int(os.environ.get("HYBRID_RETRIEVER_TOP_K", "250"))
RERANKER_TOP_K = int(os.environ.get("HYBRID_RERANKER_TOP_K", "45"))
RERANKER_MODEL = os.environ.get(
    "HYBRID_RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L12-v2",
)


@dataclass(frozen=True)
class HybridRetrievalConfig:
    """Runtime knobs for child retrieval and reranking."""

    retriever_top_k: int = RETRIEVER_TOP_K
    reranker_top_k: int = RERANKER_TOP_K
    reranker_model: str = RERANKER_MODEL
