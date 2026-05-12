"""Georgia EV Intelligence package.

The project has moved from a flat module layout into responsibility-focused
subpackages. These aliases keep older import paths working without keeping
duplicate source files.
"""

from __future__ import annotations

import sys
from importlib import import_module

_LEGACY_MODULE_ALIASES = {
    "kb_loader": "georgia_ev_intelligence.data.loader",
    "schema_index": "georgia_ev_intelligence.data.schema",
    "kb_chunker": "georgia_ev_intelligence.indexing.chunker",
    "embedding_model": "georgia_ev_intelligence.indexing.embeddings",
    "qdrant_store": "georgia_ev_intelligence.indexing.qdrant_store",
    "dense_retriever": "georgia_ev_intelligence.retrieval.dense",
    "qdrant_retriever": "georgia_ev_intelligence.retrieval.qdrant",
    "semantic_retriever": "georgia_ev_intelligence.retrieval.semantic",
    "rag_retriever": "georgia_ev_intelligence.retrieval.rag",
    "evidence_selector": "georgia_ev_intelligence.retrieval.evidence",
    "query_rewriter": "georgia_ev_intelligence.query.rewriter",
    "keyword_resolver": "georgia_ev_intelligence.query.keyword_resolver",
    "term_matcher": "georgia_ev_intelligence.query.term_matcher",
    "operation_detector": "georgia_ev_intelligence.query.operation_detector",
    "kb_term_extractor": "georgia_ev_intelligence.query.kb_term_extractor",
    "retriever": "georgia_ev_intelligence.reasoning.retriever",
    "synthesizer": "georgia_ev_intelligence.generation.synthesizer",
    "evaluator": "georgia_ev_intelligence.evaluation.evaluator",
}

for legacy_name, new_name in _LEGACY_MODULE_ALIASES.items():
    module = import_module(new_name)
    sys.modules[f"{__name__}.{legacy_name}"] = module
    globals()[legacy_name] = module
