"""Layer 2 - retrieval quality (modes 2 and 3 only).

A bad answer from a grounded pipeline has two possible causes: the retriever
never surfaced the right record, or the model had it and ignored it. These
metrics separate the two, which is the difference between fixing the index and
fixing the prompt.

Relevance labels are derived automatically: a retrieved record is relevant if
the golden answer names that company. That is a silver label, not a gold one -
it under-counts records that support the answer without being named. Spot-check
it before trusting the absolute numbers; the mode-to-mode comparison is safe.
"""

from __future__ import annotations

import math

from ..data import Doc, normalise_name
from ..evaluation.deterministic import find_entities


def gold_doc_ids(golden_answer: str, docs: list[Doc]) -> set[str]:
    """Records whose company is named in the golden answer."""
    names = [doc.title for doc in docs]
    wanted = find_entities(golden_answer, names)
    return {doc.doc_id for doc in docs if normalise_name(doc.title) in wanted}


def retrieval_scores(retrieved_ids: list[str], gold_ids: set[str], k: int | None = None) -> dict[str, float]:
    if not gold_ids:
        return {
            "recall_at_k": float("nan"),
            "precision_at_k": float("nan"),
            "mrr": float("nan"),
            "ndcg_at_k": float("nan"),
            "gold_docs": 0,
        }
    ranked = retrieved_ids[:k] if k else retrieved_ids
    hits = [doc_id for doc_id in ranked if doc_id in gold_ids]
    recall = len(set(hits)) / len(gold_ids)
    precision = len(hits) / len(ranked) if ranked else 0.0
    mrr = 0.0
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in gold_ids:
            mrr = 1.0 / rank
            break
    return {
        "recall_at_k": recall,
        "precision_at_k": precision,
        "mrr": mrr,
        "ndcg_at_k": _ndcg(ranked, gold_ids),
        "gold_docs": len(gold_ids),
    }


def _ndcg(ranked: list[str], gold_ids: set[str]) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, doc_id in enumerate(ranked, start=1) if doc_id in gold_ids)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold_ids), len(ranked)) + 1))
    return dcg / ideal if ideal else 0.0
