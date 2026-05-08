"""
Phase 4 — Cross-encoder reranking.

Takes a question plus candidate company rows and uses a local cross-encoder
to produce a better semantic order than vector similarity alone.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from sentence_transformers import CrossEncoder

from shared.config import Config
from shared.logger import get_logger

logger = get_logger("phase4.reranker")


def _candidate_text(company: dict[str, Any]) -> str:
    text = str(company.get("text") or "").strip()
    if text:
        return text

    parts = [
        f"Company: {company.get('company_name', '')}",
        f"Tier: {company.get('tier', '')}",
        f"Role: {company.get('ev_supply_chain_role', '')}",
        f"County: {company.get('location_county', '')}",
        f"Employment: {company.get('employment', '')}",
        f"Primary OEMs: {company.get('primary_oems', '')}",
        f"Industry: {company.get('industry_group', '')}",
        f"EV Relevance: {company.get('ev_battery_relevant', '')}",
        f"Facility: {company.get('facility_type', '')}",
        f"Products/Services: {company.get('products_services', '')}",
        f"Classification: {company.get('classification_method', '')}",
        f"Supplier Affiliation: {company.get('supplier_affiliation_type', '')}",
    ]
    return " | ".join(parts)


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    cfg = Config.get()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading cross-encoder reranker %s on %s", cfg.reranker_model, device)
    return CrossEncoder(cfg.reranker_model, device=device)


def rerank_companies(question: str, companies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Reorder candidate company rows using a cross-encoder score.

    PRECONDITION: candidates must already have passed evidence_validator
    (i.e. `hard_filter_passed=True` on the underlying Candidate). The
    reranker is a *ranking* layer, not a filter — it cannot resurrect a row
    that hard-filter validation rejected. Callers that pass raw rows
    (legacy V3 path) bypass this gate and accept the responsibility.

    Returns the original order unchanged if reranking fails.

    Output rows include `_reranker_score` (raw cross-encoder score) and
    `_score_breakdown` (an audit-friendly dict carrying the score under the
    'reranker' key) so retrieval_fusion / audit_logger can record the
    contribution.
    """
    if len(companies) <= 1:
        return companies

    # Defensive precondition: drop anything explicitly marked as failed
    # validation. Rows without the field are treated as validated (legacy
    # callers may not set it).
    eligible = [c for c in companies if c.get("hard_filter_passed", True)]
    rejected_count = len(companies) - len(eligible)
    if rejected_count:
        logger.info(
            "reranker: skipped %d candidate(s) with hard_filter_passed=False",
            rejected_count,
        )
    if not eligible:
        return []

    try:
        model = _get_reranker()
        pairs = [(question, _candidate_text(company)) for company in eligible]
        scores = model.predict(pairs, show_progress_bar=False)

        rescored = []
        for company, score in zip(eligible, scores):
            enriched = company.copy()
            score_f = float(score)
            enriched["_reranker_score"] = score_f
            breakdown = dict(enriched.get("_score_breakdown") or {})
            breakdown["reranker"] = score_f
            enriched["_score_breakdown"] = breakdown
            rescored.append(enriched)

        rescored.sort(key=lambda item: item["_reranker_score"], reverse=True)
        logger.info("Cross-encoder reranked %d candidates", len(rescored))
        return rescored
    except Exception as exc:
        logger.warning("Cross-encoder reranking failed, keeping dense order: %s", exc)
        return eligible
