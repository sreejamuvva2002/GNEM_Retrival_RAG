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

    Returns the original order unchanged if reranking fails.
    """
    if len(companies) <= 1:
        return companies

    try:
        model = _get_reranker()
        pairs = [(question, _candidate_text(company)) for company in companies]
        scores = model.predict(pairs, show_progress_bar=False)

        rescored = []
        for company, score in zip(companies, scores):
            enriched = company.copy()
            enriched["_reranker_score"] = float(score)
            rescored.append(enriched)

        rescored.sort(key=lambda item: item["_reranker_score"], reverse=True)
        logger.info("Cross-encoder reranked %d candidates", len(rescored))
        return rescored
    except Exception as exc:
        logger.warning("Cross-encoder reranking failed, keeping dense order: %s", exc)
        return companies
