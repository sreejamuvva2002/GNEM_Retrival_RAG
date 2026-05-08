"""
Phase 4 — Structured retrieval audit logger.

Writes one row per question to gev_retrieval_audit and one row per
candidate (selected and rejected) to gev_retrieval_candidates. The audit
log is the *only* place where retrieval provenance is reproducible after
the fact: the in-memory pipeline objects are discarded once the answer is
returned to the user.

Defensive behaviour: if the audit tables are missing (migration not run
yet), the function logs a warning and returns None instead of crashing
the pipeline. Run `python scripts/create_audit_tables.py` to create them.

CRITICAL: This module never writes to gev_domain_mapping_rules. It only
records *suggestions* in the audit row's `synonym_mappings` field. Rules
are added exclusively via human approval out-of-band.
"""
from __future__ import annotations

import json
from typing import Any

from core_agent.retrieval_types import AuditRecord, Candidate
from shared.db import RetrievalAudit, RetrievalCandidate, get_session
from shared.logger import get_logger

logger = get_logger("phase4.audit_logger")


def _serialise(value: Any) -> Any:
    """JSON-safe coerce — drops sets, dataclasses, and other non-trivial types."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {k: _serialise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialise(v) for v in value]
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if hasattr(value, "__dict__"):
        return {k: _serialise(v) for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def _candidate_row_dict(c: Candidate) -> dict[str, Any]:
    """Compact JSON-friendly view of a Candidate for the audit table."""
    return {
        "canonical_name":      c.canonical_name,
        "company_row_id":      c.company_row_id,
        "sources":             sorted(c.sources),
        "scores":              {k: round(v, 4) for k, v in c.scores.items()},
        "fused_score":         round(c.fused_score, 4),
        "hard_filter_passed":  c.hard_filter_passed,
        "rejection_reason":    c.rejection_reason,
        "final_selected":      c.final_selected,
    }


def write_audit(
    record: AuditRecord,
    candidates: list[Candidate],
) -> int | None:
    """
    Persist the audit record and per-candidate rows.

    Returns the inserted audit row's id, or None when the tables are
    missing or the write fails (we never let an audit failure break the
    user-facing answer).
    """
    try:
        session = get_session()
    except Exception as exc:
        logger.warning("audit_logger: database unavailable — %s", exc)
        return None

    try:
        audit = RetrievalAudit(
            run_id=record.run_id,
            question=record.question,
            query_class=record.query_class,
            extracted_entities=_serialise(record.extracted_entities),
            hard_filters=_serialise(record.hard_filters),
            ambiguous_terms=_serialise(record.ambiguous_terms),
            selected_interpretations=_serialise(record.selected_interpretations),
            synonym_mappings=_serialise(record.synonym_mappings),
            retrieval_methods_used=list(record.retrieval_methods_used or []),
            sql_query=record.sql_query,
            cypher_query=record.cypher_query,
            qdrant_dense_query=record.qdrant_dense_query,
            qdrant_sparse_query=record.qdrant_sparse_query,
            final_evidence=_serialise(record.final_evidence),
            answer_text=record.answer_text,
            support_level=record.support_level,
            hallucination_risk=record.hallucination_risk,
            audit_comment=record.audit_comment,
            elapsed_ms=record.elapsed_ms,
        )
        session.add(audit)
        session.flush()  # populate audit.id without committing yet

        for cand in candidates:
            row = RetrievalCandidate(
                audit_id=audit.id,
                branch_id=getattr(cand, "branch_id", None) or "A",
                company_row_id=cand.company_row_id,
                canonical_name=cand.canonical_name,
                sources=sorted(cand.sources),
                scores=_serialise(cand.scores),
                fused_score=cand.fused_score,
                hard_filter_passed=cand.hard_filter_passed,
                rejection_reason=cand.rejection_reason,
                final_selected=cand.final_selected,
            )
            session.add(row)

        session.commit()
        logger.info(
            "audit %s class=%s methods=%s evidence=%d risk=%d",
            audit.id,
            record.query_class,
            ",".join(record.retrieval_methods_used or []),
            len(record.final_evidence or []),
            record.hallucination_risk,
        )
        return int(audit.id)
    except Exception as exc:
        session.rollback()
        # Most likely: tables missing. Log and continue.
        logger.warning("audit_logger: write failed — %s", exc)
        return None
    finally:
        session.close()


def serialise_candidate_compact(c: Candidate) -> dict[str, Any]:
    """Public helper for callers building AuditRecord.final_evidence."""
    return _candidate_row_dict(c)
