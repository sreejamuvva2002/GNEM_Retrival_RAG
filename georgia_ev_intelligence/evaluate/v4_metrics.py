"""
V4 retrieval-pipeline metrics.

Read-only aggregates computed from gev_retrieval_audit and
gev_retrieval_candidates so the evaluation harness can report retrieval-
layer health alongside the existing RAGAS scores.

Three metrics:

  - branch_coverage         — % of audit rows for the run where the
                              selected_interpretations array has length ≥ 2
                              (i.e. ambiguity branching actually fired).

  - validator_rejection_rate — across all candidates seen during the run,
                               the share that were rejected by
                               evidence_validator (hard_filter_passed=false).

  - audit_completeness      — % of audit rows that have non-null values
                              for query_class, extracted_entities and at
                              least one entry in final_evidence.

These are the §L "test_audit_completeness", "test_ambiguity_branches_top_2"
style assertions translated into run-level dashboards.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import func

from shared.db import RetrievalAudit, RetrievalCandidate, get_session
from shared.logger import get_logger

logger = get_logger("evaluate.v4_metrics")


def _safe_division(numerator: float, denominator: float) -> float:
    return float(numerator) / denominator if denominator else 0.0


def compute_run_metrics(run_id: str | None = None) -> dict[str, Any]:
    """
    Compute the three V4 retrieval health metrics.

    Pass run_id="<uuid>" to scope to a single evaluation run; pass None
    to compute against the entire audit history (handy for CI smoke checks).

    Returns a dict of named metrics. Missing tables (migration not yet run)
    surface as zeroes with a logged warning so the evaluator can still
    write the rest of its results.
    """
    try:
        session = get_session()
    except Exception as exc:
        logger.warning("v4_metrics: db unavailable — %s", exc)
        return {"error": "db_unavailable"}

    try:
        q_audit = session.query(RetrievalAudit)
        if run_id:
            q_audit = q_audit.filter(RetrievalAudit.run_id == run_id)
        total_questions = q_audit.count()
        if total_questions == 0:
            return {
                "run_id": run_id,
                "total_questions": 0,
                "branch_coverage": 0.0,
                "validator_rejection_rate": 0.0,
                "audit_completeness": 0.0,
            }

        # branch_coverage: count rows where selected_interpretations length ≥ 2
        branched = 0
        complete = 0
        ambiguous_seen = 0
        for audit in q_audit.all():
            interp = audit.selected_interpretations or []
            if isinstance(interp, list) and len(interp) >= 2:
                branched += 1
            ambiguous = audit.ambiguous_terms or []
            if ambiguous:
                ambiguous_seen += 1
            ev = audit.final_evidence or []
            if (
                audit.query_class
                and audit.extracted_entities
                and (isinstance(ev, list) and len(ev) > 0 or audit.query_class in {"count_query"})
            ):
                complete += 1

        # validator_rejection_rate: across candidate rows for this run
        cand_q = session.query(RetrievalCandidate).join(
            RetrievalAudit,
            RetrievalCandidate.audit_id == RetrievalAudit.id,
        )
        if run_id:
            cand_q = cand_q.filter(RetrievalAudit.run_id == run_id)
        total_candidates = cand_q.count()
        rejected_candidates = (
            cand_q.filter(RetrievalCandidate.hard_filter_passed.is_(False)).count()
            if total_candidates else 0
        )

        # support_level distribution
        support_counts: dict[str, int] = {}
        for audit in q_audit.all():
            key = audit.support_level or "unknown"
            support_counts[key] = support_counts.get(key, 0) + 1

        # query_class distribution
        class_counts_rows = (
            q_audit
            .with_entities(RetrievalAudit.query_class, func.count(RetrievalAudit.id))
            .group_by(RetrievalAudit.query_class)
            .all()
        )
        class_counts = {row[0] or "unknown": int(row[1]) for row in class_counts_rows}

        return {
            "run_id":                    run_id,
            "total_questions":           int(total_questions),
            "branch_coverage":           round(_safe_division(branched, total_questions), 4),
            "branch_coverage_when_ambiguous": round(
                _safe_division(branched, ambiguous_seen) if ambiguous_seen else 0.0, 4,
            ),
            "validator_rejection_rate":  round(_safe_division(rejected_candidates, total_candidates), 4),
            "audit_completeness":        round(_safe_division(complete, total_questions), 4),
            "support_level_distribution": support_counts,
            "query_class_distribution":   class_counts,
            "total_candidates":           int(total_candidates),
            "rejected_candidates":        int(rejected_candidates),
        }
    finally:
        session.close()


def print_run_metrics(run_id: str | None = None) -> None:
    """Convenience CLI helper. `python -m evaluate.v4_metrics <run_id>`."""
    metrics = compute_run_metrics(run_id)
    width = max(len(k) for k in metrics)
    print(f"V4 retrieval metrics for run_id={run_id or '<all>'}")
    print("-" * (width + 16))
    for k, v in metrics.items():
        print(f"  {k:<{width}} : {v}")


if __name__ == "__main__":
    import sys
    print_run_metrics(sys.argv[1] if len(sys.argv) > 1 else None)
