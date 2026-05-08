"""
Phase 4 — Deterministic KB query planner (legacy V3 path).

WHY THIS FILE STILL EXISTS
--------------------------
The V4 retrieval pipeline (pipeline.py::EVAgent.ask) does NOT call this
planner. V4 builds plans through ambiguity_resolver._build_plans_for_class
based on the QueryClass returned by query_classifier. This module is kept
as the entry point for the V3 retrieve_context() fallback in
vector_retriever.py and for any tooling that still consumes KBQueryPlan.

WHAT CHANGED IN THE REFACTOR
----------------------------
The previous version contained 15+ `if "<phrase>" in question` branches
that pinned a specific retrieval mode to a specific golden-question
wording (product, role, and industry phrases). Every one of those
branches has been deleted because:

  1. Each branch encoded a domain fact (a phrase implies a specific KB
     keyword set) that was never reviewed against the
     gev_domain_mapping_rules approval workflow.

  2. A re-phrased version of the same eval question silently failed to
     match the branch and fell through to a different retrieval mode,
     producing brittle pass/fail behaviour against the 50-question set.

The remaining planner only handles modes that key off entity-extracted
flags or generic English question shapes (top-N, SPOF, classification
list). Anything more specific (product term routing, area set
differences) belongs in either:
  - the V4 pipeline (where ambiguity_resolver branches over KB-supported
    interpretations), or
  - an approved row in gev_domain_mapping_rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from filters_and_validation.query_entity_extractor import Entities


@dataclass(frozen=True)
class KBQueryPlan:
    mode: str = "semantic"
    keywords: list[str] = field(default_factory=list)
    group_by: str | None = None
    limit: int | None = None
    # `query_class` is the enum-string value attached by the new classifier;
    # preserved for the audit log even when the legacy V3 retriever runs.
    query_class: str = ""

    @property
    def deterministic(self) -> bool:
        return self.mode != "semantic"


def _classify_for_audit(question: str, entities: Entities) -> str:
    """Run the deterministic classifier purely for audit annotation."""
    try:
        from filters_and_validation.query_classifier import classify
        return classify(question, entities).value
    except Exception:
        return ""


def build_kb_query_plan(question: str, entities: Entities) -> KBQueryPlan:
    """Public entry: thin wrapper that annotates the legacy plan with the new query_class."""
    plan = _build_kb_query_plan_inner(question, entities)
    qclass = _classify_for_audit(question, entities)
    if qclass and plan.query_class != qclass:
        from dataclasses import replace
        plan = replace(plan, query_class=qclass)
    return plan


def _build_kb_query_plan_inner(question: str, entities: Entities) -> KBQueryPlan:
    """
    Pick the V3 retrieval mode from entity-driven signals only.

    All keyword/phrase-keyed routing has been removed. If the entities and
    generic English question shape are insufficient to pick a deterministic
    mode, return the default semantic plan and let the retriever fall back
    to vector ranking. The V4 pipeline handles richer cases via the
    QueryClass + ambiguity branches.
    """
    q = question.lower()

    if "highest employment" in q and entities.county:
        return KBQueryPlan(mode="highest_employment", limit=1)

    if entities.is_top_n:
        return KBQueryPlan(mode="top_employment", limit=entities.top_n_limit)

    if entities.is_risk_query:
        return KBQueryPlan(mode="single_supplier_roles")

    if entities.classification_method and "classified" in q:
        return KBQueryPlan(mode="structured_list")

    if entities.ev_role or entities.ev_role_list:
        list_intent_phrases = (
            "classified under",
            "classified as",
            "list every",
            "list all",
            "show all",
            "map all",
        )
        if any(phrase in q for phrase in list_intent_phrases):
            return KBQueryPlan(mode="role_list")

    if entities.tier or entities.tier_list:
        tier_list_intent = (
            "list every",
            "list all",
            "show all",
            "map all",
        )
        if any(phrase in q for phrase in tier_list_intent):
            return KBQueryPlan(mode="tier_list")

    return KBQueryPlan()
