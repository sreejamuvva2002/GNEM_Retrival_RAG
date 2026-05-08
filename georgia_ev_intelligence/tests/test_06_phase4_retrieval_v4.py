"""
Phase 4 V4 retrieval pipeline regressions.

These tests cover the deterministic parts of the new retrieval layer that
do not need a running Postgres / Qdrant / Neo4j:

  - query_classifier.classify    → correct QueryClass for golden questions
  - synonym_expander.resolve     → KB-grounded mappings, top-2 branching
  - ambiguity_resolver.branches  → branch overlay + plan composition
  - retrieval_fusion.merge/fuse  → dedup, SQL-authority overrides, scoring
  - evidence_validator.verify_answer → flags hallucinated names / numbers
  - formatters.format_branched_answer → two-section template

DB / Qdrant / Neo4j integration tests live elsewhere — those need live
services and are run only in eval contexts.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Stub Entities used to drive the classifier / branchings without DB ───────

class _StubEntities:
    """Minimal Entities-shaped stub — the classifier reads only flags + lists."""
    def __init__(self, **overrides):
        defaults = dict(
            tier=None, tier_list=[], county=None, oem=None, oem_list=[],
            company_name=None, ev_role=None, ev_role_list=[],
            exclude_ev_role_list=[], facility_type=None, industry_group=None,
            classification_method=None, supplier_affiliation_type=None,
            min_employment=None, max_employment=None, product_keywords=[],
            is_aggregate=False, is_risk_query=False, is_top_n=False,
            top_n_limit=10, ev_relevant_filter=False, ev_relevance_value=None,
            residual_abstract_terms=[],
        )
        defaults.update(overrides)
        for k, v in defaults.items():
            setattr(self, k, v)


class TestQueryClassifier(unittest.TestCase):
    def setUp(self):
        from filters_and_validation.query_classifier import classify
        from core_agent.retrieval_types import QueryClass
        self.classify = classify
        self.QC = QueryClass

    def test_tier_filter_is_exact_filter(self):
        e = _StubEntities(tier="Tier 1/2")
        qc = self.classify("List all Tier 1/2 suppliers in Georgia.", e)
        self.assertEqual(qc, self.QC.EXACT_FILTER)

    def test_top_n_overrides_aggregate(self):
        e = _StubEntities(is_top_n=True, top_n_limit=5)
        qc = self.classify("Top 5 Georgia EV companies by employment", e)
        self.assertEqual(qc, self.QC.TOP_N)

    def test_county_aggregate(self):
        e = _StubEntities(is_aggregate=True, county="Gwinnett")
        qc = self.classify(
            "In Gwinnett County, which company has the highest Employment?",
            e,
        )
        self.assertEqual(qc, self.QC.AGGREGATE)

    def test_oem_network(self):
        e = _StubEntities(oem="hyundai", oem_list=["hyundai"])
        qc = self.classify("Show the supplier network linked to Hyundai Motor Group", e)
        self.assertEqual(qc, self.QC.NETWORK)

    def test_risk_query(self):
        e = _StubEntities(is_risk_query=True)
        qc = self.classify("Which single-supplier dependencies pose the highest risk?", e)
        self.assertEqual(qc, self.QC.RISK)

    def test_count_query(self):
        e = _StubEntities(tier="Tier 1")
        qc = self.classify("How many Tier 1 companies are there?", e)
        self.assertEqual(qc, self.QC.COUNT)

    def test_product_capability(self):
        e = _StubEntities(tier="Tier 1/2", product_keywords=["electrolytes"])
        qc = self.classify(
            "Find Tier 1 or Tier 1/2 suppliers capable of producing battery electrolytes",
            e,
        )
        self.assertEqual(qc, self.QC.PRODUCT_CAPABILITY)

    def test_ambiguous_small_scale(self):
        # No is_capacity_risk flag any more — abstract routing is driven
        # by entities.residual_abstract_terms alone (populated by the
        # generic noun-phrase pass in the entity extractor).
        e = _StubEntities(
            oem="hyundai", oem_list=["hyundai"],
            residual_abstract_terms=["small scale"],
        )
        qc = self.classify("Find small-scale suppliers linked to <oem>", e)
        self.assertEqual(qc, self.QC.AMBIGUOUS_SEMANTIC)

    def test_growth_opportunity_is_ambiguous(self):
        e = _StubEntities(residual_abstract_terms=["growth opportunities"])
        qc = self.classify(
            "Which Georgia counties have the fastest growing EV supply chains?",
            e,
        )
        self.assertEqual(qc, self.QC.AMBIGUOUS_SEMANTIC)

    def test_fallback_for_bare_question(self):
        e = _StubEntities()
        qc = self.classify("Tell me about EV manufacturing", e)
        # Manufacturing keyword puts this on PRODUCT_CAPABILITY (via product hints).
        self.assertIn(qc, {self.QC.PRODUCT_CAPABILITY, self.QC.FALLBACK_SEMANTIC})


class TestSynonymExpander(unittest.TestCase):
    """
    The new policy: resolve() returns mappings ONLY from the direct schema
    match path or from approved gev_domain_mapping_rules. There is no
    heuristic fallback — abstract terms with no approved rule are returned
    as 'unresolved' for ambiguity_resolver to handle (or to drop).
    """

    def setUp(self):
        # Stub KB schema lookups so the test does not hit Postgres.
        import filters_and_validation.synonym_expander as se
        se._all_product_phrases = lambda: {"lithium battery", "electrolyte"}
        se._value_set = lambda col: {
            "tier": {"tier 1", "tier 2", "tier 2/3"},
            "facility_type": {"r&d"},
        }.get(col, set())
        # Pretend the rule store has nothing approved for these tests.
        se._rule_mappings_for = lambda *a, **k: []
        from filters_and_validation.synonym_expander import resolve
        self.resolve = resolve

    def test_small_scale_unresolved_without_approved_rule(self):
        # "small scale" is not a KB column, not a KB value, not an approved
        # rule. The expander must NOT invent a mapping — it returns the term
        # as unresolved so the audit log captures the lack of grounding.
        e = _StubEntities()
        out = self.resolve(["small scale"], e, "Find small-scale suppliers")
        self.assertEqual(len(out), 1)
        rt = out[0]
        self.assertEqual(rt.status, "unresolved")
        self.assertEqual(rt.selected_policy, "drop")
        self.assertEqual(rt.candidate_mappings, [])

    def test_kb_value_is_resolved_directly(self):
        # "tier 1" is a literal KB value — direct schema match returns
        # status="direct" with high confidence.
        e = _StubEntities()
        out = self.resolve(["tier 1"], e, "Find tier 1 suppliers")
        self.assertEqual(out[0].status, "direct")
        self.assertEqual(out[0].selected_policy, "apply")

    def test_unknown_term_is_unresolved(self):
        e = _StubEntities()
        out = self.resolve(["xyzzy nonsense"], e, "Find xyzzy nonsense")
        self.assertEqual(out[0].status, "unresolved")
        self.assertEqual(out[0].candidate_mappings, [])


class TestAmbiguityResolver(unittest.TestCase):
    def setUp(self):
        import filters_and_validation.synonym_expander as se
        se._all_product_phrases = lambda: set()
        se._value_set = lambda col: {
            "tier": {"tier 1", "tier 2", "tier 2/3"},
            "facility_type": {"r&d"},
        }.get(col, set())
        se._rule_mappings_for = lambda *a, **k: []
        from filters_and_validation.synonym_expander import resolve
        from filters_and_validation.ambiguity_resolver import branches
        from core_agent.retrieval_types import QueryClass
        self.resolve = resolve
        self.branches = branches
        self.QC = QueryClass

    def test_unambiguous_question_yields_one_branch(self):
        e = _StubEntities(tier="Tier 1/2")
        bs = self.branches(e, [], self.QC.EXACT_FILTER, "List Tier 1/2 suppliers")
        self.assertEqual(len(bs), 1)
        self.assertEqual(bs[0].branch_id, "A")
        self.assertIsNotNone(bs[0].sql_plan)
        self.assertEqual(bs[0].sql_plan.mode, "filter")

    def test_unresolved_term_with_kb_columns_yields_two_branches(self):
        # An unresolved term whose substring appears in two hard-filter
        # columns (e.g. KB has "tier 1" containing "1" — contrived but
        # exercises the promotion path) is promoted to ambiguous and
        # produces two branches via the kb_columns_supporting fallback.
        # We stub kb_columns_supporting directly so the test does not hit
        # the live KB.
        from phase4_agent import ambiguity_resolver
        ambiguity_resolver.kb_columns_supporting = (
            lambda term: ["tier", "facility_type"]
        )
        from core_agent.retrieval_types import ResolvedTerm
        e = _StubEntities(
            oem="hyundai", oem_list=["hyundai"],
            residual_abstract_terms=["mystery"],
        )
        unresolved = [
            ResolvedTerm(term="mystery", status="unresolved", selected_policy="drop")
        ]
        bs = self.branches(e, unresolved, self.QC.AMBIGUOUS_SEMANTIC, "mystery <oem>")
        self.assertEqual(len(bs), 2)
        ids = {b.branch_id for b in bs}
        self.assertEqual(ids, {"A", "B"})
        # Both branches must include a SQL filter plan.
        self.assertTrue(all(b.sql_plan is not None for b in bs))
        # Branch overlays target different columns.
        a_keys = set(bs[0].filters.keys())
        b_keys = set(bs[1].filters.keys())
        self.assertNotEqual(a_keys, b_keys)


class TestRetrievalFusion(unittest.TestCase):
    def setUp(self):
        from core_agent.retrieval_types import Candidate, QueryClass
        from retrievals.retrieval_fusion import merge, fuse, select
        self.Candidate = Candidate
        self.QC = QueryClass
        self.merge = merge
        self.fuse = fuse
        self.select = select

    def test_merge_dedupes_by_company_row_id(self):
        a = self.Candidate(canonical_name="Foo Inc", company_row_id=1, row={"id": 1})
        a.add_source("sql", 1.0)
        b = self.Candidate(canonical_name="Foo Inc", company_row_id=1, row={"id": 1, "extra": "x"})
        b.add_source("dense", 0.8)
        merged = self.merge([[a], [b]])
        self.assertEqual(len(merged), 1)
        self.assertEqual({"sql", "dense"}, merged[0].sources)

    def test_fuse_drops_vector_only_for_aggregate_query(self):
        a = self.Candidate(canonical_name="Foo", row={"id": 1})
        a.add_source("dense", 0.9)
        b = self.Candidate(canonical_name="Bar", row={"id": 2})
        b.add_source("sql", 1.0)
        out = self.fuse([a, b], self.QC.AGGREGATE)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].canonical_name, "Bar")

    def test_fuse_keeps_both_for_fallback(self):
        a = self.Candidate(canonical_name="Foo", row={"id": 1})
        a.add_source("dense", 0.9)
        b = self.Candidate(canonical_name="Bar", row={"id": 2})
        b.add_source("sparse", 0.85)
        out = self.fuse([a, b], self.QC.FALLBACK_SEMANTIC)
        self.assertEqual(len(out), 2)

    def test_select_marks_final_selected_top_n(self):
        cands = []
        for i in range(15):
            c = self.Candidate(
                canonical_name=f"C{i}",
                company_row_id=i,
                row={"id": i, "employment": (i + 1) * 10},
            )
            c.add_source("sql", 1.0)
            cands.append(c)
        out = self.select(cands, self.QC.TOP_N, limit=5)
        self.assertEqual(len(out), 5)
        self.assertTrue(all(c.final_selected for c in out))


class TestEvidenceVerification(unittest.TestCase):
    def setUp(self):
        # Stub _company_names so verify_answer doesn't hit Postgres.
        import filters_and_validation.evidence_validator as ev
        ev._company_names = lambda: ["Foo Inc", "Bar Manufacturing"]
        from filters_and_validation.evidence_validator import verify_answer
        from core_agent.retrieval_types import Candidate
        self.verify = verify_answer
        self.Candidate = Candidate

    def test_clean_answer_passes(self):
        c = self.Candidate(canonical_name="Foo Inc", row={"company_name": "Foo Inc", "employment": 100})
        result = self.verify("Foo Inc has 100 employees", [c])
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.hallucination_risk, 0)

    def test_hallucinated_name_flagged(self):
        c = self.Candidate(canonical_name="Foo Inc", row={"company_name": "Foo Inc", "employment": 100})
        result = self.verify(
            "Foo Inc and Bar Manufacturing both supply parts.",
            [c],
        )
        self.assertGreater(result.hallucination_risk, 0)
        self.assertIn("Bar Manufacturing", result.missing_names)


class TestBranchedFormatter(unittest.TestCase):
    def test_single_branch_returns_none(self):
        from core_agent.formatters import format_branched_answer
        from core_agent.retrieval_types import RetrievalBranch
        bs = [RetrievalBranch(branch_id="A", interpreted_meaning="x → y")]
        self.assertIsNone(format_branched_answer(bs))

    def test_two_branches_produces_labelled_sections(self):
        from core_agent.formatters import format_branched_answer
        from core_agent.retrieval_types import Candidate, RetrievalBranch
        # New invariant: every row a formatter sees must carry validated=True
        # (the evidence_validator sets this before the row reaches the
        # formatter in the live pipeline). Construct test rows accordingly.
        ca = Candidate(
            canonical_name="A1",
            row={"company_name": "A1", "tier": "T1", "validated": True},
        )
        cb = Candidate(
            canonical_name="B1",
            row={"company_name": "B1", "tier": "T2", "validated": True},
        )
        bs = [
            RetrievalBranch(
                branch_id="A",
                interpreted_meaning="<term> -> <colA> filter",
                evidence=[ca],
            ),
            RetrievalBranch(
                branch_id="B",
                interpreted_meaning="<term> -> <colB> filter",
                evidence=[cb],
            ),
        ]
        out = format_branched_answer(bs)
        self.assertIsNotNone(out)
        self.assertIn("Two KB-supported interpretations", out)
        self.assertIn("Branch A", out)
        self.assertIn("Branch B", out)


if __name__ == "__main__":
    unittest.main()
