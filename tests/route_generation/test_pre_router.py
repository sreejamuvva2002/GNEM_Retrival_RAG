"""Tests for the rule-based pre-router."""
from georgia_ev_intelligence.route_generation.routing.pre_router import PreRouter
from georgia_ev_intelligence.route_generation.schemas import RouteName
from georgia_ev_intelligence.route_generation.utils.text_normalization import (
    normalize_question,
)

PRE = PreRouter()


def _route(question: str):
    return PRE.route(normalize_question(question))


class TestHighConfidenceBypass:
    def test_greeting(self):
        r = _route("Hello!")
        assert r is not None and r.route == RouteName.no_retrieval
        assert r.confidence >= 0.85  # bypasses the LLM

    def test_thanks(self):
        assert _route("thanks").route == RouteName.no_retrieval

    def test_meta_question(self):
        assert _route("what can you do?").route == RouteName.no_retrieval

    def test_out_of_domain(self):
        r = _route("what's the weather today?")
        assert r is not None and r.route == RouteName.out_of_domain
        assert r.confidence >= 0.85


class TestSubThresholdProposals:
    def test_aggregate(self):
        r = _route("How many Tier 2/3 suppliers are there?")
        assert r is not None and r.route == RouteName.structured_sql
        assert r.confidence < 0.85  # LLM still runs to extract the filter

    def test_geo(self):
        r = _route("companies near Atlanta within 50 km")
        assert r is not None and r.route == RouteName.geo_search

    def test_disruption(self):
        r = _route("What are alternatives to SK Battery America?")
        assert r is not None and r.route == RouteName.disruption_analysis


class TestNoStrongSignal:
    def test_semantic_question_defers_to_llm(self):
        assert _route("Tell me about battery suppliers in the state") is None

    def test_empty(self):
        assert _route("") is None

    def test_no_false_positive_on_almost(self):
        # "almost" must not trigger the "most" aggregate signal.
        assert _route("Describe companies that almost exclusively make seats") is None
