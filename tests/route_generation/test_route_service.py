"""Tests for the route-generation service orchestration."""
from georgia_ev_intelligence.route_generation.route_service import RouteService
from georgia_ev_intelligence.route_generation.schemas import RawRoute, RouteName


def test_high_confidence_pre_route_bypasses_llm(fixture_metadata, fake_llm_factory) -> None:
    llm = fake_llm_factory(
        response=RawRoute(route=RouteName.vector_search, confidence=0.9, query_focus="unused"),
    )
    trace = RouteService(provider=fixture_metadata, llm_router=llm).route_with_trace("hello")

    assert trace.selected_router == "pre_router"
    assert trace.final_route.route == RouteName.no_retrieval
    assert llm.calls == []


def test_llm_failure_uses_semantic_fallback(fixture_metadata) -> None:
    class FailingRouter:
        def route(self, question, allowed_fields=None):
            raise RuntimeError("offline")

    trace = RouteService(
        provider=fixture_metadata,
        llm_router=FailingRouter(),
    ).route_with_trace("Tell me about battery suppliers")

    assert trace.selected_router == "fallback"
    assert trace.final_route.route == RouteName.vector_search
    assert trace.final_route.route_source == "fallback"
    assert trace.final_route.validation_status == "valid"
    assert "offline" in trace.llm_error
