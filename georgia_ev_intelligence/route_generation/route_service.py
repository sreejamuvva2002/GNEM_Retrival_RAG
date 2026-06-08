"""Orchestrate normalization, route proposal, and route validation."""
from __future__ import annotations

from dataclasses import dataclass

from .config import PRE_ROUTER_HIGH_CONFIDENCE, ROUTER_CONFIDENCE_THRESHOLD
from .metadata import build_metadata_provider
from .metadata.provider import MetadataProvider
from .routing.llm_router import LLMRouter, RouteParseError
from .routing.pre_router import PreRouter
from .routing.validator import RouteValidator
from .schemas import FinalRoute, RawRoute, RouteName
from .utils.text_normalization import normalize_question


@dataclass(frozen=True)
class RouteTrace:
    """Intermediate routing findings for diagnostics and audit exports."""

    normalized_question: dict
    pre_route: RawRoute | None
    raw_route: RawRoute
    final_route: FinalRoute
    selected_router: str
    llm_error: str = ""


class RouteService:
    """Turn one question into a validated final route."""

    def __init__(
        self,
        provider: MetadataProvider,
        pre_router: PreRouter | None = None,
        llm_router: LLMRouter | None = None,
        validator: RouteValidator | None = None,
        pre_router_high_confidence: float = PRE_ROUTER_HIGH_CONFIDENCE,
    ) -> None:
        self._provider = provider
        self._pre_router = pre_router or PreRouter()
        self._llm_router = llm_router
        self._validator = validator or RouteValidator(provider)
        self._pre_router_high_confidence = pre_router_high_confidence

    def route(self, question: str) -> FinalRoute:
        """Return only the validated route contract."""
        return self.route_with_trace(question).final_route

    def route_with_trace(self, question: str) -> RouteTrace:
        """Return the validated route and every intermediate routing finding."""
        normalized = normalize_question(question)
        pre_route = self._pre_router.route(normalized)
        llm_error = ""

        if pre_route is not None and pre_route.confidence >= self._pre_router_high_confidence:
            raw_route = pre_route
            selected_router = "pre_router"
            route_source = "pre_router_validated"
        else:
            try:
                raw_route = self._get_llm_router().route(
                    normalized["normalized"],
                    self._provider.get_allowed_fields(),
                )
                selected_router = "llm_router"
                route_source = "llm_router_validated"
            except RouteParseError as exc:
                # Malformed/invalid router output: ask for clarification rather than
                # silently degrading to a semantic (vector_search) fallback.
                llm_error = f"{type(exc).__name__}: {exc}"
                raw_route = pre_route or self._clarification_fallback_route(
                    normalized["normalized"], llm_error
                )
                selected_router = "fallback"
                route_source = "fallback"
            except Exception as exc:
                # Connectivity / unavailability: fall back to semantic retrieval.
                llm_error = f"{type(exc).__name__}: {exc}"
                raw_route = pre_route or self._fallback_route(normalized["normalized"], llm_error)
                selected_router = "fallback"
                route_source = "fallback"

        final_route = self._validator.validate(
            raw_route,
            normalized,
            route_source=route_source,
        )
        return RouteTrace(
            normalized_question=normalized,
            pre_route=pre_route,
            raw_route=raw_route,
            final_route=final_route,
            selected_router=selected_router,
            llm_error=llm_error,
        )

    def _get_llm_router(self) -> LLMRouter:
        if self._llm_router is None:
            self._llm_router = LLMRouter()
        return self._llm_router

    @staticmethod
    def _fallback_route(question: str, error: str) -> RawRoute:
        return RawRoute(
            route=RouteName.vector_search,
            confidence=ROUTER_CONFIDENCE_THRESHOLD,
            query_focus=question,
            reason=f"LLM router unavailable; using semantic retrieval fallback. {error}",
        )

    @staticmethod
    def _clarification_fallback_route(question: str, error: str) -> RawRoute:
        """Router output could not be parsed — ask the user rather than guess."""
        return RawRoute(
            route=RouteName.clarification_needed,
            confidence=0.0,
            query_focus=question,
            reason=f"Router output could not be parsed; requesting clarification. {error}",
        )


def build_default_route_service() -> RouteService:
    """Build the configured route service with live or snapshot metadata."""
    return RouteService(provider=build_metadata_provider())
