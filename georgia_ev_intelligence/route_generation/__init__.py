"""Final route generation layer.

Turns a user question into a validated ``FinalRoute`` JSON contract:

    question -> normalize -> pre-router -> (LLM router) -> validator
             -> FinalRoute | ClarificationRequest

This package only *decides* routes. It never executes retrieval, runs SQL,
performs vector/keyword/geo search, or generates final answers — that is the
responsibility of the downstream execution branch, which consumes ``FinalRoute``.

``RouteService`` / ``build_default_route_service`` are re-exported for callers
that only need the validated route contract.
"""

from .route_service import RouteService, RouteTrace, build_default_route_service

__all__ = ["RouteService", "RouteTrace", "build_default_route_service"]
