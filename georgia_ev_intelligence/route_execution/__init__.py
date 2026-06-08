"""Route execution stage.

Consumes the validated ``FinalRoute`` JSONs produced by the route-generation
stage (``outputs/final_routes.jsonl``) and executes them against the database
to produce evidence and grounded answers.

    final route -> dispatcher -> per-route executor -> evidence + answer

This stage never calls the router, never modifies the route, and never lets an
LLM write SQL. ``execute_route`` is the single public entry point.
"""

from .executor import execute_route
from .schemas import ExecutionResult

__all__ = ["execute_route", "ExecutionResult"]
