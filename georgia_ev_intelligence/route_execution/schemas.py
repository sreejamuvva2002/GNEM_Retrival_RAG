"""Data models for the route execution stage.

These are lightweight dataclasses (mirroring ``runtime_pipeline/schemas.py``)
that carry the outcome of executing a single validated ``FinalRoute`` against
the database. The router decides *what* to do; these structures describe *what
was retrieved* and the grounded answer derived from it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Status values written to the JSONL outputs. ``success`` covers every route
# that ran to completion (including direct/clarification/out_of_domain answers);
# ``failed`` is reserved for executor errors and unsupported routes.
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"


@dataclass
class ExecutionResult:
    """Outcome of executing one ``FinalRoute``.

    ``evidence`` is a JSON-serialisable dict whose ``type`` key names the shape
    of the payload (``structured_rows``, ``document_chunks``, ``direct``,
    ``clarification``, ``out_of_domain``, ``ranked_alternatives`` …).
    """

    route: str
    status: str
    answer: str
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def failure(cls, route: str, reason: str) -> "ExecutionResult":
        return cls(
            route=route,
            status=STATUS_FAILED,
            answer="",
            evidence={"type": "error"},
            error=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "route": self.route,
            "status": self.status,
            "answer": self.answer,
            "evidence": self.evidence,
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload
