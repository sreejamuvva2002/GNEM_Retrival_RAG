"""In-memory storage for pending clarification sessions."""
from __future__ import annotations

import threading
from typing import Protocol

from .exceptions import ClarificationNotFoundError
from .models import StoredClarificationSession, utc_now


class ClarificationStoreProtocol(Protocol):
    def save_session(self, session: StoredClarificationSession) -> None:
        ...

    def get_session(self, clarification_id: str) -> StoredClarificationSession:
        ...

    def update_session(self, session: StoredClarificationSession) -> None:
        ...


class InMemoryClarificationStore:
    """Thread-safe dictionary-backed clarification session store."""

    def __init__(self) -> None:
        self._sessions: dict[str, StoredClarificationSession] = {}
        self._lock = threading.Lock()

    def save_session(self, session: StoredClarificationSession) -> None:
        with self._lock:
            self._sessions[session.clarification_id] = session

    def get_session(self, clarification_id: str) -> StoredClarificationSession:
        with self._lock:
            session = self._sessions.get(clarification_id)
        if session is None:
            raise ClarificationNotFoundError(
                f"Clarification session not found: {clarification_id}"
            )
        return session

    def update_session(self, session: StoredClarificationSession) -> None:
        session.updated_at = utc_now()
        with self._lock:
            if session.clarification_id not in self._sessions:
                raise ClarificationNotFoundError(
                    f"Clarification session not found: {session.clarification_id}"
                )
            self._sessions[session.clarification_id] = session
