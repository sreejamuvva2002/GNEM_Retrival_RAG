"""LLM router — proposes a raw route via a local Ollama model (langchain-ollama).

Prefers ``ChatOllama.with_structured_output(RawRoute, method="json_schema")``;
falls back to plain JSON-mode generation parsed into ``RawRoute``. The
``langchain_ollama`` import is lazy so the module imports (and tests that inject a
fake chat run) even without the dependency installed or Ollama running.
"""
from __future__ import annotations

import json
import re

from ..config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, ROUTER_NUM_PREDICT, ROUTER_TEMPERATURE
from ..schemas import RawRoute
from .prompts import SYSTEM_PROMPT, build_clarify_prompt, build_user_prompt

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class RouteParseError(Exception):
    """Raised when LLM output cannot be parsed/validated into a ``RawRoute``.

    Distinct from connectivity/unavailability errors so the service can route a
    malformed router response to clarification instead of silently degrading to a
    semantic (vector_search) fallback.
    """


def _build_chat():
    """Construct a JSON-mode ChatOllama from shared settings (lazy import)."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise RuntimeError(
            "langchain-ollama is required for the default LLM router. "
            "Install it with: pip install langchain-ollama"
        ) from exc

    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_LLM_MODEL,
        temperature=ROUTER_TEMPERATURE,
        num_predict=ROUTER_NUM_PREDICT,
        format="json",
    )


class LLMRouter:
    """Wraps a chat model and returns structured ``RawRoute`` proposals."""

    def __init__(self, chat=None, structured=None, build_default: bool = True) -> None:
        self._chat = chat
        if self._chat is None and build_default:
            self._chat = _build_chat()
        # Prefer structured output; tolerate models/versions that don't support it.
        self._structured = structured
        if self._structured is None and self._chat is not None:
            self._structured = self._make_structured(self._chat)

    @staticmethod
    def _make_structured(chat):
        try:
            return chat.with_structured_output(
                RawRoute, method="json_schema", include_raw=True
            )
        except Exception:  # pragma: no cover - depends on model/version
            return None

    # -- public API ---------------------------------------------------------
    def route(self, question: str, allowed_fields=None) -> RawRoute:
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", build_user_prompt(question, allowed_fields)),
        ]
        return self._invoke(messages)

    def route_with_clarification(
        self, question: str, prior: RawRoute | None, answer: str, allowed_fields=None
    ) -> RawRoute:
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", build_clarify_prompt(question, prior, answer, allowed_fields)),
        ]
        return self._invoke(messages)

    # -- internals ----------------------------------------------------------
    def _invoke(self, messages) -> RawRoute:
        if self._structured is not None:
            try:
                return self._from_structured(self._structured.invoke(messages))
            except Exception:
                pass  # fall through to JSON-mode parsing
        # Connectivity / unavailability errors from ``invoke`` propagate as-is; only
        # a parse/validation failure of the response becomes a RouteParseError.
        resp = self._chat.invoke(messages)
        try:
            return self._parse_json(getattr(resp, "content", resp))
        except Exception as exc:
            raise RouteParseError(str(exc)) from exc

    def _from_structured(self, out) -> RawRoute:
        parsed = out.get("parsed") if isinstance(out, dict) else out
        if isinstance(parsed, RawRoute):
            return parsed
        if parsed is not None:
            return RawRoute.model_validate(parsed)
        # structured parse failed -> salvage the raw text
        raw = out.get("raw") if isinstance(out, dict) else None
        content = getattr(raw, "content", None)
        if content:
            return self._parse_json(content)
        raise ValueError("structured output returned no parsable result")

    @staticmethod
    def _parse_json(text) -> RawRoute:
        if isinstance(text, (dict, list)):
            return RawRoute.model_validate(text)
        s = str(text or "")
        match = _JSON_OBJECT.search(s)
        payload = json.loads(match.group(0) if match else s)
        return RawRoute.model_validate(payload)
