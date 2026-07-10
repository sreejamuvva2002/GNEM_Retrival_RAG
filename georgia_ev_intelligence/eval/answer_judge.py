"""LLM-as-judge answer scoring.

Mirrors the structured-output + JSON-mode-fallback pattern in
``route_generation/routing/llm_router.py``, but grades an already-generated
answer against its evidence instead of proposing a route. The
``langchain_ollama`` import is lazy so this module imports even without the
dependency installed or Ollama running.
"""
from __future__ import annotations

import json
import re
from typing import Any

from georgia_ev_intelligence.shared import config

from .judge_prompts import SYSTEM_PROMPT, build_user_prompt
from .judge_schemas import AnswerJudgment

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class JudgeParseError(Exception):
    """Raised when the judge's output cannot be parsed/validated into an ``AnswerJudgment``."""


def _build_chat():
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise RuntimeError(
            "langchain-ollama is required for the answer judge. "
            "Install it with: pip install langchain-ollama"
        ) from exc

    return ChatOllama(
        base_url=config.OLLAMA_BASE_URL,
        model=config.OLLAMA_LLM_MODEL,
        temperature=0.0,
        format="json",
    )


class AnswerJudge:
    """Wraps a chat model and returns structured ``AnswerJudgment`` scores."""

    def __init__(self, chat=None, structured=None, build_default: bool = True) -> None:
        self._chat = chat
        if self._chat is None and build_default:
            self._chat = _build_chat()
        self._structured = structured
        if self._structured is None and self._chat is not None:
            self._structured = self._make_structured(self._chat)

    @staticmethod
    def _make_structured(chat):
        try:
            return chat.with_structured_output(
                AnswerJudgment, method="json_schema", include_raw=True
            )
        except Exception:  # pragma: no cover - depends on model/version
            return None

    def judge(self, question: str, evidence: dict[str, Any], answer: str) -> AnswerJudgment:
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", build_user_prompt(question, evidence, answer)),
        ]
        return self._invoke(messages)

    def _invoke(self, messages) -> AnswerJudgment:
        if self._structured is not None:
            try:
                return self._from_structured(self._structured.invoke(messages))
            except Exception:
                pass  # fall through to JSON-mode parsing
        resp = self._chat.invoke(messages)
        try:
            return self._parse_json(getattr(resp, "content", resp))
        except Exception as exc:
            raise JudgeParseError(str(exc)) from exc

    def _from_structured(self, out) -> AnswerJudgment:
        parsed = out.get("parsed") if isinstance(out, dict) else out
        if isinstance(parsed, AnswerJudgment):
            return parsed
        if parsed is not None:
            return AnswerJudgment.model_validate(parsed)
        raw = out.get("raw") if isinstance(out, dict) else None
        content = getattr(raw, "content", None)
        if content:
            return self._parse_json(content)
        raise ValueError("structured output returned no parsable result")

    @staticmethod
    def _parse_json(text) -> AnswerJudgment:
        if isinstance(text, (dict, list)):
            return AnswerJudgment.model_validate(text)
        s = str(text or "")
        match = _JSON_OBJECT.search(s)
        payload = json.loads(match.group(0) if match else s)
        return AnswerJudgment.model_validate(payload)


_default_judge: AnswerJudge | None = None


def judge_answer(question: str, evidence: dict[str, Any], answer: str) -> AnswerJudgment:
    """Score one answer, reusing one lazily-built default judge across calls."""
    global _default_judge
    if _default_judge is None:
        _default_judge = AnswerJudge()
    return _default_judge.judge(question, evidence, answer)
