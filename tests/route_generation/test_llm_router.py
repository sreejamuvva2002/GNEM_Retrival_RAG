"""Tests for the LLM router — structured path, salvage, and JSON fallback.

No Ollama and no langchain are required: a fake chat / structured object is
injected and ``build_default=False`` skips constructing a real ChatOllama.
"""
import pytest

from georgia_ev_intelligence.route_generation.routing.llm_router import LLMRouter
from georgia_ev_intelligence.route_generation.schemas import RawRoute, RouteName


class FakeMsg:
    def __init__(self, content):
        self.content = content


class FakeChat:
    def __init__(self, content):
        self._content = content
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return FakeMsg(self._content)


class FakeStructured:
    def __init__(self, out, raises=False):
        self._out = out
        self._raises = raises

    def invoke(self, messages):
        if self._raises:
            raise RuntimeError("structured output unsupported")
        return self._out


def _router(chat=None, structured=None):
    return LLMRouter(chat=chat, structured=structured, build_default=False)


class TestParseJson:
    def test_prose_wrapped(self):
        rr = LLMRouter._parse_json(
            'Sure: {"route": "vector_search", "confidence": 0.5, "reason": "x"} done'
        )
        assert rr.route == RouteName.vector_search

    def test_code_fenced(self):
        rr = LLMRouter._parse_json(
            '```json\n{"route": "geo_search", "confidence": 0.6, "reason": "y"}\n```'
        )
        assert rr.route == RouteName.geo_search

    def test_dict_input(self):
        rr = LLMRouter._parse_json({"route": "no_retrieval", "confidence": 1.0, "reason": "z"})
        assert rr.route == RouteName.no_retrieval


class TestInvokePaths:
    def test_structured_returns_parsed(self):
        canned = RawRoute(route=RouteName.structured_sql, confidence=0.9, reason="r")
        router = _router(
            chat=FakeChat("{}"),
            structured=FakeStructured({"parsed": canned, "raw": FakeMsg("")}),
        )
        out = router.route("how many suppliers", ["category"])
        assert out is canned

    def test_structured_salvages_raw_on_none_parsed(self):
        raw_json = '{"route": "geo_search", "confidence": 0.7, "reason": "r"}'
        router = _router(
            chat=FakeChat("{}"),
            structured=FakeStructured({"parsed": None, "raw": FakeMsg(raw_json)}),
        )
        assert router.route("near atlanta", []).route == RouteName.geo_search

    def test_fallback_when_no_structured(self):
        chat = FakeChat('noise {"route": "structured_sql", "confidence": 0.8, "reason": "r"} end')
        router = _router(chat=chat, structured=None)
        assert router.route("list companies", []).route == RouteName.structured_sql
        assert chat.calls  # the chat was actually used

    def test_fallback_when_structured_raises(self):
        chat = FakeChat('{"route": "keyword_search", "confidence": 0.6, "reason": "r"}')
        router = _router(chat=chat, structured=FakeStructured(None, raises=True))
        assert router.route("find phrase", []).route == RouteName.keyword_search

    def test_invalid_route_value_raises(self):
        # An invented route name fails pydantic validation -> caller falls back.
        chat = FakeChat('{"route": "totally_made_up", "confidence": 0.9, "reason": "r"}')
        router = _router(chat=chat, structured=None)
        with pytest.raises(Exception):
            router.route("q", [])
