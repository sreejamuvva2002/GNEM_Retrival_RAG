from __future__ import annotations

import streamlit as st
import pytest

from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.models import (
    HybridRetrievalResult,
)
from georgia_ev_intelligence.runtime_pipeline.schemas import ParentContext
from georgia_ev_intelligence.streamlit_ui.models.chat import ChatMemory
from georgia_ev_intelligence.streamlit_ui.models.map import MapResult
from georgia_ev_intelligence.streamlit_ui.services.chat_service import ChatService
from georgia_ev_intelligence.streamlit_ui.services.interfaces import ChatResult
from georgia_ev_intelligence.streamlit_ui.services.query_dispatcher import QueryDispatcher
from georgia_ev_intelligence.streamlit_ui.state import chat_state


class FakePipeline:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve_with_sources(self, query: str) -> HybridRetrievalResult:
        self.queries.append(query)
        return HybridRetrievalResult(
            parent_contexts=[
                ParentContext(
                    record_id="p1",
                    source_row_id=1,
                    parent_chunk_text="Company: Battery Co\nProduct: battery cells",
                )
            ],
            dense_children=[],
            sparse_children=[],
            trace=None,
        )


def test_chat_service_rewrites_follow_up_before_retrieval() -> None:
    pipeline = FakePipeline()
    prompts: list[str] = []

    def fake_generate(prompt: str, timeout: int = 180) -> str:
        prompts.append(prompt)
        if "Standalone retrieval query:" in prompt:
            return "Tier 1 suppliers in Troup County that are battery-related"
        return '{"answer":"Battery Co | Product: battery cells","used_companies":["Battery Co"]}'

    service = ChatService(
        retrieval_pipeline_factory=lambda: pipeline,
        generate_answer_fn=fake_generate,
    )

    result = service.answer(
        "What about battery-related ones?",
        chat_memory=ChatMemory(
            summary="The prior topic was Tier 1 suppliers in Troup County.",
            recent_messages=[
                {"role": "user", "content": "Which Tier 1 suppliers are in Troup County?"},
                {"role": "assistant", "content": "There are several Tier 1 suppliers."},
            ],
        ),
    )

    assert pipeline.queries == ["Tier 1 suppliers in Troup County that are battery-related"]
    assert result.effective_query == "Tier 1 suppliers in Troup County that are battery-related"
    assert result.history_used is True
    assert "Conversation summary:\nThe prior topic was Tier 1 suppliers" in prompts[0]
    assert "Recent conversation:\nUser: Which Tier 1 suppliers" in prompts[0]
    assert "Original user question:\nWhat about battery-related ones?" in prompts[-1]
    assert "Standalone retrieval question:\nTier 1 suppliers in Troup County" in prompts[-1]


def test_chat_service_falls_back_when_rewrite_is_bad() -> None:
    pipeline = FakePipeline()

    def fake_generate(prompt: str, timeout: int = 180) -> str:
        if "Standalone retrieval query:" in prompt:
            return ""
        return '{"answer":"Battery Co | Product: battery cells","used_companies":["Battery Co"]}'

    service = ChatService(
        retrieval_pipeline_factory=lambda: pipeline,
        generate_answer_fn=fake_generate,
    )

    result = service.answer(
        "What about battery-related ones?",
        chat_memory=ChatMemory(
            recent_messages=[{"role": "user", "content": "Which OEMs are in Fulton County?"}]
        ),
    )

    assert pipeline.queries == ["What about battery-related ones?"]
    assert result.effective_query == "What about battery-related ones?"
    assert result.history_used is False


def test_chat_service_keeps_standalone_question_unchanged() -> None:
    pipeline = FakePipeline()

    def fake_generate(prompt: str, timeout: int = 180) -> str:
        if "Standalone retrieval query:" in prompt:
            return "Which companies are in Fulton County?"
        return '{"answer":"Battery Co | Product: battery cells","used_companies":["Battery Co"]}'

    service = ChatService(
        retrieval_pipeline_factory=lambda: pipeline,
        generate_answer_fn=fake_generate,
    )

    result = service.answer(
        "Which companies are in Fulton County?",
        chat_memory=ChatMemory(
            summary="The previous topic was battery-related suppliers in Troup County.",
            recent_messages=[{"role": "user", "content": "Show Troup County suppliers"}],
        ),
    )

    assert pipeline.queries == ["Which companies are in Fulton County?"]
    assert result.effective_query == "Which companies are in Fulton County?"
    assert result.history_used is False


def test_chat_service_summarizes_completed_turns() -> None:
    prompts: list[str] = []

    def fake_generate(prompt: str, timeout: int = 180) -> str:
        prompts.append(prompt)
        return "Updated summary: Troup County Tier 1 supplier filter; battery relevance follow-up."

    service = ChatService(
        retrieval_pipeline_factory=FakePipeline,
        generate_answer_fn=fake_generate,
    )

    summary = service.summarize_memory(
        "Existing summary: Fulton County OEM context.",
        [
            {"role": "user", "content": "Which Tier 1 suppliers are in Troup County?"},
            {"role": "assistant", "content": "There are several Tier 1 suppliers."},
        ],
    )

    assert summary == "Troup County Tier 1 supplier filter; battery relevance follow-up."
    assert len(summary) <= 1200
    assert "Existing summary:\nExisting summary: Fulton County OEM context." in prompts[0]
    assert "New completed turns:\nUser: Which Tier 1 suppliers" in prompts[0]


def test_summary_failure_is_separate_from_answer_generation() -> None:
    pipeline = FakePipeline()

    def fake_generate(prompt: str, timeout: int = 180) -> str:
        if "Updated summary:" in prompt:
            raise RuntimeError("summary failed")
        if "Standalone retrieval query:" in prompt:
            return "Which companies are in Fulton County?"
        return '{"answer":"Battery Co | Product: battery cells","used_companies":["Battery Co"]}'

    service = ChatService(
        retrieval_pipeline_factory=lambda: pipeline,
        generate_answer_fn=fake_generate,
    )

    result = service.answer(
        "Which companies are in Fulton County?",
        chat_memory=ChatMemory(summary="Prior Troup County supplier context."),
    )

    assert result.answer == "Battery Co | Product: battery cells"
    assert pipeline.queries == ["Which companies are in Fulton County?"]
    with pytest.raises(RuntimeError, match="summary failed"):
        service.summarize_memory(
            "Prior Troup County supplier context.",
            [{"role": "user", "content": "Which Tier 1 suppliers are in Troup County?"}],
        )


def test_chat_state_compacts_only_messages_older_than_recent_window() -> None:
    st.session_state.clear()
    chat_state.initialize()

    for i in range(5):
        chat_state.append_message(chat_state.make_user_message(f"question {i}"))
        chat_state.append_message(chat_state.make_assistant_message(f"answer {i}", []))

    memory = chat_state.rag_memory(recent_turns=4)
    batch, next_cursor = chat_state.summary_compaction_batch(recent_turns=4)

    assert [m["content"] for m in memory.recent_messages[:2]] == ["question 1", "answer 1"]
    assert [m["content"] for m in batch] == ["question 0", "answer 0"]
    assert next_cursor == 2


class FakeChatService:
    def answer(self, query: str, chat_memory=None, on_step=None) -> ChatResult:
        return ChatResult(
            answer="answer",
            parent_contexts=[],
            effective_query="suppliers near Atlanta",
            history_used=True,
        )


class FakeMapService:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def locate(self, query: str) -> MapResult:
        self.queries.append(query)
        return MapResult()


def test_dispatcher_uses_effective_query_for_map_lookup() -> None:
    map_service = FakeMapService()
    dispatcher = QueryDispatcher(FakeChatService(), map_service)

    result = dispatcher.dispatch(
        "Show those near Atlanta",
        chat_memory=ChatMemory(
            recent_messages=[{"role": "user", "content": "Which suppliers support Hyundai?"}]
        ),
    )

    assert result.query == "Show those near Atlanta"
    assert map_service.queries == ["suppliers near Atlanta"]
