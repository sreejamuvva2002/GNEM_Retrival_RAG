from georgia_ev_intelligence.streamlit_ui.services.interfaces import ChatResult
from georgia_ev_intelligence.streamlit_ui.services.query_dispatcher import QueryDispatcher


class NoRetrievalChatService:
    def answer(self, *args, **kwargs):
        return ChatResult(
            answer="Message is only a greeting or meta question.",
            parent_contexts=[],
            trace={"route": "no_retrieval"},
        )


class FailingMapService:
    def locate(self, *args, **kwargs):
        raise AssertionError("NO_RETRIEVAL route must not call the map service")


def test_router_no_retrieval_result_bypasses_map_service() -> None:
    result = QueryDispatcher(
        chat_service=NoRetrievalChatService(),
        map_service=FailingMapService(),
    ).dispatch("Hi")

    assert result.chat.answer == "Message is only a greeting or meta question."
    assert result.chat.parent_contexts == []
    assert result.chat.trace == {"route": "no_retrieval"}
    assert result.map.records == []
    assert result.map.context.map_mode == "no_retrieval"
