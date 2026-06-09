from georgia_ev_intelligence.streamlit_ui.services.query_dispatcher import QueryDispatcher


class FailingChatService:
    def answer(self, *args, **kwargs):
        raise AssertionError("NO_RETRIEVAL route must not call the chat service")


class FailingMapService:
    def locate(self, *args, **kwargs):
        raise AssertionError("NO_RETRIEVAL route must not call the map service")


def test_no_retrieval_route_bypasses_chat_and_map_services() -> None:
    result = QueryDispatcher(
        chat_service=FailingChatService(),
        map_service=FailingMapService(),
    ).dispatch("Hi")

    assert result.chat.answer == (
        "Hi! Ask me about companies in Georgia's automotive and EV supply chain."
    )
    assert result.chat.parent_contexts == []
    assert result.chat.trace == {
        "route": "NO_RETRIEVAL",
        "route_reason": "greeting",
    }
    assert result.map.records == []
    assert result.map.context.map_mode == "no_retrieval"
