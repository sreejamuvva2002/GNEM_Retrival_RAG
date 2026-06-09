from georgia_ev_intelligence.streamlit_ui.components import chat_messages
from georgia_ev_intelligence.streamlit_ui.models.chat import Message
from georgia_ev_intelligence.streamlit_ui.models.source import SourceViewModel


def _message() -> Message:
    return Message(
        id="answer-1",
        role="assistant",
        content="Answer",
        timestamp="2026-06-09T10:00:00",
    )


def _source() -> SourceViewModel:
    return SourceViewModel(
        id="source-1",
        title="Source One",
        snippet="Context",
        source_type="company",
        location_name="Atlanta",
        rank=1,
        rank_score=1.0,
        record_id="source-1",
        source_row_id=1,
        parent_chunk_text="Context",
    )


def test_open_sources_render_inline_below_answer(monkeypatch) -> None:
    rendered_sources = []
    component_html = []

    monkeypatch.setattr(chat_messages.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_messages.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(chat_messages, "_copy_control", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_messages.ui_state, "sources_panel_open", lambda: True)
    monkeypatch.setattr(
        chat_messages.sources_panel,
        "render",
        lambda sources: rendered_sources.extend(sources),
    )
    monkeypatch.setattr(
        chat_messages.components,
        "html",
        lambda markup, **kwargs: component_html.append(markup),
    )

    source = _source()
    chat_messages.render([_message()], [source])

    assert rendered_sources == [source]
    assert "sources-inline-anchor" in component_html[-1]


def test_closed_sources_do_not_render_details(monkeypatch) -> None:
    rendered_sources = []

    monkeypatch.setattr(chat_messages.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_messages.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(chat_messages, "_copy_control", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_messages.ui_state, "sources_panel_open", lambda: False)
    monkeypatch.setattr(
        chat_messages.sources_panel,
        "render",
        lambda sources: rendered_sources.extend(sources),
    )
    monkeypatch.setattr(chat_messages.components, "html", lambda *args, **kwargs: None)

    chat_messages.render([_message()], [_source()])

    assert rendered_sources == []
