from georgia_ev_intelligence.streamlit_ui.components import chat_messages
from georgia_ev_intelligence.streamlit_ui.models.chat import Message
from georgia_ev_intelligence.streamlit_ui.models.source import Provenance, SourceViewModel


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
    rendered_provenance = []

    monkeypatch.setattr(chat_messages.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_messages.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(chat_messages, "_copy_control", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_messages.ui_state, "sources_panel_open", lambda: True)
    monkeypatch.setattr(
        chat_messages.sources_panel,
        "render",
        lambda provenance: rendered_provenance.append(provenance),
    )
    monkeypatch.setattr(
        chat_messages.components,
        "html",
        lambda *args, **kwargs: None,
    )

    provenance = Provenance(sources=[_source()], kind="records")
    chat_messages.render([_message()], provenance)

    assert rendered_provenance == [provenance]


def test_closed_sources_do_not_render_details(monkeypatch) -> None:
    rendered_provenance = []

    monkeypatch.setattr(chat_messages.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_messages.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(chat_messages, "_copy_control", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_messages.ui_state, "sources_panel_open", lambda: False)
    monkeypatch.setattr(
        chat_messages.sources_panel,
        "render",
        lambda provenance: rendered_provenance.append(provenance),
    )
    monkeypatch.setattr(chat_messages.components, "html", lambda *args, **kwargs: None)

    chat_messages.render([_message()], Provenance(sources=[_source()], kind="records"))

    assert rendered_provenance == []


def test_sql_only_provenance_is_not_user_facing() -> None:
    provenance = Provenance(
        sql_queries=[{"label": "SQL query", "sql": "SELECT secret_details;"}],
        kind="count",
    )

    assert provenance.has_content() is False
