from types import SimpleNamespace

from georgia_ev_intelligence.runtime_pipeline.schemas import ParentContext
from georgia_ev_intelligence.streamlit_ui.services.chat_service import (
    ChatService,
    _contextualize_retrieval_query,
    _filter_parent_contexts_by_companies,
    _parse_json_response,
)


def _parent(company: str, row_id: int = 1) -> ParentContext:
    return ParentContext(
        record_id=f"KB_ROW_{row_id}",
        source_row_id=row_id,
        parent_chunk_text=f"Company: {company}\nCategory: EV",
    )


def test_salvages_unescaped_multiline_json_answer() -> None:
    raw = """
    {
      "answer": "There are 17 companies in Georgia related to EV supply chain."
      toyota industries group (tacg-tica) [OEM Footprint] | Employment: 18100
      blue bird corp. [Tier 2/3] | Employment: 275
      ',
      "used_companies": [
        "toyota industries group (tacg-tica)",
        "blue bird corp."
      ]
    }
    """

    answer, companies, parsed = _parse_json_response(raw)

    assert parsed is True
    assert answer.startswith("There are 2 companies")
    assert '"answer"' not in answer
    assert "used_companies" not in answer
    assert companies == ["toyota industries group (tacg-tica)", "blue bird corp."]


def test_salvages_truncated_json_without_used_companies() -> None:
    answer, companies, parsed = _parse_json_response(
        '{"answer": "There are 2 companies in Georgia.\\nCompany A\\nCompany B"'
    )

    assert parsed is True
    assert answer == "There are 2 companies in Georgia.\nCompany A\nCompany B"
    assert companies == []


def test_contextual_follow_up_reuses_previous_result_set() -> None:
    previous = [_parent("Company A", 1), _parent("Company B", 2)]

    class Pipeline:
        def retrieve_with_sources(self, query):
            raise AssertionError("follow-up should reuse prior contexts")

    captured = {}

    def generate(prompt, timeout=180, json_mode=False):
        captured["prompt"] = prompt
        captured["json_mode"] = json_mode
        return (
            '{"answer":"There are 2 companies in Georgia.\\n'
            'Company A\\nCompany B","used_companies":["Company A","Company B"]}'
        )

    service = ChatService(
        retrieval_pipeline_factory=Pipeline,
        generate_answer_fn=generate,
    )
    result = service.answer(
        "Give me the list of these companies.",
        history=[
            ("user", "How many EV companies are in Georgia?"),
            ("assistant", "There are 2 EV companies in Georgia."),
        ],
        previous_contexts=previous,
    )

    assert result.error == ""
    assert [parent.record_id for parent in result.parent_contexts] == ["KB_ROW_1", "KB_ROW_2"]
    assert "How many EV companies are in Georgia?" in captured["prompt"]
    assert captured["json_mode"] is True


def test_contextualizes_retrieval_when_prior_contexts_are_unavailable() -> None:
    query = _contextualize_retrieval_query(
        "Give me the list of these companies.",
        [("user", "How many EV companies are in Georgia?")],
    )

    assert query == (
        "How many EV companies are in Georgia?\n"
        "Follow-up: Give me the list of these companies."
    )


def test_non_follow_up_still_runs_retrieval() -> None:
    calls = []
    parent = _parent("Company A")

    class Pipeline:
        def retrieve_with_sources(self, query):
            calls.append(query)
            return SimpleNamespace(parent_contexts=[parent], trace=None)

    service = ChatService(
        retrieval_pipeline_factory=Pipeline,
        generate_answer_fn=lambda prompt, timeout=180: (
            '{"answer":"Company A","used_companies":["Company A"]}'
        ),
    )
    result = service.answer(
        "Which EV company is listed?",
        history=[("user", "An unrelated earlier question")],
        previous_contexts=[_parent("Old Company")],
    )

    assert calls == ["Which EV company is listed?"]
    assert [context.record_id for context in result.parent_contexts] == ["KB_ROW_1"]


def test_exact_role_query_uses_all_structured_matches() -> None:
    companies = [
        ("F&P Georgia Manufacturing", "Battery Pack"),
        ("Hitachi Astemo Americas Inc.", "Battery Cell"),
        ("Hollingsworth & Vose Co.", "Battery Pack"),
        ("Honda Development & Manufacturing", "Battery Cell"),
        ("Hyundai Motor Group", "Battery Pack"),
        ("IMMI", "Battery Pack"),
    ]
    lookup = {
        index: {
            "company": company,
            "category": "Tier 1/2",
            "ev_supply_chain_role": role,
        }
        for index, (company, role) in enumerate(companies)
    }
    lookup[99] = {
        "company": "SK Battery America",
        "category": "Tier 1",
        "ev_supply_chain_role": "Materials",
    }

    def fail():
        raise AssertionError("exact role query should not run approximate retrieval")

    service = ChatService(
        retrieval_pipeline_factory=fail,
        generate_answer_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("exact role query should not call the LLM")
        ),
        structured_lookup_fn=lambda: lookup,
    )
    result = service.answer(
        "Which Georgia companies are classified under Battery Cell or Battery Pack roles, "
        "and what tier is each assigned?"
    )

    assert result.error == ""
    assert result.answer.startswith(
        "There are 6 Georgia companies classified under Battery Cell or Battery Pack roles."
    )
    assert "F&P Georgia Manufacturing [Tier 1/2]" in result.answer
    assert "Hitachi Astemo Americas Inc. [Tier 1/2]" in result.answer
    assert "SK Battery America" not in result.answer
    assert len(result.parent_contexts) == 6
    assert result.trace["matched_company_count"] == 6


def test_contextual_proximity_then_distance_uses_coordinates() -> None:
    lookup = {
        0: {
            "company": "Company A",
            "category": "Tier 1/2",
            "ev_supply_chain_role": "Battery Cell",
            "location": "Nearville, Fulton County",
            "city": "Nearville",
            "county": "Fulton",
            "latitude": 0.0,
            "longitude": 0.5,
        },
        1: {
            "company": "Company B",
            "category": "Tier 1/2",
            "ev_supply_chain_role": "Battery Pack",
            "location": "Nearville, Fulton County",
            "city": "Nearville",
            "county": "Fulton",
            "latitude": 0.0,
            "longitude": 0.8,
        },
        2: {
            "company": "Company C",
            "category": "Tier 1/2",
            "ev_supply_chain_role": "Battery Pack",
            "location": "Farville, Other County",
            "city": "Farville",
            "county": "Other",
            "latitude": 0.0,
            "longitude": 1.2,
        },
        3: {
            "company": "Atlanta Reference",
            "category": "Other",
            "ev_supply_chain_role": "General Automotive",
            "location": "Atlanta, Fulton County",
            "city": "Atlanta",
            "county": "Fulton",
            "latitude": 0.0,
            "longitude": 0.0,
        },
    }

    def fail():
        raise AssertionError("coordinate follow-ups should not run approximate retrieval")

    service = ChatService(
        retrieval_pipeline_factory=fail,
        generate_answer_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("coordinate follow-ups should not call the LLM")
        ),
        structured_lookup_fn=lambda: lookup,
    )
    role_result = service.answer(
        "Which companies have Battery Cell or Battery Pack roles?"
    )
    near_result = service.answer(
        "Which of these companies is near to Atlanta?",
        previous_contexts=role_result.parent_contexts,
    )
    distance_result = service.answer(
        "Give me the exact distance between these companies.",
        previous_contexts=near_result.parent_contexts,
    )

    assert near_result.answer.startswith(
        "There are 2 of these companies within 100 km of Atlanta."
    )
    assert "Company A | Distance from Atlanta: 55.6 km" in near_result.answer
    assert "Company B | Distance from Atlanta: 89.0 km" in near_result.answer
    assert "Company C" not in near_result.answer
    assert len(near_result.parent_contexts) == 2
    assert distance_result.answer == (
        "The straight-line distance between Company A and Company B is "
        "33.4 km (20.7 mi), calculated from their coordinates."
    )
    assert len(distance_result.parent_contexts) == 2


def test_source_filter_does_not_match_similarly_named_company() -> None:
    parents = [
        _parent("Hitachi Astemo", 1),
        _parent("Hitachi Astemo Americas Inc.", 2),
    ]

    filtered = _filter_parent_contexts_by_companies(
        parents,
        ["Hitachi Astemo Americas Inc."],
    )

    assert [parent.record_id for parent in filtered] == ["KB_ROW_2"]
