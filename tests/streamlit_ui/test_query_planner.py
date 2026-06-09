from georgia_ev_intelligence.streamlit_ui.spatial.query_planner import QueryPlanner


def test_near_to_city_is_parsed_as_a_radius_query() -> None:
    plan = QueryPlanner().plan("Which of these companies is near to Atlanta?")

    assert plan["hints"]["city"] == "Atlanta"
    assert plan["hints"]["radius_km"] == 100.0
