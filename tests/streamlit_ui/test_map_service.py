import pandas as pd

from georgia_ev_intelligence.streamlit_ui.services.map_service import (
    _records_to_list,
    filter_records_to_companies,
)


def test_map_records_are_not_truncated_before_cited_company_filtering() -> None:
    df = pd.DataFrame(
        [
            {
                "company": f"Company {index}",
                "latitude": 30.0 + index / 1000,
                "longitude": -84.0,
            }
            for index in range(205)
        ]
    )

    records = _records_to_list(df)
    filtered = filter_records_to_companies(records, {"company204"})

    assert len(records) == 205
    assert [record["company"] for record in filtered] == ["Company 204"]


def test_map_shows_no_company_markers_without_citations() -> None:
    records = [{"company": "Company A", "latitude": 33.0, "longitude": -84.0}]

    assert filter_records_to_companies(records, set()) == []
