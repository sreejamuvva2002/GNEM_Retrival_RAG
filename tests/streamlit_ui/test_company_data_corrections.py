import pandas as pd

from georgia_ev_intelligence.streamlit_ui.company_data_corrections import (
    apply_company_data_corrections,
)


def test_verified_georgia_company_locations_replace_bad_source_rows() -> None:
    source = pd.DataFrame(
        [
            {
                "company": "Hitachi Astemo Americas Inc.",
                "location": "Shorter, Harris County",
                "address": "400 Hanon Dr, Shorter, GA 36075",
                "latitude": 32.4705166,
                "longitude": -85.8969782,
            },
            {
                "company": "Honda Development & Manufacturing",
                "location": "Tallapoosa, Cherokee County",
                "address": "1000 Honda Dr, Tallapoosa, GA 30176",
                "latitude": 34.2575874,
                "longitude": -84.5070413,
            },
        ]
    )

    corrected = apply_company_data_corrections(source).set_index("company")

    assert corrected.loc["Hitachi Astemo Americas Inc.", "location"] == "Monroe, Walton County"
    assert corrected.loc["Hitachi Astemo Americas Inc.", "longitude"] == -83.6764364
    assert (
        corrected.loc["Honda Development & Manufacturing", "location"]
        == "Tallapoosa, Haralson County"
    )
    assert corrected.loc["Honda Development & Manufacturing", "longitude"] == -85.2728142
