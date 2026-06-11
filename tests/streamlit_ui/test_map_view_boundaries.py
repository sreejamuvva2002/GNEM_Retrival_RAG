from georgia_ev_intelligence.streamlit_ui.components.map_view import (
    _COUNTY_BOUNDARIES_PATH,
    _STATE_BOUNDARY_PATH,
    _build_map,
    _load_geojson,
)


def test_boundary_assets_cover_georgia() -> None:
    counties = _load_geojson(_COUNTY_BOUNDARIES_PATH)
    state = _load_geojson(_STATE_BOUNDARY_PATH)

    assert len(counties["features"]) == 159
    assert len(state["features"]) == 1
    assert state["features"][0]["properties"]["NAME"] == "Georgia"


def test_map_renders_boundary_layers_and_attribution() -> None:
    rendered = _build_map([], {"counties": ["Fulton County"]}).get_root().render()

    assert "Georgia county boundaries" in rendered
    assert "Georgia state boundary" in rendered
    assert "U.S. Census TIGERweb" in rendered
    assert "OpenStreetMap" in rendered
    assert "NAMELSAD10" in rendered
    assert "markerCluster" in rendered
