from georgia_ev_intelligence.route_execution.executors import geo_search


def test_coordinate_query_uses_postgis_distance_and_filters():
    sql, params = geo_search._coordinate_query(
        33.749,
        -84.388,
        25.0,
        {"category": {"operator": "EQUALS", "value": "Tier 1"}},
        20,
    )

    assert "ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography" in sql
    assert "ST_DWithin(c.geo, center.geo, %s)" in sql
    assert "category = %s" in sql
    assert params[:3] == [-84.388, 33.749, "Tier 1"]
    assert params[-1] == 25.0 * geo_search._METERS_PER_MILE


def test_county_query_uses_polygon_containment():
    sql, params = geo_search._county_containment_query(
        "Fulton",
        {"category": {"operator": "EQUALS", "value": "Tier 1"}},
        100,
    )

    assert "ST_Covers(g.geom, c.geom)" in sql
    assert params == ["Tier 1", "Fulton"]


def test_company_radius_query_applies_structured_filters():
    sql, params = geo_search._company_query(
        "Kia Georgia",
        25.0,
        {"category": {"operator": "EQUALS", "value": "Tier 1"}},
        20,
    )

    assert "ST_DWithin(c.geo, center.geo, %s)" in sql
    assert "category = %s" in sql
    assert params[:2] == ["Kia Georgia", "Tier 1"]


def test_execute_coordinate_search_records_postgis_evidence(monkeypatch):
    monkeypatch.setattr(
        geo_search,
        "_fetch",
        lambda sql, params, limit=None: [{"company": "Example", "distance_miles": 2.5}],
    )

    result = geo_search.execute_geo_search(
        {
            "question": "Show Tier 1 suppliers within 25 miles of 33.749, -84.388",
            "route": "geo_search",
            "resolved_filters": {
                "category": {"operator": "EQUALS", "value": "Tier 1"},
            },
        }
    )

    assert result.status == "success"
    assert result.evidence["spatial_backend"] == "PostGIS"
    assert result.evidence["spatial_operation"] == "ST_DWithin_coordinate"
    assert "ST_DWithin" in result.evidence["sql_commands"][0]["sql"]
