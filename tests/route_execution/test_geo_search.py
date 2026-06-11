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
    assert "c.primary_oems" in sql


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
    assert "DISTINCT ON (c.company)" in sql
    assert "category = %s" in sql
    assert params[:2] == ["Kia Georgia", "Tier 1"]


def test_closest_company_query_orders_by_postgis_distance_without_radius():
    sql, params = geo_search._closest_to_company_query(
        "hyundai motor group",
        {"category": {"operator": "EQUALS", "value": "Tier 1"}},
        10,
    )

    assert "ST_Distance(c.geo, center.geo)" in sql
    assert "ST_DWithin" not in sql
    assert "ORDER BY distance_miles" in sql
    assert params == ["hyundai motor group", "Tier 1"]


def test_company_center_filters_drop_accidental_oem_center_filter():
    filters = geo_search._company_center_filters(
        {
            "question": "Closest companies to Hyundai",
            "resolved_filters": {
                "primary_oems": {"operator": "CONTAINS", "value": "Hyundai"},
                "category": {"operator": "EQUALS", "value": "Tier 1"},
            },
        }
    )

    assert "primary_oems" not in filters
    assert "category" in filters


def test_company_center_filters_keep_explicit_oem_relationship_filter():
    filters = geo_search._company_center_filters(
        {
            "question": "Closest companies linked to Hyundai",
            "resolved_filters": {
                "primary_oems": {"operator": "CONTAINS", "value": "Hyundai"},
            },
        }
    )

    assert "primary_oems" in filters


def test_distance_to_company_query_uses_postgis_for_explicit_targets():
    sql, params = geo_search._distance_to_company_query(
        "kia georgia inc.",
        ["freudenberg-nok", "novelis inc."],
        20,
    )

    assert "ST_Distance(t.geo, center.geo)" in sql
    assert "lower(company) = ANY(%s)" in sql
    assert params == [
        "kia georgia inc.",
        ["freudenberg-nok", "novelis inc."],
    ]


def test_nearby_targets_query_uses_postgis_radius_for_explicit_targets():
    sql, params = geo_search._nearby_targets_query(
        "kia georgia inc.",
        ["freudenberg-nok", "novelis inc."],
        50.0,
        20,
    )

    assert "ST_DWithin(t.geo, center.geo, %s)" in sql
    assert params[:2] == [
        "kia georgia inc.",
        ["freudenberg-nok", "novelis inc."],
    ]
    assert params[-1] == 50.0 * geo_search._METERS_PER_MILE


def test_execute_contextual_nearby_search(monkeypatch):
    monkeypatch.setattr(
        geo_search,
        "_resolve_company_name",
        lambda name: "kia georgia inc.",
    )
    monkeypatch.setattr(
        geo_search,
        "_fetch",
        lambda sql, params, limit=None: [
            {
                "company": "freudenberg-nok",
                "updated_location": "LaGrange, Troup County",
                "distance_miles": 6.7,
            }
        ],
    )

    result = geo_search.execute_geo_search(
        {
            "question": "Which of these companies are near to Kia Georgia?",
            "route": "geo_search",
            "operation": "nearby_search",
            "entities": ["Kia Georgia"],
            "context_entities": ["freudenberg-nok", "novelis inc."],
        }
    )

    assert result.status == "success"
    assert result.evidence["spatial_operation"] == "ST_DWithin_company_targets"
    assert result.evidence["radius_miles"] == 50.0
    assert "freudenberg-nok" in result.answer


def test_contextual_nearby_no_results_mentions_listed_companies(monkeypatch):
    monkeypatch.setattr(
        geo_search,
        "_resolve_company_name",
        lambda name: "kia georgia inc.",
    )
    monkeypatch.setattr(geo_search, "_fetch", lambda sql, params, limit=None: [])

    result = geo_search.execute_geo_search(
        {
            "question": "Which of these companies are near to Kia Georgia?",
            "route": "geo_search",
            "operation": "nearby_search",
            "entities": ["Kia Georgia"],
            "context_entities": ["f&p georgia manufacturing", "immi"],
        }
    )

    assert result.status == "success"
    assert "None of the listed companies are within 50 miles" in result.answer


def test_execute_contextual_distance_search(monkeypatch):
    monkeypatch.setattr(
        geo_search,
        "_resolve_company_name",
        lambda name: "kia georgia inc.",
    )
    monkeypatch.setattr(
        geo_search,
        "_fetch",
        lambda sql, params, limit=None: [
            {
                "company": "freudenberg-nok",
                "updated_location": "LaGrange, Troup County",
                "distance_miles": 18.5,
            }
        ],
    )

    result = geo_search.execute_geo_search(
        {
            "question": "Distance of these companies to Kia georgia",
            "route": "geo_search",
            "operation": "distance_search",
            "entities": ["Kia georgia"],
            "context_entities": ["freudenberg-nok"],
        }
    )

    assert result.status == "success"
    assert result.evidence["spatial_operation"] == "ST_Distance_company_targets"
    assert result.evidence["center"]["name"] == "kia georgia inc."
    assert "18.5 miles" in result.answer


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


def test_geo_answer_includes_primary_oems():
    answer = geo_search._format_rows(
        "the requested filters",
        [
            {
                "company": "Example",
                "updated_location": "Atlanta, Fulton County",
                "primary_oems": "Hyundai Kia",
            }
        ],
    )

    assert "Primary OEMs: Hyundai Kia" in answer


def test_geo_answer_includes_unknown_primary_oems():
    answer = geo_search._format_rows(
        "the requested filters",
        [
            {
                "company": "Example",
                "updated_location": "Gainesville, Hall County",
                "primary_oems": "Unknown",
            }
        ],
    )

    assert "Primary OEMs: Unknown" in answer
