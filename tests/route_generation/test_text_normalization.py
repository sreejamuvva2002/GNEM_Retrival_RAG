"""Lock the canonical (slash) normalization forms.

These assertions mirror how ``loader.normalize_dataframe`` builds the stored KB
values. If the loader's per-column cleaners ever change, value resolution would
silently start missing matches — these tests fail loudly instead.
"""
from georgia_ev_intelligence.route_generation.utils import text_normalization as tn


class TestNormalizeForField:
    def test_category_slash_kept_without_space(self):
        # category keeps the slash with NO surrounding space.
        assert tn.normalize_for_field("category", "Tier 2/3") == "Tier 2/3"
        assert tn.normalize_for_field("category", "Tier 1/2") == "Tier 1/2"

    def test_category_preserves_case(self):
        # clean_category does NOT lowercase -> case-insensitive matching is the
        # resolver's responsibility, not the normalizer's.
        assert tn.normalize_for_field("category", "tier 2/3") == "tier 2/3"

    def test_facility_slash_gets_trailing_space(self):
        assert (
            tn.normalize_for_field("primary_facility_type", "Engineering/Manufacturing")
            == "Engineering/ Manufacturing"
        )

    def test_primary_oems_slash_gets_trailing_space(self):
        assert tn.normalize_for_field("primary_oems", "Hyundai/Kia") == "Hyundai/ Kia"

    def test_product_service_slash_gets_space_both_sides(self):
        assert tn.normalize_for_field("product_service", "A/B") == "A / B"

    def test_company_is_lowercased(self):
        assert tn.normalize_for_field("company", "Rivian") == "rivian"

    def test_unlisted_column_falls_back_to_clean_text(self):
        # industry_group is not in COLUMN_NORMALIZERS -> clean_text (whitespace collapse).
        assert (
            tn.normalize_for_field("industry_group", "  Automotive   electronics ")
            == "Automotive electronics"
        )

    def test_location_kept_verbatim(self):
        assert (
            tn.normalize_for_field("updated_location", "West Point, Harris County")
            == "West Point, Harris County"
        )


class TestQuotedPhrases:
    def test_extracts_quoted_substrings(self):
        assert tn.extract_quoted_phrases('Find "A/B" and "C"') == ["A/B", "C"]

    def test_no_quotes_returns_empty(self):
        assert tn.extract_quoted_phrases("no quotes here") == []

    def test_handles_none(self):
        assert tn.extract_quoted_phrases(None) == []


class TestNormalizeQuestion:
    def test_collapses_whitespace_and_lowercases(self):
        out = tn.normalize_question('  Find   "X"  near   Atlanta ')
        assert out["normalized"] == 'Find "X" near Atlanta'
        assert out["lowercase"] == 'find "x" near atlanta'
        assert out["quoted_phrases"] == ["X"]

    def test_empty_question(self):
        out = tn.normalize_question("")
        assert out == {
            "original": "",
            "normalized": "",
            "lowercase": "",
            "quoted_phrases": [],
        }
