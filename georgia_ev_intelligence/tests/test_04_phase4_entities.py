"""
Phase 4 entity extraction regressions.

These tests protect retrieval from false-positive hard filters. The extractor
may keep broad words as keywords, but ambiguous words must not become hard
metadata filters unless the question clearly asks for that field.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase4_agent.entity_extractor import extract


class TestPhase4EntityExtractor(unittest.TestCase):
    def test_supplier_word_does_not_force_supplier_classification(self):
        e = extract(
            "Map all Thermal Management suppliers in Georgia and show which "
            "Primary OEMs they are linked to."
        )
        self.assertEqual(e.ev_role, "Thermal Management")
        self.assertIsNone(e.classification_method)
        self.assertIsNone(e.tier)

    def test_oem_relationship_does_not_force_company_filter(self):
        e = extract(
            "Show the full supplier network linked to Rivian Automotive in "
            "Georgia, broken down by tier and EV Supply Chain Role."
        )
        self.assertIsNone(e.company_name)
        self.assertIn("rivian", e.oem_list)
        self.assertIsNone(e.classification_method)

    def test_generic_oem_words_do_not_force_tier(self):
        e = extract(
            "Which Georgia Battery Cell or Battery Pack suppliers are "
            "sole-sourced by a specific OEM, indicating high dependency risk?"
        )
        self.assertIsNone(e.tier)
        self.assertCountEqual(e.ev_role_list, ["Battery Cell", "Battery Pack"])

    def test_multi_tier_question_keeps_all_requested_tiers(self):
        e = extract(
            "Find Georgia-based Tier 1 or Tier 1/2 suppliers capable of "
            "producing battery electrolytes."
        )
        self.assertCountEqual(e.tier_list, ["Tier 1", "Tier 1/2"])

    def test_negated_roles_are_excluded_not_included(self):
        e = extract(
            "Identify Georgia areas that currently lack Battery Cell or Battery "
            "Pack suppliers but have existing Tier 1 general automotive infrastructure."
        )
        self.assertEqual(e.tier, "Tier 1")
        self.assertEqual(e.ev_role, "General Automotive")
        self.assertCountEqual(e.exclude_ev_role_list, ["Battery Cell", "Battery Pack"])

    def test_hypothetical_new_company_does_not_filter_existing_tier_or_role(self):
        e = extract(
            "For a new Tier 1 battery thermal management company looking to "
            "locate in Georgia, which areas have the highest concentration of "
            "Materials-category suppliers?"
        )
        self.assertIsNone(e.tier)
        self.assertEqual(e.ev_role, "Materials")
        self.assertNotIn("Thermal Management", e.ev_role_list)

    def test_chemical_manufacturing_is_not_facility_type(self):
        e = extract(
            "For an international battery materials company seeking a Georgia "
            "location, which areas have existing chemical manufacturing infrastructure?"
        )
        self.assertIsNone(e.facility_type)
        self.assertIn("chemical", e.product_keywords)

    def test_explicit_facility_type_still_matches(self):
        e = extract(
            "Which Georgia areas have R&D facility types in the automotive sector?"
        )
        self.assertEqual(e.facility_type, "R&D")

    def test_short_oem_names_are_detected(self):
        e = extract("Which Georgia suppliers have Kia contracts?")
        self.assertIn("kia", e.oem_list)

    def test_generic_battery_word_is_not_oem(self):
        e = extract(
            "Find Georgia-based Tier 1 or Tier 1/2 suppliers capable of "
            "producing battery electrolytes."
        )
        self.assertNotIn("battery", e.oem_list)

    def test_indirect_ev_relevance_is_exact_filter(self):
        e = extract(
            "Which companies with over 1,000 employees are indirectly relevant "
            "to the EV sector, and what are their main products/services?"
        )
        self.assertEqual(e.min_employment, 1001)
        self.assertEqual(e.ev_relevance_value, "Indirect")
        self.assertFalse(e.ev_relevant_filter)

    def test_over_employment_is_strictly_greater_than(self):
        e = extract(
            "Find Tier 2/3 suppliers with employment over 300 that are classified "
            "as General Automotive."
        )
        self.assertEqual(e.min_employment, 301)

    def test_negated_ev_specific_presence_is_not_positive_filter(self):
        e = extract(
            "Which Georgia areas have Manufacturing Plant facility types but no "
            "EV-specific production presence?"
        )
        self.assertEqual(e.facility_type, "Manufacturing Plant")
        self.assertEqual(e.ev_relevance_value, "No")
        self.assertFalse(e.ev_relevant_filter)

    def test_word_number_largest_employment_sets_top_n(self):
        e = extract(
            "Which three Georgia companies have the largest employment and are "
            "Thermal Management suppliers?"
        )
        self.assertTrue(e.is_top_n)
        self.assertEqual(e.top_n_limit, 3)
        self.assertEqual(e.ev_role, "Thermal Management")

    def test_thermal_related_products_maps_to_thermal_management_role(self):
        e = extract(
            "Which companies are producing thermal-related products or services?"
        )
        self.assertEqual(e.ev_role, "Thermal Management")


if __name__ == "__main__":
    unittest.main()
