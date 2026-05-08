"""
Phase 4 KB-first retrieval regressions.

These tests protect spreadsheet-style KB questions from falling through to
semantic vector ranking. They run against an in-memory company set, not Qdrant.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from filters_and_validation.query_entity_extractor import extract
from core_agent.agent_pipeline import EVAgent, _parse_company_context
from retrievals.vector_retriever import retrieve_context


def company(
    name: str,
    *,
    tier: str = "Tier 1",
    role: str = "General Automotive",
    city: str = "Atlanta",
    county: str = "Fulton County",
    employment: float = 100,
    oems: str = "Multiple OEMs",
    ev: str = "No",
    industry: str = "Transportation Equipment",
    facility: str = "Manufacturing Plant",
    products: str = "",
    classification: str = "",
) -> dict:
    return {
        "company_row_id": name,
        "company_name": name,
        "tier": tier,
        "ev_supply_chain_role": role,
        "primary_oems": oems,
        "ev_battery_relevant": ev,
        "industry_group": industry,
        "facility_type": facility,
        "location_city": city,
        "location_county": county,
        "location_state": "Georgia",
        "employment": employment,
        "products_services": products,
        "classification_method": classification,
        "supplier_affiliation_type": "",
        "text": (
            f"Company: {name} | Tier: {tier} | Role: {role} | "
            f"Location: {city}, {county} | Products: {products}"
        ),
    }


SAMPLE_KB = [
    company(
        "WIKA USA",
        tier="OEM Supply Chain",
        role="HV and LV wiring harnesses for EVs and ICE vehicles",
        city="Lawrenceville",
        county="Gwinnett County",
        employment=250000,
        ev="Yes",
        products="Vehicle power & data solutions; wiring harnesses and connectors",
    ),
    company(
        "Woodbridge Foam Corp.",
        tier="OEM Supply Chain",
        role="EV wiring harnesses and power distribution",
        city="Atlanta",
        county="Fulton County",
        employment=1657,
        ev="Yes",
        products="Automotive wiring harnesses & electrical distribution systems",
    ),
    company(
        "Solvay Specialty Polymers USA LLC",
        tier="Tier 1",
        role="General Automotive",
        city="Alpharetta",
        county="Fulton County",
        employment=70,
        ev="Indirect",
        products="Automotive wire harnesses",
    ),
    company(
        "Yazaki North America",
        tier="OEM Footprint",
        role="Power electronics, sensors, and EV systems",
        city="Griffin",
        county="Spalding County",
        employment=230000,
        ev="Yes",
        products="Automotive supplier division mobility solutions",
    ),
    company(
        "Kia Georgia Inc.",
        tier="OEM",
        role="Vehicle Assembly",
        city="West Point",
        county="Troup County",
        employment=2800,
        oems="Club Car LLC",
        ev="Yes",
        products="Vehicle assembly operations",
        classification="Direct Manufacturer",
    ),
    company(
        "Arising Industries Inc.",
        tier="Tier 2/3",
        city="Norcross",
        county="Gwinnett County",
        employment=582,
        ev="Indirect",
        industry="Chemicals and Allied Products",
        products="Automotive and aircraft tires",
    ),
    company(
        "Archer Aviation Inc.",
        tier="Tier 2/3",
        city="Covington",
        county="Morgan County",
        employment=200,
        ev="Indirect",
        industry="Chemicals and Allied Products",
        products="Powder coatings for automotive and other industries",
    ),
    company(
        "GSC Steel Stamping LLC",
        tier="Tier 2/3",
        role="Power Electronics",
        city="Adairsville",
        county="Bartow County",
        employment=350,
        oems="Hyundai Kia Rivian",
        ev="Yes",
        products="DC-to-DC converters for EV power electronics",
    ),
    company(
        "ZF Gainesville LLC",
        tier="OEM Supply Chain",
        role="EV thermal management and power electronics",
        city="Gainesville",
        county="Hall County",
        employment=17500,
        ev="Yes",
        products="Automotive supplier footprint for electrification and power electronics",
    ),
    company(
        "Duckyang",
        tier="Tier 2/3",
        city="Braselton",
        county="Jackson County",
        employment=250,
        oems="Hyundai Kia Rivian",
        ev="Yes",
        industry="Primary Metal Industries",
        products="High-quality electrodeposited copper foil for electric vehicles",
    ),
    company(
        "Enchem America Inc.",
        tier="Tier 2/3",
        city="Commerce",
        county="Jackson County",
        employment=155,
        oems="Hyundai Kia Rivian",
        ev="Yes",
        products="Electric wiring components",
    ),
    company(
        "Hyundai Transys Georgia Powertrain",
        tier="Tier 1/2",
        role="Thermal Management",
        city="West Point",
        county="Troup County",
        employment=130,
        oems="Hyundai Kia Rivian",
        ev="Yes",
        products="Electrical heaters, control units, and actuators",
    ),
    company(
        "Hyundai MOBIS (Georgia)",
        tier="Tier 1/2",
        role="General Automotive",
        city="West Point",
        county="Troup County",
        employment=170,
        oems="Multiple OEMs",
        ev="Indirect",
        products="Capacitors, electronic automotive components",
    ),
    company(
        "Hyundai Transys Georgia Seating Systems",
        tier="Tier 1/2",
        role="General Automotive",
        city="West Point",
        county="Troup County",
        employment=110,
        oems="Multiple OEMs",
        ev="Indirect",
        products="Automotive electronics, resonators, capacitors, resistors, electronics and electronic parts",
    ),
    company(
        "Generic Tier 1/2 Components Co.",
        tier="Tier 1/2",
        city="Dublin",
        county="Laurens County",
        employment=100,
        products="Electronic automotive components",
    ),
    company(
        "Racemark International LLC",
        tier="Tier 1",
        city="Gray",
        county="Jones County",
        employment=120,
        oems="Hyundai Kia Rivian",
        ev="Yes",
        facility="R&D",
        products="Manufacturing and R&D engine parts for EV",
    ),
    company(
        "JTEKT North America Corp.",
        tier="Tier 1",
        city="Jefferson",
        county="Jackson County",
        employment=860,
        ev="Indirect",
        products="Vehicle roofing systems",
    ),
    company(
        "Kautex Inc.",
        tier="Tier 1",
        city="Tiger",
        county="Rabun County",
        employment=800,
        ev="Indirect",
        products="Automotive exterior and interior systems",
    ),
    company(
        "Large Indirect Supplier",
        tier="Tier 2/3",
        city="Augusta",
        county="Richmond County",
        employment=1200,
        ev="Indirect",
        products="Industrial materials and automotive products",
    ),
    company(
        "Battery Cell Co.",
        tier="Tier 1/2",
        role="Battery Cell",
        city="Savannah",
        county="Chatham County",
        employment=400,
        ev="Yes",
        products="Battery modules",
    ),
    company(
        "F&P Georgia Manufacturing",
        tier="Tier 1/2",
        role="Battery Pack",
        city="Commerce",
        county="Jackson County",
        employment=104,
        oems="Hyundai Kia",
        ev="Yes",
        products="Lithium-ion battery recycler and raw materials provider",
    ),
    company(
        "Hitachi Astemo Americas Inc.",
        tier="Tier 1/2",
        role="Battery Cell",
        city="Monroe",
        county="Walton County",
        employment=723,
        oems="Hyundai Kia",
        ev="Yes",
        products="Battery cells for electric mobility",
    ),
    company(
        "Hollingsworth & Vose Co.",
        tier="Tier 1/2",
        role="Battery Pack",
        city="Hawkinsville",
        county="Pulaski County",
        employment=400,
        oems="Hyundai Kia",
        ev="Yes",
        products="Lithium-ion battery materials",
    ),
    company(
        "Honda Development & Manufacturing",
        tier="Tier 1/2",
        role="Battery Cell",
        city="Tallapoosa",
        county="Haralson County",
        employment=400,
        oems="Hyundai Kia",
        ev="Yes",
        products="Battery cells for electric mobility",
    ),
    company(
        "Hyundai Motor Group",
        tier="Tier 1/2",
        role="Battery Pack",
        city="Savannah",
        county="Chatham County",
        employment=164,
        oems="Hyundai Kia",
        ev="Yes",
        products="Battery parts for electric vehicles",
    ),
    company(
        "IMMI",
        tier="Tier 1/2",
        role="Battery Pack",
        city="Athens",
        county="Clarke County",
        employment=100,
        oems="Hyundai Kia",
        ev="Yes",
        products="Battery electrolyte",
    ),
    company(
        "EVCO Plastics",
        tier="Tier 2/3",
        role="Materials",
        city="Calhoun",
        county="Gordon County",
        employment=125,
        ev="No",
        products="Recycler of copper, precious metals, and non-ferrous materials",
    ),
    company(
        "Enplas USA Inc.",
        tier="Tier 2/3",
        role="General Automotive",
        city="Atlanta",
        county="Fulton County",
        employment=150,
        ev="No",
        products="Recycler of lithium ion batteries",
    ),
    company(
        "General Auto Fulton",
        tier="Tier 1",
        city="Atlanta",
        county="Fulton County",
        employment=200,
        products="Seat and interior parts",
    ),
    company(
        "General Auto Chatham",
        tier="Tier 1",
        city="Savannah",
        county="Chatham County",
        employment=200,
        products="Brake pads",
    ),
    company(
        "Exact 300 Tier 2/3",
        tier="Tier 2/3",
        role="General Automotive",
        city="Atlanta",
        county="Fulton County",
        employment=300,
        products="Automotive components",
    ),
    company(
        "Above 300 Tier 2/3",
        tier="Tier 2/3",
        role="General Automotive",
        city="Atlanta",
        county="Fulton County",
        employment=301,
        products="Automotive components",
    ),
    company(
        "DAEHAN Solution Georgia LLC",
        tier="Tier 2/3",
        role="Materials",
        city="Covington",
        county="Henry County",
        employment=130,
        ev="Indirect",
        products="Chemically treated industrial yarns and rubber compounds",
    ),
    company(
        "Second Materials Co.",
        tier="Tier 2/3",
        role="Materials",
        city="Covington",
        county="Henry County",
        employment=90,
        ev="Indirect",
        products="Composite materials for automotive components",
    ),
    company(
        "Non EV Plant A",
        city="LaGrange",
        county="Troup County",
        employment=50,
        ev="No",
        products="Seats",
    ),
    company(
        "Non EV Plant B",
        city="LaGrange",
        county="Troup County",
        employment=60,
        ev="No",
        products="Trim",
    ),
    company(
        "EV Plant",
        city="West Point",
        county="Troup County",
        employment=70,
        ev="Yes",
        products="EV parts",
    ),
    company(
        "Non EV Plant C",
        city="West Point",
        county="Troup County",
        employment=65,
        ev="No",
        products="ICE parts",
    ),
]


class TestKBFirstRetrieval(unittest.TestCase):
    def retrieve(self, question: str):
        with patch("retrievals.vector_retriever._load_all_companies", return_value=SAMPLE_KB):
            return retrieve_context(question, extract(question))

    def names(self, result) -> set[str]:
        self.assertIsInstance(result, list)
        return {row["company_name"] for row in result}

    def test_highest_employment_in_county_is_deterministic(self):
        result = self.retrieve(
            "In Gwinnett County, which company has the highest Employment and "
            "what is its EV Supply Chain Role?"
        )
        self.assertEqual(result[0]["company_name"], "WIKA USA")

    def test_product_contains_powder_coating(self):
        result = self.retrieve(
            "Which Georgia companies provide powder coating-related products or "
            "services, and what tier are they classified under?"
        )
        self.assertEqual(self.names(result), {"Archer Aviation Inc."})

    def test_role_list_expands_power_electronics_substrings(self):
        result = self.retrieve(
            "List every Georgia company classified under Power Electronics or "
            "Charging Infrastructure, along with their Employment size."
        )
        names = self.names(result)
        self.assertIn("GSC Steel Stamping LLC", names)
        self.assertIn("ZF Gainesville LLC", names)

    def test_quoted_product_terms_find_exact_product_signal(self):
        result = self.retrieve(
            "Identify Georgia companies whose product descriptions include "
            "'high-voltage', 'DC-to-DC', 'inverter', or 'motor controller'."
        )
        self.assertEqual(self.names(result), {"GSC Steel Stamping LLC"})

    def test_generic_product_words_do_not_create_false_positive(self):
        result = self.retrieve(
            "Which Georgia Tier 1/2 companies produce engineered plastics, "
            "polymers, or composite materials applicable to EV structural or "
            "thermal components?"
        )
        self.assertEqual(result, "No matching companies found.")

    def test_dual_oem_capability_uses_primary_oems_field(self):
        result = self.retrieve(
            "Which Georgia suppliers currently serving traditional OEMs are "
            "also linked to EV-native OEMs, showing dual-platform supply capability?"
        )
        names = self.names(result)
        self.assertIn("Duckyang", names)
        self.assertIn("GSC Steel Stamping LLC", names)
        self.assertIn("Racemark International LLC", names)

    def test_area_set_difference_uses_full_kb(self):
        result = self.retrieve(
            "Identify Georgia areas that currently lack Battery Cell or Battery "
            "Pack suppliers but have existing Tier 1 general automotive infrastructure."
        )
        self.assertIsInstance(result, str)
        self.assertIn("Fulton County", result)
        self.assertNotIn("Chatham County", result)

    def test_materials_area_concentration_groups_by_city_and_county(self):
        result = self.retrieve(
            "For a new Tier 1 battery thermal management company looking to "
            "locate in Georgia, which areas have the highest concentration of "
            "Materials-category suppliers that could support thermal management production?"
        )
        self.assertIsInstance(result, str)
        self.assertIn("Covington, Henry County: 2 companies", result)

    def test_manufacturing_plant_without_ev_groups_by_area(self):
        result = self.retrieve(
            "How many Georgia areas have concentrated Manufacturing Plant "
            "facilities but no EV-specific production presence?"
        )
        self.assertIsInstance(result, str)
        self.assertIn("LaGrange, Troup County: 2 companies", result)
        self.assertIn("West Point, Troup County", result)

    def test_oem_footprint_parentheses_are_normalized(self):
        result = self.retrieve(
            "Identify any EV-relevant Georgia companies classified as OEM "
            "Footprint or OEM Supply Chain?"
        )
        names = self.names(result)
        self.assertIn("Yazaki North America", names)
        self.assertIn("WIKA USA", names)

    def test_exact_indirect_ev_relevance_filters_structured_results(self):
        question = (
            "Which companies with over 1,000 employees are indirectly relevant "
            "to the EV sector, and what are their main products/services?"
        )
        with patch("retrievals.vector_retriever._load_all_companies", return_value=SAMPLE_KB), \
             patch("retrievals.vector_retriever._semantic_rank", return_value=({}, [])):
            result = retrieve_context(question, extract(question))
        self.assertEqual(self.names(result), {"Large Indirect Supplier"})

    def test_role_related_to_wiring_harnesses_searches_role_field(self):
        result = self.retrieve(
            "Identify all Georgia companies with an EV Supply Chain Role related "
            "to wiring harnesses and show their Primary OEMs."
        )
        self.assertEqual(self.names(result), {"WIKA USA", "Woodbridge Foam Corp."})

    def test_bev_wiring_harness_question_uses_exact_kb_text_filters(self):
        result = self.retrieve(
            "Which Georgia companies manufacture high-voltage wiring harnesses or "
            "EV electrical distribution components suitable for BEV platforms?"
        )
        self.assertEqual(self.names(result), {"WIKA USA", "Woodbridge Foam Corp."})

    def test_power_electronics_component_question_uses_specific_product_terms(self):
        result = self.retrieve(
            "Identify Georgia companies producing DC-to-DC converters, capacitors, "
            "or power electronics components relevant to EV drivetrains and what "
            "tier is each assigned?"
        )
        self.assertEqual(
            self.names(result),
            {
                "GSC Steel Stamping LLC",
                "Hyundai MOBIS (Georgia)",
                "Hyundai Transys Georgia Seating Systems",
            },
        )

    def test_over_300_excludes_exactly_300_employment(self):
        result = self.retrieve(
            "Find Tier 2/3 Georgia-based suppliers with employment over 300 that "
            "are classified as General Automotive but produce components "
            "transferable to EV platforms."
        )
        names = self.names(result)
        self.assertIn("Above 300 Tier 2/3", names)
        self.assertNotIn("Exact 300 Tier 2/3", names)

    def test_multiple_oems_exact_list_skips_soft_filtering(self):
        question = (
            "Identify all Georgia-based Tier 1/2 automotive suppliers that "
            "maintain a diversified customer base (serving 'Multiple OEMs')."
        )
        with patch("retrievals.vector_retriever._load_all_companies", return_value=SAMPLE_KB), \
             patch("retrievals.vector_retriever.interpret_soft_filters") as soft_filter, \
             patch("retrievals.vector_retriever._semantic_rank", return_value=({}, [])), \
             patch("retrievals.vector_retriever.rerank_companies", side_effect=lambda _q, rows: rows):
            result = retrieve_context(question, extract(question))
        soft_filter.assert_not_called()
        names = self.names(result)
        self.assertIn("Hyundai MOBIS (Georgia)", names)
        self.assertIn("Hyundai Transys Georgia Seating Systems", names)

    def test_direct_manufacturer_question_uses_structured_classification(self):
        result = self.retrieve(
            "Which companies are classified as Direct Manufacturer, and what EV "
            "Supply Chain Roles do they cover?"
        )
        self.assertEqual(self.names(result), {"Kia Georgia Inc."})

    def test_battery_materials_question_uses_ev_source_terms(self):
        result = self.retrieve(
            "Which Georgia companies produce battery materials such as anodes, "
            "cathodes, electrolytes, or copper foil, and what tier are they "
            "classified as?"
        )
        self.assertEqual(
            self.names(result),
            {
                "Duckyang",
                "F&P Georgia Manufacturing",
                "Hollingsworth & Vose Co.",
                "IMMI",
            },
        )

    def test_company_context_table_preserves_schema_fields_and_source_products(self):
        """Formatted retrieval context should expose the source facts the judge needs."""
        rows = [
            {
                **SAMPLE_KB[0],
                "classification_method": "Supplier",
                "supplier_affiliation_type": "Automotive supply chain participant",
                "products_services": (
                    "Vehicle power | data solutions with high-voltage wiring "
                    "harnesses, connectors, and EV electrical distribution systems"
                ),
            }
        ]
        context = EVAgent._format_companies(rows)

        self.assertIn("Classification | Affiliation | Products", context)
        self.assertIn("Supplier", context)
        self.assertIn("Automotive supply chain participant", context)
        self.assertIn("high-voltage wiring harnesses", context)
        self.assertNotIn("Vehicle power | data solutions", context)

        parsed = _parse_company_context(context)
        self.assertEqual(parsed[0]["classification_method"], "Supplier")
        self.assertEqual(
            parsed[0]["supplier_affiliation_type"],
            "Automotive supply chain participant",
        )
        self.assertIn("Vehicle power / data solutions", parsed[0]["products_services"])

    def test_lithium_ion_materials_cells_electrolytes_excludes_non_ev_recyclers(self):
        result = self.retrieve(
            "How many Georgia companies are now producing lithium-ion battery "
            "materials, cells, or electrolytes?"
        )
        self.assertEqual(
            self.names(result),
            {
                "F&P Georgia Manufacturing",
                "Hitachi Astemo Americas Inc.",
                "Hollingsworth & Vose Co.",
                "Honda Development & Manufacturing",
                "IMMI",
            },
        )

    def test_battery_recycling_uses_recycler_source_text_without_ev_filter(self):
        result = self.retrieve(
            "Which Georgia companies are involved in battery recycling or "
            "second-life battery processing, reflecting the emerging circular "
            "economy trend?"
        )
        self.assertEqual(
            self.names(result),
            {"Enplas USA Inc.", "EVCO Plastics", "F&P Georgia Manufacturing"},
        )

    def test_tier_one_half_battery_parts_source_terms(self):
        result = self.retrieve(
            "Which Georgia companies manufacture battery parts or enclosure "
            "systems and are classified as Tier 1/2, making them ready for direct "
            "OEM engagement and show which Primary OEMs they are linked to."
        )
        self.assertEqual(
            self.names(result),
            {
                "F&P Georgia Manufacturing",
                "Hitachi Astemo Americas Inc.",
                "Hollingsworth & Vose Co.",
                "Honda Development & Manufacturing",
                "Hyundai Motor Group",
                "IMMI",
            },
        )


if __name__ == "__main__":
    unittest.main()
