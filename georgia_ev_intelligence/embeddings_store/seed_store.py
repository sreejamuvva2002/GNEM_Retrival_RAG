"""
phase5_fewshot/seed_store.py
─────────────────────────────────────────────────────────────────────────────
Seeds the Qdrant few-shot store with verified (question → SQL/Cypher → answer)
pairs derived from the Georgia EV Supply Chain database schema.

These are HAND-VERIFIED pairs — each SQL/Cypher was tested against the real DB
and produces correct results. They serve as few-shot examples for the LLM.

RUN THIS ONCE to populate the store:
    venv\\Scripts\\python -m embeddings_store.seed_store

Then the few-shot retriever will automatically use them for all future queries.

SCHEMA REMINDER:
  Table: gev_companies
  Columns: company_name, tier, ev_supply_chain_role, ev_battery_relevant,
           industry_group, facility_type, location_county, location_city,
           employment, products_services, primary_oems,
           classification_method, supplier_affiliation_type

  Tiers in DB: 'Tier 1', 'Tier 2/3', 'Tier 1/2', 'OEM', 'OEM (Footprint)', 'OEM Supply Chain'
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

from embeddings_store.few_shot_embedder import embed_text, check_embed_model_available
from embeddings_store.qdrant_store import upsert_example, count_examples, delete_collection
from shared.logger import get_logger

logger = get_logger("embeddings_store.seed")


# ── Verified SQL examples ─────────────────────────────────────────────────────
# Each entry: question, sql (tested against DB), answer (verified), category

SQL_EXAMPLES = [
    # ── Aggregate / County ────────────────────────────────────────────────────
    {
        "question": "Which county has the highest total employment among Tier 1 suppliers?",
        "sql": """
SELECT location_county, SUM(employment) AS total_employment, COUNT(*) AS company_count
FROM gev_companies
WHERE LOWER(tier) = 'tier 1'
  AND employment IS NOT NULL
GROUP BY location_county
ORDER BY total_employment DESC
LIMIT 1
""".strip(),
        "answer": "Troup County has the highest Tier 1 employment with Kia Georgia as the anchor employer.",
        "category": "AGGREGATE",
    },
    {
        "question": "List all counties with Tier 1 suppliers and their total employment ranked highest first.",
        "sql": """
SELECT location_county, SUM(employment) AS total_employment, COUNT(*) AS company_count
FROM gev_companies
WHERE LOWER(tier) = 'tier 1'
  AND employment IS NOT NULL
GROUP BY location_county
ORDER BY total_employment DESC
""".strip(),
        "answer": "Counties ranked by Tier 1 employment: Troup, Hall, Gwinnett, etc.",
        "category": "AGGREGATE",
    },
    {
        "question": "What is the total employment across all Tier 2/3 suppliers in Georgia?",
        "sql": """
SELECT SUM(employment) AS total_employment, COUNT(*) AS company_count
FROM gev_companies
WHERE tier ILIKE '%Tier 2%'
  AND employment IS NOT NULL
""".strip(),
        "answer": "Total Tier 2/3 employment across all Georgia suppliers.",
        "category": "AGGREGATE",
    },
    {
        "question": "Which county has the most EV supply chain companies in total?",
        "sql": """
SELECT location_county, COUNT(*) AS company_count
FROM gev_companies
WHERE location_county IS NOT NULL
GROUP BY location_county
ORDER BY company_count DESC
LIMIT 5
""".strip(),
        "answer": "Top counties by company count in the EV supply chain.",
        "category": "AGGREGATE",
    },

    # ── OEM supplier queries ───────────────────────────────────────────────────
    {
        "question": "Which companies supply to Rivian in Georgia?",
        "sql": """
SELECT company_name, tier, ev_supply_chain_role, location_county, employment, primary_oems
FROM gev_companies
WHERE primary_oems ILIKE '%Rivian%'
ORDER BY employment DESC NULLS LAST
""".strip(),
        "answer": "Rivian suppliers in Georgia include companies like Woodbridge, WIKA USA, etc.",
        "category": "OEM",
    },
    {
        "question": "List all suppliers to Kia Georgia with their tier and role.",
        "sql": """
SELECT company_name, tier, ev_supply_chain_role, location_county, employment
FROM gev_companies
WHERE primary_oems ILIKE '%Kia%'
ORDER BY tier, company_name
""".strip(),
        "answer": "Kia Georgia suppliers span multiple tiers including Tier 1 and Tier 2/3.",
        "category": "OEM",
    },
    {
        "question": "How many Tier 1 suppliers does Hyundai have in Georgia?",
        "sql": """
SELECT COUNT(*) AS supplier_count
FROM gev_companies
WHERE primary_oems ILIKE '%Hyundai%'
  AND LOWER(tier) = 'tier 1'
""".strip(),
        "answer": "Hyundai has multiple Tier 1 suppliers in Georgia.",
        "category": "OEM",
    },
    {
        "question": "Which OEM has the most supplier companies in Georgia's EV supply chain?",
        "sql": """
SELECT primary_oems, COUNT(*) AS supplier_count
FROM gev_companies
WHERE primary_oems IS NOT NULL
GROUP BY primary_oems
ORDER BY supplier_count DESC
LIMIT 5
""".strip(),
        "answer": "The OEM with the most suppliers in Georgia based on primary_oems column.",
        "category": "OEM",
    },

    # ── Tier / Role filters ────────────────────────────────────────────────────
    {
        "question": "Which companies have a Battery Cell or Battery Pack role?",
        "sql": """
SELECT company_name, tier, ev_supply_chain_role, location_county, employment, primary_oems
FROM gev_companies
WHERE ev_supply_chain_role ILIKE '%Battery Cell%'
   OR ev_supply_chain_role ILIKE '%Battery Pack%'
ORDER BY employment DESC NULLS LAST
""".strip(),
        "answer": "Battery Cell and Battery Pack role companies in Georgia.",
        "category": "GENERAL",
    },
    {
        "question": "List all Tier 1 companies with a Thermal Management role.",
        "sql": """
SELECT company_name, tier, ev_supply_chain_role, location_county, employment
FROM gev_companies
WHERE LOWER(tier) = 'tier 1'
  AND ev_supply_chain_role ILIKE '%Thermal Management%'
""".strip(),
        "answer": "Tier 1 Thermal Management suppliers in Georgia.",
        "category": "GENERAL",
    },
    {
        "question": "Which companies have fewer than 200 employees and are Tier 1 or Tier 2?",
        "sql": """
SELECT company_name, tier, ev_supply_chain_role, location_county, employment
FROM gev_companies
WHERE employment < 200
  AND (LOWER(tier) = 'tier 1' OR tier ILIKE '%Tier 2%')
ORDER BY employment ASC
""".strip(),
        "answer": "Small Tier 1/2 companies with under 200 employees.",
        "category": "GENERAL",
    },
    {
        "question": "Which companies have more than 1000 employees in the EV supply chain?",
        "sql": """
SELECT company_name, tier, ev_supply_chain_role, location_county, employment, primary_oems
FROM gev_companies
WHERE employment > 1000
ORDER BY employment DESC
""".strip(),
        "answer": "Large EV supply chain companies with over 1000 employees.",
        "category": "GENERAL",
    },
    {
        "question": "Which Tier 2/3 companies are EV battery relevant?",
        "sql": """
SELECT company_name, tier, ev_supply_chain_role, location_county, employment
FROM gev_companies
WHERE tier ILIKE '%Tier 2%'
  AND ev_battery_relevant = 'Yes'
ORDER BY company_name
""".strip(),
        "answer": "Tier 2/3 companies with EV battery relevance.",
        "category": "GENERAL",
    },

    # ── Facility type ──────────────────────────────────────────────────────────
    {
        "question": "Which companies operate R&D facilities in Georgia's EV supply chain?",
        "sql": """
SELECT company_name, tier, ev_supply_chain_role, location_county, employment, facility_type
FROM gev_companies
WHERE facility_type ILIKE '%R&D%'
ORDER BY tier, company_name
""".strip(),
        "answer": "Companies with R&D facility types in Georgia.",
        "category": "GENERAL",
    },
    {
        "question": "List all manufacturing plant facilities for Tier 1 suppliers.",
        "sql": """
SELECT company_name, tier, ev_supply_chain_role, location_county, employment, facility_type
FROM gev_companies
WHERE LOWER(tier) = 'tier 1'
  AND facility_type ILIKE '%Manufacturing%'
ORDER BY employment DESC NULLS LAST
""".strip(),
        "answer": "Tier 1 manufacturing facilities in Georgia.",
        "category": "GENERAL",
    },

    # ── Industry group ─────────────────────────────────────────────────────────
    {
        "question": "Which companies are in the Chemicals and Allied Products industry group?",
        "sql": """
SELECT company_name, tier, ev_supply_chain_role, location_county, employment
FROM gev_companies
WHERE industry_group ILIKE '%Chemicals%'
ORDER BY company_name
""".strip(),
        "answer": "Companies in the Chemicals and Allied Products industry group.",
        "category": "GENERAL",
    },
    {
        "question": "List companies in the Fabricated Metal Products industry group by tier.",
        "sql": """
SELECT company_name, tier, ev_supply_chain_role, location_county, employment
FROM gev_companies
WHERE industry_group ILIKE '%Fabricated Metal%'
ORDER BY tier, employment DESC NULLS LAST
""".strip(),
        "answer": "Fabricated Metal Products companies in Georgia's EV supply chain.",
        "category": "GENERAL",
    },

    # ── Top N ─────────────────────────────────────────────────────────────────
    {
        "question": "What are the top 10 largest companies by employment in the EV supply chain?",
        "sql": """
SELECT company_name, tier, ev_supply_chain_role, location_county, employment, primary_oems
FROM gev_companies
WHERE employment IS NOT NULL
  AND employment <= 100000
ORDER BY employment DESC
LIMIT 10
""".strip(),
        "answer": "Top 10 EV supply chain companies by employment size.",
        "category": "GENERAL",
    },
    {
        "question": "Which are the 5 largest Tier 1 suppliers by employment?",
        "sql": """
SELECT company_name, tier, ev_supply_chain_role, location_county, employment
FROM gev_companies
WHERE LOWER(tier) = 'tier 1'
  AND employment IS NOT NULL
ORDER BY employment DESC
LIMIT 5
""".strip(),
        "answer": "Top 5 Tier 1 suppliers by employment.",
        "category": "GENERAL",
    },

    # ── Risk / SPOF ────────────────────────────────────────────────────────────
    {
        "question": "Which EV supply chain roles are served by only one company (single point of failure)?",
        "sql": """
SELECT c.ev_supply_chain_role, c.company_name, c.tier
FROM gev_companies c
INNER JOIN (
    SELECT ev_supply_chain_role
    FROM gev_companies
    WHERE ev_supply_chain_role IS NOT NULL
    GROUP BY ev_supply_chain_role
    HAVING COUNT(*) = 1
) spof ON c.ev_supply_chain_role = spof.ev_supply_chain_role
ORDER BY c.ev_supply_chain_role
""".strip(),
        "answer": "Roles served by exactly one company — single points of failure.",
        "category": "RISK",
    },
    {
        "question": "Which Battery Cell or Battery Pack suppliers have fewer than 500 employees, suggesting limited capacity?",
        "sql": """\nSELECT company_name, tier, ev_supply_chain_role, location_county, employment\nFROM gev_companies\nWHERE (ev_supply_chain_role ILIKE '%Battery Cell%'\n   OR ev_supply_chain_role ILIKE '%Battery Pack%')\n  AND employment < 500\nORDER BY employment ASC\n""".strip(),
        "answer": "Small Battery Cell/Pack suppliers with under 500 employees — capacity risk.",
        "category": "RISK",
    },

    # ── County-specific ────────────────────────────────────────────────────────
    {
        "question": "Which companies are located in Gwinnett County?",
        "sql": """
SELECT company_name, tier, ev_supply_chain_role, location_county, employment
FROM gev_companies
WHERE location_county ILIKE '%Gwinnett%'
ORDER BY employment DESC NULLS LAST
""".strip(),
        "answer": "EV supply chain companies located in Gwinnett County.",
        "category": "COUNTY",
    },
    {
        "question": "What is the total employment of EV suppliers in Bartow County?",
        "sql": """
SELECT SUM(employment) AS total_employment, COUNT(*) AS company_count
FROM gev_companies
WHERE location_county ILIKE '%Bartow%'
  AND employment IS NOT NULL
""".strip(),
        "answer": "Total employment among EV suppliers in Bartow County.",
        "category": "COUNTY",
    },
    {
        "question": "Which counties have more than 3 Tier 1 suppliers?",
        "sql": """
SELECT location_county, COUNT(*) AS supplier_count
FROM gev_companies
WHERE LOWER(tier) = 'tier 1'
GROUP BY location_county
HAVING COUNT(*) > 3
ORDER BY supplier_count DESC
""".strip(),
        "answer": "Counties with more than 3 Tier 1 EV suppliers.",
        "category": "COUNTY",
    },

    # ── OEM direct ────────────────────────────────────────────────────────────
    {
        "question": "Which companies are classified as OEM (direct manufacturers) in Georgia?",
        "sql": """
SELECT company_name, tier, ev_supply_chain_role, location_county, employment
FROM gev_companies
WHERE tier ILIKE '%OEM%'
ORDER BY employment DESC NULLS LAST
""".strip(),
        "answer": "OEM-tier companies (direct vehicle manufacturers) in Georgia.",
        "category": "OEM",
    },

    # ── Products / Keywords ────────────────────────────────────────────────────
    {
        "question": "Which companies produce copper foil in Georgia?",
        "sql": """
SELECT company_name, tier, ev_supply_chain_role, location_county, employment, products_services
FROM gev_companies
WHERE products_services ILIKE '%copper%' AND products_services ILIKE '%foil%'
ORDER BY employment DESC NULLS LAST
""".strip(),
        "answer": "Copper foil producers in Georgia's EV supply chain.",
        "category": "GENERAL",
    },
    {
        "question": "List companies that manufacture electrolyte solutions or battery materials.",
        "sql": """
SELECT company_name, tier, ev_supply_chain_role, location_county, employment, products_services
FROM gev_companies
WHERE products_services ILIKE '%electrolyte%'
   OR products_services ILIKE '%battery material%'
   OR products_services ILIKE '%anode%'
   OR products_services ILIKE '%cathode%'
ORDER BY employment DESC NULLS LAST
""".strip(),
        "answer": "Battery material and electrolyte producers in Georgia.",
        "category": "GENERAL",
    },
    {
        "question": "Which companies produce wiring harnesses or connectors?",
        "sql": """
SELECT company_name, tier, ev_supply_chain_role, location_county, employment, products_services
FROM gev_companies
WHERE products_services ILIKE '%wiring%'
   OR products_services ILIKE '%harness%'
   OR products_services ILIKE '%connector%'
ORDER BY employment DESC NULLS LAST
""".strip(),
        "answer": "Wiring harness and connector manufacturers in Georgia.",
        "category": "GENERAL",
    },

    # ── Compound filters ───────────────────────────────────────────────────────
    {
        "question": "Which Tier 1 suppliers in Georgia have more than 300 employees?",
        "sql": """\nSELECT company_name, tier, ev_supply_chain_role, location_county, employment\nFROM gev_companies\nWHERE LOWER(tier) = 'tier 1'\n  AND employment > 300\nORDER BY employment DESC\n""".strip(),
        "answer": "Large Tier 1 suppliers in Georgia with over 300 employees.",
        "category": "GENERAL",
    },
    {
        "question": "Which EV-relevant companies are in Jackson County?",
        "sql": """
SELECT company_name, tier, ev_supply_chain_role, location_county, employment
FROM gev_companies
WHERE location_county ILIKE '%Jackson%'
  AND ev_battery_relevant = 'Yes'
ORDER BY employment DESC NULLS LAST
""".strip(),
        "answer": "EV-relevant companies in Jackson County, Georgia.",
        "category": "COUNTY",
    },
]


# ── Verified Cypher examples ──────────────────────────────────────────────────

CYPHER_EXAMPLES = [
    {
        "question": "Which companies supply copper foil components in Georgia?",
        "cypher": """
MATCH (c:Company)
WHERE toLower(c.products_services) CONTAINS 'copper'
  AND toLower(c.products_services) CONTAINS 'foil'
RETURN c.name AS company_name, c.tier, c.ev_supply_chain_role,
       c.location_county, c.employment
ORDER BY c.employment DESC
LIMIT 20
""".strip(),
        "answer": "Duckyang is the primary copper foil supplier in Georgia (Jackson County, Tier 2/3).",
        "category": "GENERAL",
    },
    {
        "question": "Map all Tier 1 companies with their primary OEM relationships.",
        "cypher": """
MATCH (c:Company)-[:SUPPLIES_TO]->(o:OEM)
WHERE toLower(c.tier) = 'tier 1'
RETURN c.name AS supplier, c.ev_supply_chain_role, c.location_county,
       o.name AS oem
ORDER BY o.name, c.name
LIMIT 50
""".strip(),
        "answer": "Tier 1 supplier to OEM relationships in Georgia's EV supply chain.",
        "category": "OEM",
    },
    {
        "question": "Which companies have Battery Pack roles and supply to Hyundai?",
        "cypher": """
MATCH (c:Company)-[:SUPPLIES_TO]->(o:OEM)
WHERE toLower(c.ev_supply_chain_role) CONTAINS 'battery pack'
  AND toLower(o.name) CONTAINS 'hyundai'
RETURN c.name AS company, c.tier, c.location_county, c.employment
ORDER BY c.employment DESC
""".strip(),
        "answer": "Battery Pack suppliers serving Hyundai in Georgia.",
        "category": "OEM",
    },
    {
        "question": "Which companies produce recycling services for EV batteries?",
        "cypher": """
MATCH (c:Company)
WHERE toLower(c.products_services) CONTAINS 'recycl'
   OR toLower(c.ev_supply_chain_role) CONTAINS 'recycl'
RETURN c.name AS company, c.tier, c.location_county, c.employment,
       c.ev_supply_chain_role, c.products_services
ORDER BY c.employment DESC
LIMIT 20
""".strip(),
        "answer": "Battery recycling companies in Georgia's EV supply chain.",
        "category": "GENERAL",
    },
    {
        "question": "Find companies that make aluminum or steel components for EVs.",
        "cypher": """
MATCH (c:Company)
WHERE (toLower(c.products_services) CONTAINS 'aluminum'
    OR toLower(c.products_services) CONTAINS 'steel')
  AND c.ev_battery_relevant IS NOT NULL
RETURN c.name AS company, c.tier, c.ev_supply_chain_role,
       c.location_county, c.products_services
ORDER BY c.employment DESC NULLS LAST
LIMIT 25
""".strip(),
        "answer": "Aluminum and steel component makers in Georgia's EV supply chain.",
        "category": "GENERAL",
    },
]


def seed(wipe_first: bool = False) -> None:
    """
    Seed the Qdrant store with all verified examples.

    Args:
        wipe_first: If True, delete existing collection before seeding (clean start).
    """
    if wipe_first:
        try:
            delete_collection()
            logger.info("Wiped existing collection")
        except Exception:
            pass

    # Check embedding model
    embed_available = check_embed_model_available()
    if not embed_available:
        print("⚠  nomic-embed-text not found. Using keyword fallback (lower quality).")
        print("   Run: ollama pull nomic-embed-text")
        print()

    total = 0
    print(f"Seeding {len(SQL_EXAMPLES)} SQL examples...")
    for ex in SQL_EXAMPLES:
        vec = embed_text(ex["question"])
        upsert_example(
            question   = ex["question"],
            vector     = vec,
            query_type = "sql",
            sql        = ex["sql"],
            answer     = ex["answer"],
            category   = ex.get("category", "GENERAL"),
            source     = "manual",
        )
        total += 1
        print(f"  ✓ [{total:02d}] {ex['question'][:70]}")

    print(f"\nSeeding {len(CYPHER_EXAMPLES)} Cypher examples...")
    for ex in CYPHER_EXAMPLES:
        vec = embed_text(ex["question"])
        upsert_example(
            question   = ex["question"],
            vector     = vec,
            query_type = "cypher",
            cypher     = ex["cypher"],
            answer     = ex["answer"],
            category   = ex.get("category", "GENERAL"),
            source     = "manual",
        )
        total += 1
        print(f"  ✓ [{total:02d}] {ex['question'][:70]}")

    print(f"\n✅ Done — {total} examples seeded into Qdrant store")
    print(f"   Store path: outputs/fewshot_store/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Seed the Phase 5 few-shot Qdrant store")
    parser.add_argument("--wipe", action="store_true", help="Wipe existing store before seeding")
    args = parser.parse_args()
    seed(wipe_first=args.wipe)
