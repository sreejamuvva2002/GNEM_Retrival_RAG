"""Quick diagnostic: tier values in DB and WIKA employment."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from retrievals.sql_retriever import aggregate_employment_by_county
from shared.db import get_session, Company

session = get_session()

# 1. What tier values actually exist?
tiers = session.query(Company.tier).distinct().order_by(Company.tier).all()
print("=== Tiers in DB ===")
for t in tiers:
    cnt = session.query(Company).filter(Company.tier == t.tier).count()
    print(f"  '{t.tier}' → {cnt} companies")

# 2. WIKA USA data quality check
print("\n=== WIKA USA ===")
wika = session.query(Company).filter(Company.company_name.ilike("%WIKA%")).first()
if wika:
    print(f"  Name: {wika.company_name}")
    print(f"  Employment: {wika.employment}")
    print(f"  County: {wika.location_county}")
    print(f"  Tier: {wika.tier}")
else:
    print("  Not found")

session.close()

# 3. Aggregate with exact 'Tier 1' filter
print("\n=== aggregate_employment_by_county(tier='Tier 1') ===")
r = aggregate_employment_by_county(tier="Tier 1")
print(f"  Rows returned: {len(r)}")
if r:
    print(f"  Top 3: {r[:3]}")

# 4. Aggregate with no tier filter (all tiers)
print("\n=== aggregate_employment_by_county(tier=None) — top 5 ===")
r_all = aggregate_employment_by_county(tier=None)
print(f"  Total rows: {len(r_all)}")
for row in r_all[:5]:
    print(f"  {row}")
