"""
Check what DB columns have data and what needs to be added.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.db import get_session, Company

s = get_session()

total = s.query(Company).count()
print(f"Total companies: {total}")

# Check each key column
for col_name, col in [
    ("ev_battery_relevant", Company.ev_battery_relevant),
    ("industry_group", Company.industry_group),
    ("supplier_affiliation_type", Company.supplier_affiliation_type),
    ("classification_method", Company.classification_method),
]:
    filled = s.query(Company).filter(col.isnot(None)).count()
    vals = [r[0] for r in s.query(col).distinct().all() if r[0]]
    print(f"\n{col_name}: {filled}/{total} filled")
    print(f"  Distinct values: {vals[:15]}")

# Check if facility_type column exists at all
try:
    ft = s.query(Company.facility_type).limit(1).all()
    print("\nfacility_type: column EXISTS")
except Exception as e:
    print(f"\nfacility_type: MISSING from DB — {e}")

s.close()
