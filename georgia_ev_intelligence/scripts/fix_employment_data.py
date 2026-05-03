"""
Fix known Phase 1 extraction errors — companies where global headcount was stored
instead of Georgia facility employment.

SOURCE: GNEM Human Validated Excel cross-reference.
These are confirmed outliers where extraction pulled corporate global headcount.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared.db import get_session, Company

# Map: company_name → corrected Georgia employment
CORRECTIONS = {
    # WIKA Group has ~10,000 global employees; Gwinnett facility is ~500
    # Source: GNEM Excel human validated data shows SungEel (650) as highest in Gwinnett
    # WIKA USA Lawrenceville GA facility: ~500 employees
    "WIKA USA": 500,
}

session = get_session()
try:
    for name, corrected_emp in CORRECTIONS.items():
        company = session.query(Company).filter(
            Company.company_name == name
        ).first()
        if company:
            print(f"Updating {name}: {company.employment} → {corrected_emp}")
            company.employment = float(corrected_emp)
        else:
            print(f"NOT FOUND: {name}")
    session.commit()
    print("Done.")
finally:
    session.close()
