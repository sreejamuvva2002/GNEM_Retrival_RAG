"""Run this to fix the Neon DB schema after column size changes."""
from shared.db import get_engine
from sqlalchemy import text

engine = get_engine()

alterations = [
    "ALTER TABLE gev_companies ALTER COLUMN ev_battery_relevant TYPE VARCHAR(100)",
    "ALTER TABLE gev_companies ALTER COLUMN ev_supply_chain_role TYPE VARCHAR(200)",
    "ALTER TABLE gev_companies ALTER COLUMN industry_group TYPE VARCHAR(200)",
    "ALTER TABLE gev_companies ALTER COLUMN classification_method TYPE VARCHAR(100)",
    "ALTER TABLE gev_companies ALTER COLUMN supplier_affiliation_type TYPE VARCHAR(200)",
    "ALTER TABLE gev_companies ALTER COLUMN primary_oems TYPE VARCHAR(200)",
]

with engine.connect() as conn:
    for sql in alterations:
        conn.execute(text(sql))
        print("OK:", sql[:60])
    conn.commit()

print("\nVerification:")
with engine.connect() as conn:
    r = conn.execute(text("SELECT COUNT(*) FROM gev_companies"))
    print("Total companies:", r.fetchone()[0])
    r2 = conn.execute(text("SELECT column_name, character_maximum_length FROM information_schema.columns WHERE table_name='gev_companies' AND data_type='character varying' ORDER BY column_name"))
    for row in r2:
        print(f"  {row[0]}: VARCHAR({row[1]})")
