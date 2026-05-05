import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

# Load env
env_path = Path(".env")
if env_path.exists():
    load_dotenv(env_path)

db_url = os.environ.get("DATABASE_URL")
if not db_url:
    print("Error: DATABASE_URL not found in .env")
    sys.exit(1)

# Connect to database
print(f"Connecting to database...")
engine = create_engine(db_url)

try:
    # Get all table names
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print("\n--- Tables in Database ---")
    if not tables:
        print("No tables found in the database!")
    for table in tables:
        print(f"- {table}")
        
    # Check specific table
    target_table = "gev_companies"
    if target_table in tables:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {target_table}"))
            count = result.scalar()
            print(f"\n✅ Table '{target_table}' exists and has {count} rows.")
            
            # Print a few columns to verify structure
            print("\nColumns in gev_companies:")
            cols = inspector.get_columns(target_table)
            for col in cols:
                print(f"  - {col['name']} ({col['type']})")
    else:
        print(f"\n❌ Table '{target_table}' DOES NOT EXIST in this database.")
        
except Exception as e:
    print(f"\nError accessing database: {e}")
