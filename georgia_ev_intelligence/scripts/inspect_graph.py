"""
Graph inspection — what's actually in Neo4j?
Run: venv\\Scripts\\python scripts\\inspect_graph.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase3_graph.graph_loader import get_driver, get_graph_stats

driver = get_driver()
SEP = "-" * 50

with driver.session() as s:

    print("\n=== OEM NODES (what's in graph) ===")
    rows = s.run("MATCH (o:OEM) RETURN o.name AS name ORDER BY o.name").data()
    for r in rows:
        print(f"  {r['name']}")

    print(f"\n=== COMPANIES WITH NO LOCATION ({SEP[:10]}) ===")
    rows = s.run(
        "MATCH (c:Company) WHERE NOT (c)-[:LOCATED_IN]->() RETURN c.name AS name"
    ).data()
    print(f"  Count: {len(rows)}")
    for r in rows:
        print(f"  - {r['name']}")

    print("\n=== TIER DISTRIBUTION ===")
    rows = s.run(
        "MATCH (c:Company)-[:IN_TIER]->(t:Tier) "
        "RETURN t.name AS tier, count(c) AS cnt ORDER BY cnt DESC"
    ).data()
    for r in rows:
        print(f"  {r['tier']:<25} {r['cnt']} companies")

    print("\n=== INDUSTRY GROUPS ===")
    rows = s.run(
        "MATCH (c:Company)-[:IN_INDUSTRY]->(i:IndustryGroup) "
        "RETURN i.name AS industry, count(c) AS cnt ORDER BY cnt DESC"
    ).data()
    for r in rows:
        print(f"  {r['industry']:<40} {r['cnt']}")

    print("\n=== OVERALL STATS ===")
    stats = get_graph_stats()
    print(f"  Company nodes     : {stats.get('nodes_company', 0)}")
    print(f"  Location nodes    : {stats.get('nodes_location', 0)}")
    print(f"  OEM nodes         : {stats.get('nodes_oem', 0)}")
    print(f"  Tier nodes        : {stats.get('nodes_tier', 0)}")
    print(f"  IndustryGroup     : {stats.get('nodes_industrygroup', 0)}")
    print(f"  Product nodes     : {stats.get('nodes_product', 0)}")
    print(f"  Total nodes       : {stats.get('total_nodes', 0)}")
    print(f"  Total rels        : {stats.get('total_rels', 0)}")
