# Georgia EV Intelligence — Architecture V2: GraphRAG + Text-to-Cypher

## The Core Problem with V1

We were over-engineering a 193-row database with:
- Fragile stop word lists tuned to specific questions
- Bigram keyword matching that never matches DB text
- Complex entity extraction just to decide which table to query
- Neo4j used for only 3 out of 50 question types

The system was **question-engineering, not data-engineering.**

---

## What the graphrag-finetune Repo Shows Us

The repo (`eswarashish/graphrag-finetune`) uses the exact pattern we should follow:

```
User Question
    ↓
LangChain Agent (decides which tool to use)
    ↓
GraphRAG Service
    ├── Vector Search (semantic similarity in Neo4j)
    ├── Cypher Query (structured traversal)
    └── LLM Synthesis (final answer)
```

Key insight: **The LLM IS the router**. You don't need hand-coded intent detection. The LLM, given tools + schema, decides which query to run.

---

## New Architecture: Text-to-Cypher Primary + SQL Aggregate Fallback

```
Question
    │
    ▼
┌──────────────────────────────────────────────┐
│  STEP 1: Generate Cypher (LLM call #1)       │
│  Input:  question + Neo4j schema + examples  │
│  Output: Cypher query string                 │
│  Time:   ~3-5 seconds                        │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│  STEP 2: Execute Cypher → Neo4j              │
│  If syntax error → self-heal (retry once)    │
│  If 0 results → try PostgreSQL aggregate     │
│  Time: ~0.5 seconds                          │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│  STEP 3: Synthesize Answer (LLM call #2)     │
│  Input:  question + query results            │
│  Output: Final human-readable answer         │
│  Time:   ~10-30 seconds                      │
└──────────────────────────────────────────────┘
```

### Why Neo4j as Primary (not PostgreSQL)

| Query Type | Neo4j Cypher | PostgreSQL SQL |
|-----------|-------------|----------------|
| Tier filter | `MATCH (c)-[:IN_TIER]->(t {name:'Tier 1'})` | `WHERE tier ILIKE '%Tier 1%'` |
| OEM relationship | `MATCH (c)-[:SUPPLIES_TO]->(oem {name:'Rivian'})` | ❌ No JOIN table |
| Multi-hop | `MATCH (t2)-[:SUPPLIES_TO]->(t1)-[:SUPPLIES_TO]->(oem)` | ❌ Impossible |
| County filter | `MATCH (c)-[:LOCATED_IN]->(l {county:'Gwinnett'})` | `WHERE county ILIKE '%Gwinnett%'` |
| Product search | `MATCH (c)-[:MANUFACTURES]->(p) WHERE p.name CONTAINS 'copper'` | `WHERE products ILIKE '%copper%'` |
| Aggregate (SUM) | Possible but messy in Cypher | ✅ `GROUP BY county SUM(employment)` |
| Industry filter | `MATCH (c)-[:IN_INDUSTRY]->(i {name:'Chemicals'})` | `WHERE industry ILIKE '%Chemicals%'` |

**Neo4j wins on 6/7 types. PostgreSQL wins on 1 (aggregates).**

---

## New File Structure

```
phase4_agent/
├── pipeline.py              ← Orchestrator (simplified)
├── text_to_cypher.py        ← NEW: LLM generates Cypher from question
├── cypher_retriever.py      ← Enhanced: execute + self-heal
├── sql_retriever.py         ← Kept for: aggregates only
└── entity_extractor.py      ← Simplified: only extracts aggregation flag
```

---

## text_to_cypher.py — The Core New File

```python
"""
Text-to-Cypher: uses the local LLM (qwen2.5:14b) to convert any natural
language question into a valid Cypher query against our Neo4j schema.

WHY THIS IS GENERIC:
  - No hardcoded questions/answers
  - Works for any new question about company relationships, tiers, 
    counties, OEMs, products, facility types, or employment
  - Self-heals on syntax errors (retries with error message)
  - Falls back to SQL for aggregate (GROUP BY) queries
"""

NEO4J_SCHEMA = """
NODES:
  (:Company)
    Properties: name, tier, ev_supply_chain_role, ev_battery_relevant,
                industry_group, facility_type, location_city, location_county,
                employment, products_services, primary_oems, classification

  (:OEM)         — e.g. "Hyundai", "Rivian", "Kia", "Toyota", "Honda"
  (:Tier)        — e.g. "Tier 1", "Tier 1/2", "Tier 2/3", "OEM", "OEM Supply Chain"
  (:Location)    — Properties: city, county, state
  (:IndustryGroup) — e.g. "Primary Metal Industries", "Chemicals and Allied Products"
  (:Product)     — e.g. "ED copper foil for electric vehicles"

RELATIONSHIPS:
  (Company)-[:SUPPLIES_TO]->(OEM)
  (Company)-[:LOCATED_IN]->(Location)
  (Company)-[:IN_TIER]->(Tier)
  (Company)-[:IN_INDUSTRY]->(IndustryGroup)
  (Company)-[:MANUFACTURES]->(Product)
  (Company)-[:PEER_OF]->(Company)   -- same tier, same OEM

VALID TIER VALUES: "OEM", "Tier 1", "Tier 1/2", "Tier 2/3", 
                   "OEM Supply Chain", "OEM Footprint"
VALID EV_BATTERY_RELEVANT VALUES: "Yes", "No", "Indirect"
VALID EV_SUPPLY_CHAIN_ROLE: "Battery Cell", "Battery Pack", "Thermal Management",
                              "Power Electronics", "Charging Infrastructure",
                              "Vehicle Assembly", "General Automotive", "Materials"
"""

FEW_SHOT_EXAMPLES = """
Q: Which companies supply Rivian?
A: MATCH (c:Company)-[:SUPPLIES_TO]->(o:OEM)
   WHERE toLower(o.name) CONTAINS 'rivian'
   RETURN c.name AS company_name, c.tier, c.ev_supply_chain_role,
          c.employment, c.location_county
   ORDER BY c.tier, c.name

Q: Find all Tier 1/2 companies in Troup County
A: MATCH (c:Company)-[:IN_TIER]->(t:Tier)-[:LOCATED_IN]->(l:Location)
   WHERE t.name CONTAINS 'Tier 1' AND l.county CONTAINS 'Troup'
   RETURN c.name, c.tier, c.ev_supply_chain_role, c.employment

Q: Which companies manufacture copper foil?
A: MATCH (c:Company)-[:MANUFACTURES]->(p:Product)
   WHERE toLower(p.name) CONTAINS 'copper'
   RETURN c.name, c.tier, c.location_county, c.employment

Q: Show companies that supply both Rivian and Hyundai
A: MATCH (c:Company)-[:SUPPLIES_TO]->(o1:OEM)
   MATCH (c)-[:SUPPLIES_TO]->(o2:OEM)
   WHERE toLower(o1.name) CONTAINS 'rivian'
   AND toLower(o2.name) CONTAINS 'hyundai'
   RETURN c.name, c.tier, c.location_county

Q: Which roles have only one supplier (single point of failure)?
A: MATCH (c:Company)
   WHERE c.ev_supply_chain_role IS NOT NULL
   WITH c.ev_supply_chain_role AS role, collect(c.name) AS companies
   WHERE size(companies) = 1
   RETURN role, companies[0] AS company
   ORDER BY role

Q: Find R&D facilities in Georgia
A: MATCH (c:Company)
   WHERE c.facility_type CONTAINS 'R&D' OR c.facility_type CONTAINS 'Research'
   RETURN c.name, c.tier, c.location_county, c.employment, c.facility_type
"""

CYPHER_GENERATION_PROMPT = """You are a Neo4j Cypher expert for the Georgia EV Supply Chain Intelligence System.

GRAPH SCHEMA:
{schema}

EXAMPLE QUESTION → CYPHER PAIRS:
{examples}

RULES:
1. Return ONLY the Cypher query — no explanation, no markdown
2. Always RETURN: company name, tier, role, employment, county at minimum
3. Use toLower() for case-insensitive string matching
4. Use CONTAINS for partial string matching, not =
5. Add ORDER BY c.employment DESC or c.name
6. Add LIMIT 50 to prevent excessive results
7. For employment filters: c.employment > $min_emp or c.employment < $max_emp

QUESTION: {question}

CYPHER QUERY:"""
```

---

## pipeline.py V2 — Simplified Orchestrator

```python
def ask(self, question: str) -> dict:
    # Step 1: Is this an aggregate (GROUP BY) question?
    if is_aggregate_question(question):
        # PostgreSQL is better for SUM/GROUP BY
        data = aggregate_employment_by_county(tier=extract_tier(question))
        context = format_aggregate_table(data)
    else:
        # Step 2: Generate Cypher from question
        cypher = generate_cypher(question)          # LLM call #1 (~3-5s)
        
        # Step 3: Execute Cypher with self-heal
        results = execute_cypher_safe(cypher, question)  # auto-retries on error
        
        # Step 4: If Neo4j returns 0, fallback to full-text PostgreSQL
        if not results:
            results = full_text_search(question.split())
        
        context = format_compact_table(results)
    
    # Step 5: Synthesize answer
    answer = synthesize_answer(question, context)   # LLM call #2 (~10-30s)
    
    return {"question": question, "answer": answer, ...}
```

---

## What This Solves (Mapping to Failed Questions)

| Failed Q | Root Cause | V2 Fix |
|----------|-----------|--------|
| Q10: OEM + Tier 1 suppliers | SQL can't JOIN | Cypher multi-hop: `MATCH (oem)<-[:SUPPLIES_TO]-(t1:Company)-[:IN_TIER]->(tier)` |
| Q23: Powder coating | Bigram mismatch | Cypher: `MATCH (c)-[:MANUFACTURES]->(p) WHERE p.name CONTAINS 'powder'` |
| Q40: Thermal-related | Stop words blocked | Cypher: `WHERE c.ev_supply_chain_role CONTAINS 'thermal'` |
| Q43: Battery recycling | company_name unsearched | Cypher: `WHERE c.name CONTAINS 'recycling'` |
| Q44: R&D facilities | Stop words, facility_type not searched | Cypher: `WHERE c.facility_type CONTAINS 'R&D'` |
| Q45: Multi-OEM suppliers | SQL can't multi-join | Cypher: `MATCH (c)-[:SUPPLIES_TO]->(o1) MATCH (c)-[:SUPPLIES_TO]->(o2)` |
| Q46: County gap analysis | Not possible in SQL | Cypher: NOT EXISTS pattern |
| Q50: R&D areas | 'R&D' not extracted | Cypher generates the right WHERE clause |

---

## Self-Healing Cypher Execution

```python
def execute_cypher_safe(cypher: str, question: str) -> list[dict]:
    try:
        return execute_cypher(cypher)
    except CypherSyntaxError as e:
        # Feed error back to LLM for self-correction
        fixed_cypher = generate_cypher(
            question,
            error_feedback=f"Previous attempt failed: {e}\nFix the query."
        )
        try:
            return execute_cypher(fixed_cypher)
        except:
            return []  # Fall through to PostgreSQL fallback
```

---

## Aggregate Detection (Simple, No LLM)

```python
_AGGREGATE_TRIGGERS = {
    "how many", "total employment", "highest total", "which county",
    "sum", "count", "aggregate", "combined employment"
}

def is_aggregate_question(question: str) -> bool:
    q = question.lower()
    return any(t in q for t in _AGGREGATE_TRIGGERS)
```

---

## Implementation Order

1. **Create `phase4_agent/text_to_cypher.py`** — schema + prompt + generate + self-heal
2. **Update `phase4_agent/pipeline.py`** — new orchestrator (much simpler)
3. **Update `phase3_graph/graph_loader.py`** — sync `facility_type` to Neo4j nodes
4. **Keep `phase4_agent/sql_retriever.py`** — `aggregate_employment_by_county` only
5. **Remove** — entity_extractor.py complexity (keep only aggregate flag detection)

---

## Expected Impact

| Metric | V1 | V2 |
|--------|----|----|
| Questions answered correctly | ~22/50 | ~38-42/50 (estimated) |
| Works for new questions | Partially | Yes — LLM generates the Cypher |
| Stop word bugs | Yes | Gone — LLM handles it |
| "No data" false negatives | ~12 cases | ~2-3 cases (genuine data gaps) |
| Architecture complexity | High | Low (2 LLM calls, 1 DB call) |
| Multi-hop relationship support | None | Full |

> [!IMPORTANT]
> The key requirement before implementing: ensure Neo4j Company nodes have `facility_type` property synced.
> Run `sync_neo4j_from_postgres.py` to push `facility_type` from PostgreSQL → Neo4j.

> [!NOTE]  
> The graphrag-finetune repo shows the advanced path: fine-tune a small model on
> (question → cypher) pairs from your own schema. This would make the system even faster
> (no 3-5s Cypher generation call). For now, qwen2.5:14b is capable enough.
