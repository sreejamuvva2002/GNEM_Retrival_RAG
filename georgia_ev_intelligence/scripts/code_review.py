"""
Full Code Review — Phase 4 Agent Pipeline
Reviewer mindset: What will break on question 23 of 50 that hasn't shown up in 5?
"""

FINDINGS = """
══════════════════════════════════════════════════════════════════
BUG 1 — entity_extractor.py:288-289
  known_extracted only adds e.ev_role, NOT e.ev_role_list
  When multiple roles match (e.g. 'Battery Cell OR Battery Pack'),
  ev_role is None and ev_role_list = ['Battery Cell','Battery Pack'].
  So 'battery','cell','pack','roles' all stay in keyword candidates.
  Q1 result: keywords=['classified under','under battery','battery cell','battery pack','pack roles']
  Fix: iterate ev_role_list too when building known_extracted.
══════════════════════════════════════════════════════════════════
BUG 2 — entity_extractor.py: NO company name extraction
  The 50 questions WILL include: "What does Hanwha Q CELLS do?"
  or "What tier is SungEel Recycling Park Georgia?"
  No company_name field in Entities. No SQL filter for company_name.
  Pipeline falls through to: sql_filters={} → SQL skipped → keyword
  search for "hanwha cells" → maybe finds it, maybe not.
  Fix: Add company_name extraction using real company names from DB.
  Add sql_filters["company_name"] = company in sql_retriever.

══════════════════════════════════════════════════════════════════
BUG 3 — pipeline.py: fallback dumps 80 companies
  if not sections: query_companies()[:80] → 80 companies to LLM.
  For any question where extractor finds nothing AND keyword search
  finds nothing, LLM gets 80 random companies and will fail.
  Fix: Return "Insufficient data" message. No 80-company dump.

══════════════════════════════════════════════════════════════════
BUG 4 — sql_retriever.py:83 — Tier filter is EXACT MATCH
  Company.tier == tier  ← exact match only
  "Tier 1" question → exact 'Tier 1' companies (71) ✓ for Q2
  BUT: "List all Tier 2 companies" → tier='Tier 2' → 0 results
  because DB has 'Tier 2/3' not 'Tier 2'.
  Questions about "Tier 2" will return nothing from SQL.
  Fix: Use ilike for tier filter too.

══════════════════════════════════════════════════════════════════
BUG 5 — entity_extractor.py: Tier extraction only gets exact values
  Regex looks for Tier 1, Tier 2, Tier 1/2, Tier 2/3 etc.
  But question "show me tier two suppliers" (written out) → no match.
  Question "Tier 1 and Tier 2 companies" → only captures first match.
  This is a scope limitation, not catastrophic for the 50 Qs.

══════════════════════════════════════════════════════════════════
BUG 6 — pipeline.py: keyword search ALWAYS runs even for pure OEM/County Qs
  Q4 (Rivian): keywords=['network linked','broken down','supply chain','chain role']
  Keyword search runs for 'network linked' → searches products_services
  → finds 1 noisy company (that's the +1 seen in retrieved=7 not 6).
  Grammatical bigrams should never reach product_services search.
  Fix: Only run keyword search if keywords look like product terms
  (contain nouns, not verbs/prepositions). OR: only run keywords
  when sql_filters was empty (pure product-search questions like Q3).

══════════════════════════════════════════════════════════════════
BUG 7 — pipeline.py: Qdrant (vector search) never called
  Phase 1 scraped web documents into Qdrant. Pipeline never queries it.
  Questions about policy, market reports, or companies not in GNEM
  will fail entirely. The user specified: "if not in DB, use web/Qdrant".
  This is a missing feature, not a bug per se, but affects accuracy.

══════════════════════════════════════════════════════════════════
PRIORITY ORDER TO FIX BEFORE 50 QUESTIONS:
  P1: BUG 2 — company name extraction (affects ~10-15 of 50 questions)
  P2: BUG 1 — ev_role_list not in known_extracted (keyword noise)
  P3: BUG 4 — Tier filter exact match (affects "Tier 2" questions)
  P4: BUG 6 — keyword search runs on grammatical bigrams (noise)
  P5: BUG 3 — fallback dumps 80 companies (correctness risk)
"""
print(FINDINGS)
