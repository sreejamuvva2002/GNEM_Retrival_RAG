# Architecture Analysis — Hitting the 5s SLA

## Current Problem Statement

| Model | Role | Size | CPU Speed |
|---|---|---|---|
| `qwen2.5:14b` | Synthesis | ~8.5 GB | 3-8 tok/s → **30-80s** |
| `gemma3:4b` | Cypher gen | ~2.5 GB | 10-15 tok/s → **15-30s** |

**With CPU-only inference, 5s is physically impossible for a 14B model.**
A 200-token answer at 5 tok/s = 40s. Physics, not a bug.

---

## Why Deterministic Rules Exist (and Why It's Not Wrong)

The `cypher_builder.py` uses deterministic rules NOT because we hardcoded questions,
but because **for a fixed schema with known entities, the query structure is always the same**.

```
"Show Rivian suppliers" → always: MATCH (c)-[:SUPPLIES_TO]->(o:OEM {name: 'Rivian'})
"Battery Cell companies" → always: MATCH (c) WHERE c.ev_supply_chain_role CONTAINS 'Battery Cell'
```

The entity values come from the DB. The Cypher structure follows the schema.
This is correct. The issue is SYNTHESIS speed, not Cypher generation.

---

## 5 Real Architectures for <5s Response

### Architecture 1 — Semantic Cache (Best First Win)
**Concept**: Before touching any LLM, check if a semantically similar question was answered before.

```
User: "Which county has most Tier 1 workers?"
        ↓
  Embed question → vector
  Search Qdrant (you already have it!) for similar past queries
  Similarity > 0.92? → Return cached answer INSTANTLY (0ms)
  Miss? → Run full pipeline → cache the result
```

**Why this works for you:**
- You already have Qdrant set up with `nomic-embed-text`
- EV supply chain questions are repetitive in practice
- "Top Tier 1 county?" ≈ "Highest employment Tier 1?" ≈ "Leading county for Tier 1?"
- First user pays 30s. Every similar user after pays 0ms.

**Implementation**: ~50 lines of Python. No new infrastructure.
**Speed**: 0ms for cache hits (likely 60-80% of real chatbot traffic)

---

### Architecture 2 — Streaming (Immediate UX Fix)
**Concept**: Don't wait 30s for the full answer. Send tokens as they generate.

```
Current:  [user waits 30s] ........... [answer appears all at once]
Streaming: [answer starts in <1s]  T.r.o.u.p. C.o.u.n.t.y. h.a.s...
```

The total time is SAME but the perceived experience is completely different.
ChatGPT uses this. Every production LLM chatbot uses this.

**Ollama already supports streaming** (`"stream": true` in payload).
**FastAPI supports SSE** (Server-Sent Events) for real-time token push to browser.

**Implementation**: Change `"stream": False` → `True`, add SSE endpoint.
**Speed**: First token in <2s. User sees progress immediately.

---

### Architecture 3 — Switch to qwen2.5:7b (Already Installed)
**You already have it**: Line 52 in `.env` comments says `qwen2.5:7b is installed`.

```
qwen2.5:14b → 30-80s synthesis
qwen2.5:7b  → 15-35s synthesis  (2x faster, similar quality)
qwen2.5:3b  → 8-15s synthesis   (3-4x faster, slightly lower quality)
```

Change ONE line in `.env`:
```
OLLAMA_LLM_MODEL=qwen2.5:7b
```

**For structured data Q&A (not creative writing), 7b performs ~90% as well as 14b.**

---

### Architecture 4 — Template-Based Synthesis (Skip LLM for Structured Queries)
**Concept**: When data is already structured, the LLM only adds formatting overhead.

```python
# Instead of: LLM reads 20 rows → thinks → writes "Troup County has..."
# Direct formatter:

def answer_aggregate(data, tier):
    top = data[0]
    return (
        f"{top['county']} has the highest total employment"
        f" among {tier} suppliers with {top['total_employment']:,} employees"
        f" across {top['company_count']} companies."
    )
```

**Applicable to**: Aggregate (Q1), Risk (Q2-already done), County (Q6), Top-N queries.
These are ~40% of all questions. Skip LLM = instant answer.

**For complex questions (OEM network, product search)**: Still use LLM.

---

### Architecture 5 — Dynamic Few-Shot + Smaller Cypher Model
**Based on the graphrag-finetune repo you shared:**

Instead of full fine-tuning (expensive), use **dynamic few-shot retrieval**:

```
User asks: "Show Tier 2 suppliers in Gwinnett County for Rivian"
  ↓
Embed question → search Qdrant for most similar (question, cypher) pairs
  ↓
Inject top-3 similar examples into prompt
  ↓
Even qwen2.5:3b generates correct Cypher reliably with good examples
  ↓
Cypher executes in <1s → LLM or template synthesizes answer
```

**This is exactly what the graphrag-finetune repo does** — it builds a corpus of
(question → cypher) pairs, then retrieves similar ones as few-shot examples.

**Implementation**: Uses your existing Qdrant. Need to build the (q, cypher) dataset.

---

## Recommended Immediate Action Plan

### Phase A — This Week (No new tools, max impact)

| Step | Change | Expected gain |
|---|---|---|
| 1 | `OLLAMA_LLM_MODEL=qwen2.5:7b` | Q1: 33s→15s, Q3: 81s→35s |
| 2 | Enable Ollama streaming (stream=True) | Perceived latency: 30s→<2s to first token |
| 3 | Template answers for aggregate/county/risk | Q1,Q2,Q6: 33s/1s/15s → all <1s |

**After Phase A: 4/7 questions answer in <1s. Complex questions stream in <2s to first token.**

### Phase B — Next Sprint (Semantic Cache)

| Step | Change | Expected gain |
|---|---|---|
| 4 | Add semantic cache layer (Qdrant + nomic-embed) | 60-80% of repeated questions: 0ms |
| 5 | Build (question, cypher) pair dataset from smoke tests | Feed into dynamic few-shot |

### Phase C — Future (Fine-tuning path from your repo)

| Step | Change | Expected gain |
|---|---|---|
| 6 | Fine-tune qwen2.5:1.5b on (q→cypher) pairs | Cypher in <0.5s |
| 7 | Combine: tiny Cypher model + template synthesis | Full pipeline <3s |

---

## Why NOT Use 2 LLMs in Production

```
Current: qwen2.5:14b (8.5GB) + gemma3:4b (2.5GB) = 11GB RAM for models alone
         → Memory pressure → 500 errors → timeouts

Better: ONE model (qwen2.5:7b, 4.5GB) for everything
         → No contention
         → Deterministic Cypher builder handles Cypher (0ms, no model)
         → LLM only called once for synthesis

Best (future): ONE tiny fine-tuned model (qwen2.5:1.5b, 1GB) for Cypher
               + ONE fast 7b for synthesis
               = 5.5GB total, no contention, Cypher in <0.5s
```

---

## The Honest SLA Math

```
Target: 5s end-to-end

Breakdown:
  Entity extraction:      0.01s  (deterministic, in-memory)
  DB/Neo4j query:         0.3s   (network + query)
  LLM synthesis (7b):    15-30s  ← THE BOTTLENECK

To hit 5s with local LLM you need ONE of:
  a) GPU (even consumer RTX 3080) → 7b runs at 50-80 tok/s → 3-5s ✅
  b) Template synthesis (no LLM)  → <1s for structured queries ✅
  c) Semantic cache hit            → 0ms for repeated queries ✅
  d) Streaming                    → 5s SLA becomes "first token in <2s" ✅

Realistically, with CPU only and local LLM:
  - Cache miss + complex question = 15-30s with 7b
  - Cache hit = 0ms
  - Aggregate/County/Risk = <1s (template)
  - True 5s SLA requires GPU or API
```

> [!IMPORTANT]
> The graphrag-finetune repo's approach works perfectly when combined with templates:
> Fine-tuned 1.5B model generates Cypher in <0.5s → Template formats answer in 0ms → Total: <1s.
> This is the correct long-term architecture for your use case.

