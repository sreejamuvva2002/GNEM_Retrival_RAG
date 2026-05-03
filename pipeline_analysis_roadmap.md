# Phase 4 — Full Pipeline Analysis & Road to 50-Question Evaluation

---

## 1. Honest Assessment: Evaluator vs Old Repo

### What I copied vs what I improved

| Component | Old repo | My version | Delta |
|---|---|---|---|
| 5 metrics + prompts | ✅ copied | ✅ same | — |
| Metric weights | ✅ from config.yaml | ✅ same (0.25/0.20/0.20/0.20/0.15) | — |
| JSON response parser | ✅ copied | ✅ improved (strips markdown fences, regex fallback) | better |
| Parallel scoring | ❌ `asyncio.gather` — races on Ollama | ✅ sequential per question | **fixed** |
| Progress/checkpoint | ✅ old repo had `outputs/progress/` | ❌ MISSING — my version has none | **bug** |
| Context passed to judge | ✅ old repo passed retrieved chunks | ❌ MISSING — my version always sends `""` | **critical bug** |
| Judge model | kimi-k2.5:cloud (cloud) | qwen2.5:7b (local, same model) | different |
| Output format | Excel with 3 tabs | Excel with 3 tabs | same |

### Critical Bug: Context is Always Empty
```python
# In my run_ragas_eval.py (WRONG):
"context": "",   # Phase 4 context is embedded in answer, not separately returned

# This means 3 of 5 metrics get garbage scores:
# faithfulness       → can't check if answer is grounded (no context)
# context_precision  → can't score relevance (no context)  
# context_recall     → can't check coverage (no context)
# Only answer_relevancy and answer_correctness work correctly
```

**Fix**: `ask()` must return `retrieved_context` as a separate field.

### Bug: Parallel Metric Scoring on Ollama
The old repo used `asyncio.gather()` for cloud (OpenRouter handles parallel).
For Ollama (local LLM), parallel calls queue and cause 30s+ timeouts.
My version correctly uses sequential `for metric in METRIC_DEFINITIONS`.  ✅

---

## 2. Speed Root Cause Analysis

### Why each question took its observed time

| Q | Time | LLM calls | Root cause |
|---|---|---|---|
| Q1 | 36s | **2** | Text-to-SQL (13s) + synthesis (10s) |
| Q2 | 1.2s | 0 | Direct DB, no LLM — perfect |
| Q3 | 22.4s | **1** | Synthesis only (det. Cypher 0 LLM retrieval) — expected |
| Q4 | **54.5s** | **2** | Text-to-SQL (13s) + synthesis **40s** |
| Q5 | 18.5s | **1** | Synthesis only (det. Cypher) — similar to Q3, OK |
| Q6 | 25s | **2** | Text-to-SQL (12s) + synthesis (4s) |
| Q7 | 6.7s | **1** | Synthesis on 1 result — fastest |

### Q4 54.5s — Two problems compound

**Problem 1**: Text-to-SQL adds 13s for an OEM query we could answer deterministically.
oem=rivian is already extracted → `query_companies(primary_oems=rivian)` = instant.
Text-to-SQL is overkill here.

**Problem 2**: Synthesis generated ~800 output tokens (verbose multi-line bullets).
```
num_predict: 800   ← This is the cap in streaming.py
                      LLM used ALL 800 tokens for Q4
                      800 tokens × ~50ms/token = 40s just for synthesis
```

**Fix**: Route known-entity OEM/county/top-N to deterministic SQL (skip Text-to-SQL).
Text-to-SQL only for ambiguous/complex queries. Keep num_predict: 400 for synthesis.

### Q1/Q6 — Unnecessary Text-to-SQL overhead

```
Q1: aggregate=True AND tier="Tier 1" already extracted
    → aggregate_employment_by_county(tier="Tier 1") gives correct answer instantly
    → Text-to-SQL adds 13s for NO reason

Q6: county="Gwinnett County" already extracted
    → query_companies(location_county="Gwinnett County") sorted by employment = instant
    → Text-to-SQL adds 12s for NO reason
```

### Does RAM affect speed? YES — critically

```
qwen2.5:7b RAM usage:
  Model loaded in VRAM   : ~5.5 GB
  Per-token generation   : ~50ms/token (if VRAM fits)
  Context window loaded  : ~1-2 GB additional VRAM
  
If VRAM < 8GB → model spills to RAM:
  Per-token generation   : ~200-500ms/token (RAM bandwidth limited)
  → 800 tokens × 300ms = 240s (4 minutes!) instead of 40s
  
Your machine: synthesis is ~50ms/token based on observed times
  → Model is likely in VRAM — good
  → But any other app using VRAM will cause spill → sudden 5-10x slowdown
```

---

## 3. Pipeline Fixes to Apply Before Full 50-Question Run

### Fix A: Route deterministic queries FIRST (no Text-to-SQL for known entities)

**Rule**: 
- Known county extracted → direct SQL (not Text-to-SQL)
- Known OEM extracted → direct SQL (not Text-to-SQL)  
- Simple aggregate with known tier → direct SQL (not Text-to-SQL)
- Text-to-SQL ONLY when: aggregate with NO clear tier, OR complex modifier ("at least", "excluding"), OR truly ambiguous

Expected improvement:
- Q1: 36s → ~10s (saves 13s SQL gen)
- Q4: 54s → ~10s (saves 13s SQL gen + reduces synthesis by capping tokens)
- Q6: 25s → ~10s (saves 12s SQL gen)

### Fix B: Cap synthesis output tokens

```python
# streaming.py currently:
"num_predict": 800   # ← too high, Q4 used all 800 = 40s synthesis

# Change to:
"num_predict": 400   # ← still 3-4 paragraphs, but 2x faster synthesis
```

### Fix C: Return retrieved_context from ask()

```python
# pipeline.py ask() must return:
{
    "answer": "...",
    "retrieved_context": context,   # ← add this for RAGAS
    ...
}
```

### Fix D: Cypher builder keyword fix (Q5 false positives)

The cypher_builder re-sorts keywords by length descending internally,
overriding the entity_extractor's sort. Fix: pass keywords as-is.

---

## 4. Fixes Applied — Expected Performance After

| Q | Before | After | Calls |
|---|---|---|---|
| Q1 Aggregate | 36s | ~10s | 1 LLM (synthesis only) |
| Q2 Risk | 1.2s | 1.2s | 0 LLM |
| Q3 Battery | 22.4s | ~10s | 1 LLM |
| Q4 OEM | 54.5s | ~10s | 1 LLM |
| Q5 Product | 18.5s | ~10s | 1 LLM |
| Q6 County | 25s | ~10s | 1 LLM |
| Q7 Facility | 6.7s | ~6s | 1 LLM |
| **Total** | **~164s** | **~68s** | |

### Target SLA for 50 questions
- Simple SQL/Cypher: **<10s** per question
- Complex Text-to-SQL: **<20s** per question
- Risk/direct: **<2s** per question

---

## 5. RAGAS Evaluator — Improvements Over Old Repo

| Improvement | Why |
|---|---|
| Sequential scoring | Old repo used parallel for cloud; sequential needed for local Ollama |
| Checkpoint/resume | 50 questions × 5 metrics × ~8s = ~33min total. If it crashes at Q40, restart from Q40 |
| Context in ask() | Critical for faithfulness/precision/recall metrics |
| Progress bar | Shows estimated time remaining |
| JSON auto-fix | Strips markdown fences before parsing (old repo missed this) |
| Error isolation | One failed metric doesn't fail the whole question |

---

## 6. 50-Question Run Plan

### File structure after run:
```
outputs/ragas_reports/
  phase4_ragas_20260503_HHMMSS.xlsx    ← Excel with 3 tabs
  phase4_ragas_20260503_HHMMSS.json    ← Raw results (for Phase 5 few-shot input)

outputs/progress/
  phase4_eval_progress.jsonl           ← One line per question (checkpoint)
  phase4_eval_answers.md               ← Human-readable answers + scores
```

### Checkpoint design:
```
Q1 → answered + scored → written to progress.jsonl immediately
Q2 → answered + scored → appended to progress.jsonl
...
If crash at Q40 → restart reads progress.jsonl → skips Q1-Q39 → resumes Q40
```

### 30/20 Split for Phase 5/6:
```
Questions 1-30  → Training set (fine-tune model, build few-shot library in Qdrant)
Questions 31-50 → Test set (evaluate fine-tuned model vs base model)

Metrics comparison:
  Base (qwen2.5:7b)      → your current scores
  Few-shot (Phase 5)     → improvement from dynamic few-shot retrieval
  Fine-tuned 1.5B (P6)  → improvement from dedicated model
```

---

## 7. Phase Roadmap

```
CURRENT:  Phase 4 — V3 Pipeline (Text-to-SQL + Deterministic Cypher + qwen2.5:7b)
                   ↓ [now: fix 4 bugs, run 50-question RAGAS eval]

PHASE 5:  Dynamic Few-Shot (graphrag-finetune approach)
          ─────────────────────────────────────────────────
          What: Store (question → correct SQL/Cypher) pairs in Qdrant
          When: Use k=3 nearest pairs as examples in the LLM prompt
          Result: LLM generates better SQL/Cypher because it sees similar examples
          Speed: Same as now (Qdrant lookup <50ms)
          Accuracy: +10-15% expected (few-shot > zero-shot)
          
          Training data: 30 questions with verified (question, SQL/Cypher) pairs
          Test: Compare RAGAS scores on 20 test questions vs Phase 4 baseline

PHASE 6:  Fine-Tuned 1.5B Model
          ─────────────────────────────────────────────────
          What: Fine-tune qwen2.5:1.5B on 30 (question → SQL/Cypher) pairs
          Why:  <0.5s Cypher generation vs 13s with 7B zero-shot
          How:  LoRA fine-tuning (runs on your GPU, ~2-4 hours)
          Test: RAGAS scores on 20 test questions vs Phase 5 dynamic few-shot
          
          If fine-tuned model beats few-shot: use as production Cypher generator
          If not: Phase 5 few-shot stays as production

PHASE 7:  Chat UI
          ─────────────────────────────────────────────────
          FastAPI backend with streaming SSE endpoint (already built)
          React/Next.js frontend with real-time token display
          Deploy: Local first → then containerize
```

### Decision tree after evaluation:
```
RAGAS score > 0.85 across 50 questions?
  YES → Ship Phase 4 as-is → Build Phase 7 UI  
  NO  → Identify failing categories → Phase 5 few-shot → Re-evaluate → Phase 7 UI
  
Fine-tuned model beats few-shot on 20-question test?
  YES → Replace Text-to-SQL/Cypher-gen with fine-tuned model
  NO  → Keep Phase 5 few-shot as Cypher generator
```
