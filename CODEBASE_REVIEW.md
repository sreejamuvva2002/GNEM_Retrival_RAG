# CODEBASE REVIEW REPORT

Codex final codebase review for the Georgia EV Supply Chain RAG evaluation project.

Review date: 2026-05-20  
Repository: `/home/sm11926/GNEM_Retrival_RAG`

## 1. Executive Verdict

Verdict: **INVALID COMPARISON — RESULTS MAY BE MISLEADING**

- Is the codebase runnable? **Partially.** The core data, chunking, indexing, retrieval, and JSONL generation paths are runnable in this environment. I verified the current artifacts and database contain 205 parent records, 1025 child chunks, 0 orphan child chunks, and 768-dimensional vectors. However, the targeted test suite has one failing test, RAGAS is not installed in the project `.venv`, and the actual RAGAS evaluator is outside this repository.
- Is the retrieval pipeline logically correct? **Mostly for basic RAG, not fully for research-grade exhaustive QA.** BM25 and dense retrieval query the same `child_chunks` table, child-parent mapping is correct, and parent-level reranking is actually used. The main research risks are no score-aware hybrid fusion, no structured filtering, top-45 context truncation risk, no retrieval recall validation, and weak support for exhaustive list/count questions.
- Is the evaluation pipeline logically correct? **No, not as currently wired.** `build_ragas_report.py` creates an Excel workbook, but the evaluator is external (`/home/sm11926/Downloads/Comparision_Report/evaluate_ragas_ollama.py`). That evaluator groups contexts only by question, not by model and pipeline. This can mix `direct_kb`, `rag_only`, and `hybrid_rag` contexts and apply the same context set to every answer for a question.
- Are the 4 compared pipelines truly comparable? **Not cleanly.** `pretrained_only` correctly receives no context. `rag_only` receives retrieved context. `direct_kb` receives all 205 normalized KB rows in the generation prompt. But `hybrid_rag` is mislabeled: its prompt also says "Use ONLY the retrieved context" and does not actually allow external pretrained knowledge. Also, direct-KB and top-45 RAG prompts may exceed model context windows without detection.
- Are there critical bugs that invalidate results? **Yes.** The biggest issue is evaluation-context mismatch: the external RAGAS evaluator does not preserve pipeline-specific contexts. It also truncates evaluation contexts to 8 chunks by default, while generation may use 45 retrieved parent chunks or 205 direct-KB records. Therefore faithfulness/groundedness metrics may not evaluate the evidence the model actually saw.
- Is this ready for research reporting? **No.** The code is useful for smoke testing and internal baseline generation, but current outputs should not be presented as final research results until the evaluation pipeline, context tracking, prompt definitions, and tests are fixed.

## 2. What Is Correct

- **Data loading:** `georgia_ev_intelligence/shared/data/loader.py` loads `kb/GNEM - Auto Landscape Lat Long Updated.xlsx`, normalizes column names, fills missing values, and writes `Normalized_kb.xlsx`. Verified artifact shape: 205 rows x 16 columns.
- **Question loading:** `run_hybrid_rag._load_questions()` supports the current human-validated workbook columns: `Question`, `Human validated answers`, and `Num`.
- **Parent chunking:** `offline_pipeline/chunking/parent_chunk.py` creates one parent record per normalized KB row and includes all major KB fields in `parent_chunk_text`.
- **Child chunking:** `offline_pipeline/chunking/child_chunk.py` creates five child chunk types per parent: `identity`, `product_role`, `oem_relationship`, `location_employment`, and `classification`.
- **Parent-child validation:** `validate_relationships()` checks expected child count, duplicate chunk IDs, orphan children, and exactly 5 children per parent.
- **Database integrity in current environment:** Verified with PostgreSQL queries:
  - `parent_chunks`: 205 rows.
  - `child_chunks`: 1025 rows.
  - distinct child parent IDs: 205.
  - orphan children: 0.
  - vector dimensions: 768 for all 1025 child chunks.
  - each child type count: 205.
- **pgvector storage:** `offline_pipeline/pgvector_store.py` creates the `vector` extension, stores child vectors, uses `chunk_id` as primary key, and upserts child rows.
- **Parent PostgreSQL storage:** `offline_pipeline/postgres_store.py` uses `record_id` as primary key and upserts parent rows.
- **BM25 retrieval:** `runtime_pipeline/retrieval/bm25_retriever.py` builds an in-memory BM25 index from the `child_chunks` table.
- **Dense retrieval:** `runtime_pipeline/retrieval/dense_pgvector_retriever.py` embeds the query and searches `child_chunks` by pgvector cosine distance.
- **Hybrid orchestration:** `HybridRetrievalOrchestrator.retrieve_with_sources()` runs sparse and dense retrieval in parallel, merges child chunks, maps them to parents, and reranks parents.
- **Reranking:** `CrossEncoderReranker.rerank_parents()` is called in the active path. Reranking is parent-level, matching the evidence unit passed to the LLM.
- **Retrieval reuse across models:** `run_baseline.RetrievalCache` caches retrieval by question text, so retrieval is run once per question within a baseline run and reused for all models and both retrieval-based pipelines.
- **Model adapter:** `generation/llm_adapter.py` introduces `OllamaAdapter(model_name=...)`, allowing model selection without changing global config.
- **Baseline JSONL generation:** `run_baseline.py` writes one JSONL file per model/pipeline combination. Each record includes `question_id`, `question`, `ground_truth`, `answer`, `contexts`, `pipeline`, `model`, and `trace`.
- **Direct KB generation:** `direct_kb_pipeline.py` uses `georgia_ev_intelligence/outputs/Normalized_kb.xlsx` as requested and formats each row as one context string. Current smoke output records contain 205 direct-KB contexts.
- **Report generation:** `build_ragas_report.py` can convert baseline JSONL files to an Excel workbook with `responses`, `retrieval`, `golden_answers`, and `run_info` sheets.

## 3. Critical Bugs or Mistakes

### 3.1 RAGAS evaluation contexts are not pipeline-specific

- File/function affected: external `/home/sm11926/Downloads/Comparision_Report/evaluate_ragas_ollama.py`, `build_context_map()` lines 171-201.
- Current problem: The evaluator builds `context_map` keyed only by `question`. It ignores `pipeline` and model response column. Every answer for the same question receives the same retrieved context list during RAGAS scoring.
- Interaction with repo code: `build_ragas_report.py` writes a `retrieval` sheet with columns `question`, `pipeline`, `rank`, `chunk_type`, `text`, but the external evaluator discards `pipeline` when constructing contexts.
- Why it matters: `rag_only`, `hybrid_rag`, and `direct_kb` have different evidence conditions during generation, but RAGAS may judge them against a mixed context set. This can invalidate faithfulness and response-groundedness scores.

### 3.2 RAGAS evaluation truncates contexts differently from generation

- File/function affected: external evaluator `parse_args()` lines 92-103 and `build_context_map()` lines 171-201.
- Current problem: The evaluator defaults to `--max-contexts 8` and `--max-context-chars 4000`. Generation uses up to 45 retrieved parent chunks for RAG and 205 normalized KB rows for direct-KB.
- Why it matters: RAGAS does not evaluate the full context used by the LLM. A response can be faithful to generation context but appear unsupported to RAGAS, or vice versa.

### 3.3 RAGAS evaluator does not use the human golden answers for answer accuracy

- File/function affected: external evaluator `generate_reference_answer()` lines 253-267 and main loop lines 514-527.
- Current problem: The evaluator generates a new reference answer from contexts using the judge/reference model. It does not read the `golden_answers` sheet produced by `build_ragas_report.py`.
- Why it matters: The project goal says answers should be scored against human-validated golden answers. Current external evaluation instead scores against judge-generated references, which changes the research question.

### 3.4 RAGAS evaluator is external and not reproducible from this repo

- File/function affected: `README.md` lines 91-92 and 291-299; `build_ragas_report.py` lines 187-191.
- Current problem: The README points to a separate evaluation environment and an external script. The project `.venv` does not have `ragas` installed, and `requirements.txt` does not include `ragas`, `openai`, or RAGAS-specific dependencies.
- Verified: `.venv/bin/python -c "import ragas"` fails with `ModuleNotFoundError`.
- Why it matters: Another researcher cannot reproduce the evaluation from this repository alone.

### 3.5 `hybrid_rag` is mislabeled

- File/function affected: `run_hybrid_rag.py` prompt lines 21-24; `run_baseline.py` lines 249-275.
- Current problem: `hybrid_rag` uses the prompt from `run_hybrid_rag.build_prompt()`, but that prompt says: "Use ONLY the retrieved context below. Do not use outside knowledge."
- Why it matters: This is not "retrieved KB chunks plus LLM reasoning" if "hybrid" means allowing pretrained knowledge. It is a stricter formatting prompt over the same retrieved context.

### 3.6 `direct_kb` may not actually fit in the model context window

- File/function affected: `direct_kb_pipeline.py` lines 63-85; `llm_adapter.py` lines 38-55.
- Current problem: `direct_kb` concatenates all 205 normalized KB rows into a single prompt. `OllamaAdapter.generate()` does not set `num_ctx`, measure token count, record prompt length, or detect truncation.
- Why it matters: The JSONL may say 205 contexts were provided, but the model may not actually attend to all 205 records. This is especially risky for 7B/8B/14B models and for long direct-KB prompts.

### 3.7 RAG prompts may exceed practical context budgets

- File/function affected: `HybridRetrievalConfig` in `config.py`; default `RERANKER_TOP_K = 45`.
- Current problem: RAG-based generation passes up to 45 full parent chunks. No prompt token counting or context-window enforcement exists.
- Why it matters: RAGAS `contexts` may list 45 chunks, but generation may have silently truncated context. This breaks the link between generated answer and recorded evidence.

### 3.8 Hybrid merge loses retrieval scores and rank information

- File/function affected: `merger.py` lines 12-25 and `orchestrator.py` lines 42-47.
- Current problem: The merger concatenates BM25 results and dense results, deduplicating only by `chunk_id`. It discards BM25 scores, dense distances, and original ranks. There is no RRF, normalized score fusion, or score logging.
- Why it matters: The candidate pool is sparse-first and score-blind. Cross-encoder reranking helps after parent expansion, but the initial hybrid candidate formation is not a rigorous hybrid retrieval method.

### 3.9 Unit tests fail

- Command run: `pytest -q tests/runtime_pipeline/test_hybrid_retrieval_orchestrator.py tests/runtime_pipeline/test_run_retrieval_only.py tests/runtime_pipeline/test_run_all_pipelines.py`
- Result: 9 passed, 1 failed.
- Failing test: `tests/runtime_pipeline/test_run_all_pipelines.py::test_workbooks_include_dense_and_sparse_columns`.
- Current problem: The test expects `dense retrieved context` and `sparse retrieved context` columns in per-pipeline workbooks, but `run_all_pipelines.OUTPUT_COLUMNS` does not include those columns.
- Why it matters: The code and tests disagree on output contracts. Even if `run_all_pipelines.py` is legacy relative to `run_baseline.py`, a failing test in an output path is a reproducibility risk.

### 3.10 Existing smoke outputs are not final results

- File affected: `georgia_ev_intelligence/outputs/baselines/20260520_185816/`.
- Current problem: The run contains 2 models x 4 pipelines x 3 questions = 24 records, not 7 models x 4 pipelines x 50 questions = 1400 records.
- Why it matters: These files are smoke-test artifacts only.

### 3.11 Raw company capitalization is lost

- File/function affected: `loader.py`, `clean_company()` lines 123-125.
- Current problem: Company names are lowercased during normalization.
- Why it matters: Prompts ask models to preserve company names exactly, but the normalized KB no longer preserves original capitalization. This can affect answer readability and exact/string-based evaluation.

### 3.12 Data ingestion is not a full DB reset

- File/function affected: `postgres_store.py` lines 43-75; `pgvector_store.py` lines 73-84.
- Current problem: Both tables use upserts. Parent rows are not deleted if the source KB shrinks or record IDs change. Child table is dropped only with `--recreate-child-table`.
- Why it matters: The current DB is clean, but reproducibility depends on rebuild discipline. A future changed KB could leave stale rows unless tables are explicitly reset.

## 4. Pipeline Comparability Review

### `pretrained_only`

- Input evidence: no KB evidence.
- Prompt: `pretrained_only_pipeline.py` asks the model to answer from pretraining only and to admit uncertainty.
- Allowed to use pretrained knowledge: yes.
- Receives retrieved context: no.
- Receives full KB: no.
- Output format: JSONL with `contexts=[]`.
- Evaluation fairness: answer accuracy and answer relevancy can be useful. Faithfulness and groundedness should be marked not applicable. If the external evaluator gives it mixed contexts by question, those metrics become invalid.

### `rag_only`

- Input evidence: retrieved parent chunks after BM25, dense retrieval, merge, parent expansion, and cross-encoder reranking.
- Prompt: `rag_only_pipeline.py` lines 52-69. It explicitly says to use only context and abstain when unsupported.
- Allowed to use pretrained knowledge: no, by prompt instruction only.
- Receives retrieved context: yes.
- Receives full KB: no.
- Output format: JSONL with retrieved parent chunks as `contexts`.
- Evaluation fairness: context metrics are appropriate only if RAGAS receives exactly this pipeline's retrieved context list. Current external evaluator does not ensure that.

### `hybrid_rag`

- Input evidence: same retrieved parent chunks as `rag_only`.
- Prompt: `run_hybrid_rag.py` lines 21-103. It also forbids outside knowledge and uses only retrieved context.
- Allowed to use pretrained knowledge: no, despite the name.
- Receives retrieved context: yes.
- Receives full KB: no.
- Output format: JSONL with retrieved parent chunks as `contexts`.
- Evaluation fairness: comparable to `rag_only` as a different prompt style, not as a different evidence policy.
- Main issue: It should be renamed or redefined. Current README says it "allows LLM reasoning on top", but implementation says use only retrieved context.

### `direct_kb`

- Input evidence: all rows from `Normalized_kb.xlsx`.
- Prompt: `direct_kb_pipeline.py` lines 13-47. It says use only full KB records.
- Allowed to use pretrained knowledge: no, by prompt instruction.
- Receives retrieved context: no.
- Receives full KB: intended yes. Current smoke JSONL has 205 contexts per direct-KB record.
- Output format: JSONL with all normalized KB records as `contexts`.
- Evaluation fairness: answer accuracy and answer relevancy can be compared. Faithfulness/groundedness can be computed if the full KB contexts are passed, but context precision/recall are not retrieval metrics here.
- Main issue: context-window overflow is unmeasured and RAGAS defaults to only 8 contexts, so direct-KB generation/evaluation are not aligned.

## 5. Retrieval Pipeline Review

- **BM25 tokenizer correctness:** `tokenize_bm25()` lowercases, strips possessives, keeps hyphen/slash compounds, and expands them into parts. This helps `Hyundai-Kia` and `Tier 1/2`.
- **Hyphenated terms:** `Hyundai-Kia` becomes `hyundai-kia`, `hyundai`, `kia`. Good.
- **Slash terms:** `Tier 1/2` becomes tokens including `1/2`, `1`, and `2`. Good.
- **Company names:** Basic tokenization is adequate, but there is no alias handling for `Hyundai Kia`, `Hyundai-Kia`, `Hyundai & Kia`, `LGES`, or abbreviated supplier names.
- **Locations:** There is no normalization for `GA` vs `Georgia`, county-only vs city/county wording, or spelling variants. Dense retrieval may help, but BM25 does not.
- **BM25 corpus:** Built from `child_chunks.metadata`, skipping `Unknown` values. This is reasonable but not identical to dense `embedding_text`.
- **Dense corpus:** Built from child `embedding_text` with document prefix. Query uses query prefix. This is consistent with embedding model design.
- **Score normalization:** None. BM25 scores and dense distances are not surfaced or fused.
- **Merge behavior:** Sparse results are in position 0 and dense results in position 1, so duplicate child chunks keep the sparse ordering. This can bias the candidate pool.
- **Duplicate handling:** Child dedupe by `chunk_id`; parent dedupe by `parent_record_id`. Verified current DB has no child orphans.
- **Parent-child mapping:** Correct in current DB: 1025 children map to 205 parent IDs.
- **Reranker input:** Parent reranker receives `parent_chunk_text`, which matches the final LLM evidence.
- **Reranker top-k:** Default is 45. It is used, but not justified with recall or context-size experiments.
- **Exhaustive/list/count questions:** High risk. Broad questions can require many records. Top-45 may omit required rows, and the LLM must count from partial evidence.
- **Metadata filters:** None are used. Questions that map cleanly to columns (`category`, `primary_oems`, `ev_supply_chain_role`, `employment`) still use fuzzy retrieval.
- **Structured question bypass:** Missing. Count/list/highest questions would be more reliable with deterministic DataFrame or SQL filtering before or instead of retrieval.

Likely retrieval failure cases:

- "Show all" questions where relevant rows exceed top-45.
- Count questions requiring exact totals.
- Highest/largest employment questions where lexical/dense retrieval may miss the true maximum.
- OEM alias questions involving `Hyundai`, `Kia`, `Hyundai-Kia`, `Multiple OEMs`, or abbreviated OEM names.
- Category questions involving `Tier 1/2`, `Tier 2/3`, `OEM Footprint`, or multi-value categories.
- Location questions using `GA`, city names, counties, or address fragments inconsistently.

## 6. Data and Knowledge Base Integrity

- All 205 normalized rows are present in `georgia_ev_intelligence/outputs/Normalized_kb.xlsx`.
- `parent_chunks.xlsx` has 205 rows and 205 unique `record_id` values.
- `child_chunks.xlsx` has 1025 rows and 205 unique `parent_record_id` values.
- PostgreSQL currently matches the artifacts: 205 parents, 1025 children, 0 orphan children, 768 vector dimensions.
- Empty rows are dropped only when company identity is missing. This is reasonable for the current workbook.
- Column names are normalized safely for the known schema.
- Missing values are converted to `Unknown`. This is consistent, but `Unknown` can become part of parent prompts and may affect model answers.
- Parent chunks include major fields: company, category, industry group, location, address, lat/long, facility type, supply chain role, OEMs, supplier type, employment, product/service, battery relevance, classification method.
- Child chunks cover relevant retrieval facets: identity, product/role, OEM relationship, location/employment, and classification.
- Raw values are not fully preserved because `clean_company()` lowercases company names.
- Category/OEM/product separators are normalized. This improves consistency but may alter exact original formatting.
- There are 193 unique company names across 205 rows. This is probably valid multi-site/multi-record data, but research reporting must distinguish company-level vs row/site-level counting.
- Ingestion is idempotent by primary key but not self-cleaning if the KB shrinks or record IDs change.

## 7. Evaluation Pipeline Review

- **Generated answer alignment:** `run_baseline.py` reads the human-validated workbook and stores `question_id`, `question`, and `ground_truth` in every JSONL record. This is correct.
- **JSONL fields:** Smoke artifacts contain all required fields: `question_id`, `question`, `ground_truth`, `answer`, `contexts`, `pipeline`, `model`, and `trace`.
- **Canonical storage:** JSONL context lists are good. Excel should not be the source of truth.
- **RAGAS workbook:** `build_ragas_report.py` creates `responses`, `retrieval`, `golden_answers`, and `run_info`.
- **Critical issue:** The external evaluator ignores the `golden_answers` sheet for reference answers and generates references using the judge/reference model.
- **Critical issue:** The external evaluator ignores pipeline when assigning contexts, so metric inputs can be wrong for every pipeline.
- **Critical issue:** The external evaluator uses only up to 8 contexts by default, while generation records 45 or 205 contexts.
- **Failed judge calls:** In the external evaluator, `score_response()` catches exceptions and writes `NaN` silently, without storing error messages. This can hide evaluation failures.
- **Score types:** Scores are converted to floats when successful. Failed metrics become `NaN`.
- **Averages:** Summary uses pandas mean, which ignores `NaN` by default. This can overstate scores if failed rows are silently dropped from averages.
- **Pretrained-only:** Should only receive `answer_accuracy` and `answer_relevancy`. In the current external script, it can still receive context-dependent metrics if requested, because metric applicability is not pipeline-aware.
- **Direct-KB:** During generation, it uses all 205 contexts. During RAGAS external evaluation, only the first 8 mixed question-level contexts may be used by default.

RAGAS defensibility:

- RAGAS can be useful for `rag_only` and `hybrid_rag` only if it receives the exact retrieved contexts used for that pipeline and model answer.
- RAGAS faithfulness/groundedness are not meaningful for `pretrained_only`.
- For `direct_kb`, context precision/recall are not retrieval metrics. Faithfulness can be meaningful, but only if the evaluator can handle the full KB context or uses a documented truncation policy.
- Answer accuracy should use the human-validated golden answers, not generated references, if the research claim is "agreement with human validation."

## 8. Reproducibility Review

- Random seeds: not set. Retrieval is mostly deterministic, but model generation and tie ordering may vary.
- LLM temperature: controlled through `OLLAMA_TEMPERATURE`, default 0.1. This is low but not fully deterministic.
- Top-p: default 0.9. This can introduce variability.
- `num_ctx`: not set for Ollama. This is a major reproducibility gap for long-context prompts.
- Input question order: fixed by the Excel row order.
- KB version: no checksum recorded in `config.json`.
- DB state: not fully recorded in baseline config. The config does not include parent count, child count, vector dimension, embedding model, reranker model, or DB rebuild timestamp.
- Model versions: model names are recorded, but Ollama model IDs/digests are not recorded in `config.json`.
- Git state: no commit hash or dirty-worktree status is recorded.
- Output filenames: timestamped run folders prevent accidental overwrite. Individual JSONL files are overwritten only within their run folder.
- Resume behavior: absent. If a 1400-call run fails halfway, there is no built-in skip-completed/resume mode.
- Failure recovery: failed LLM calls are written as `ERROR: ...`, but there is no retry policy or failure summary.
- Another person could not fully reproduce the experiment from this repo alone because the evaluator is external, `.env` is required, DB state is required, and RAGAS dependencies are not declared.

## 9. Performance and Cost Review

- Retrieval reuse is correctly implemented within `run_baseline.py`.
- Dense query embeddings are not repeated per model for retrieval-based pipelines because retrieval is cached by question.
- Direct-KB is expensive: it sends all 205 normalized rows for every direct-KB call.
- RAG prompts with 45 parent chunks are likely slow for large models and may exceed context windows.
- Full run size is 7 models x 4 pipelines x 50 questions = 1400 generation calls. This is large but feasible sequentially if timeouts and resume are implemented.
- RAGAS evaluation is potentially much more expensive than generation. External script may generate references for 50 questions and score 28 response columns x 50 questions x up to 4 metrics.
- RAGAS batching is not implemented in the external script; it scores sequentially with caching.
- Top-45 parent chunks are not empirically justified. They increase prompt cost and still may not solve exhaustive recall.
- Large local models on 24GB VRAM may offload or fail with long contexts because KV cache memory is not accounted for.

## 10. Research Validity Risks

- Pipeline differences are confounded with prompt differences. `rag_only` and `hybrid_rag` differ mostly in formatting instructions, not evidence permission.
- Pipelines receive unequal information by design. That is acceptable only if framed as evidence-condition comparison, not a pure model capability comparison.
- `pretrained_only` is expected to underperform on KB-specific questions and should be treated as a no-KB baseline.
- `direct_kb` has both an advantage and a disadvantage: it receives all evidence in principle, but may overflow context and be computationally harder.
- RAGAS metrics are currently not fed the same contexts used during generation.
- The external judge/reference model can bias results and should be disclosed.
- Generated references from the judge model may not match human-validated golden answers.
- Golden answers may be incomplete or may encode a row-level vs company-level interpretation. This must be audited for list/count questions.
- Retrieval quality and generation quality are not separately measured. Retrieval recall should be evaluated before model answer quality is interpreted.
- Lowercased company names may hurt exact formatting and perceived answer quality.
- Smoke outputs already show inconsistent counts for the same first question across pipelines/models, which suggests retrieval, prompt, or context-window effects must be diagnosed before reporting.

## 11. Recommended Fixes

### Critical fixes before running experiments

1. **Fix RAGAS context alignment.**
   - File/function affected: `build_ragas_report.py`; external `evaluate_ragas_ollama.py`.
   - Current problem: Contexts are grouped only by question during evaluation.
   - Exact recommended change: Store/evaluate one long-format row per `(question_id, model, pipeline)` with its own `contexts` list. Do not use a question-only context map. If Excel is still used, include model/pipeline in the retrieval sheet and have the evaluator join by `(question, response_column/pipeline)`.
   - Why it matters: Without this, faithfulness and groundedness scores are invalid.

2. **Use human golden answers for answer accuracy.**
   - File/function affected: external evaluator lines 514-527.
   - Current problem: The evaluator generates references from context instead of using `golden_answers`.
   - Exact recommended change: Read `golden_answers` or, better, read JSONL `ground_truth` directly and pass it as `reference` for `AnswerAccuracy`.
   - Why it matters: The project claims comparison against human-validated answers.

3. **Make RAGAS evaluator part of the repository or package it as a documented dependency.**
   - File/function affected: add `georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_ragas_evaluation.py` or `runtime_pipeline/evaluation/run_ragas.py`; update `requirements.txt`.
   - Current problem: Evaluation depends on an external file outside the repo and undeclared dependencies.
   - Exact recommended change: Move or copy the evaluator into the repo, add dependencies, and add a smoke test.
   - Why it matters: Reproducibility.

4. **Record and control context-window behavior.**
   - File/function affected: `llm_adapter.py`, `run_baseline.py`, `direct_kb_pipeline.py`.
   - Current problem: No `num_ctx`, prompt token/char count, or truncation flag.
   - Exact recommended change: Add `num_ctx` option to `OllamaAdapter`, record `prompt_char_count`, approximate token count, `context_count_used`, and `context_truncated` in JSONL. Fail or warn if prompt exceeds configured context.
   - Why it matters: Direct-KB and top-45 RAG results may be invalid if context is truncated.

5. **Fix or redefine `hybrid_rag`.**
   - File/function affected: `run_hybrid_rag.py` prompt and README.
   - Current problem: Prompt forbids outside knowledge, contradicting "hybrid" claim.
   - Exact recommended change: Either rename it to `rag_formatted` or change the prompt to explicitly allow limited model reasoning while requiring all KB-specific facts to be grounded in retrieved context.
   - Why it matters: Pipeline labels must match experimental conditions.

6. **Fix failing tests before full runs.**
   - File/function affected: `run_all_pipelines.py`, `tests/runtime_pipeline/test_run_all_pipelines.py`.
   - Current problem: Test expects dense/sparse context columns not present in workbook.
   - Exact recommended change: Either restore those columns or update the test and document that dense/sparse contexts are only in sidecar JSON.
   - Why it matters: Red tests signal broken output contracts.

### Important fixes before writing results

1. **Add retrieval recall checks.**
   - File/function affected: new tests or analysis script under `runtime_pipeline/evaluation`.
   - Current problem: No direct measurement that top-45 contains all gold entities for list/count questions.
   - Exact recommended change: For a small set of gold questions, check whether expected companies appear in retrieved parent chunks before generation.
   - Why it matters: Separates retrieval failure from generation failure.

2. **Add structured retrieval/filtering for deterministic questions.**
   - File/function affected: retrieval pipeline or a new structured query helper.
   - Current problem: Exact count/list/highest questions are handled by fuzzy retrieval plus LLM counting.
   - Exact recommended change: Add column-aware filters for category, OEM, location, employment, and role questions when the question pattern is clear.
   - Why it matters: Research answers become more accurate and explainable.

3. **Record run manifest metadata.**
   - File/function affected: `run_baseline._write_config()`.
   - Current problem: Config lacks KB hash, question file hash, DB counts, embedding model, reranker model, Ollama model IDs, git commit, and dirty status.
   - Exact recommended change: Add these fields to `config.json`.
   - Why it matters: Reproducibility.

4. **Add resume/skip-completed behavior.**
   - File/function affected: `run_baseline.py`.
   - Current problem: Long runs cannot resume cleanly.
   - Exact recommended change: If output JSONL exists, validate completed question IDs and skip them unless `--overwrite` is passed.
   - Why it matters: 1400 calls can fail halfway.

5. **Preserve original company casing.**
   - File/function affected: `loader.clean_company()`.
   - Current problem: Company names are lowercased.
   - Exact recommended change: Preserve display value and optionally add a separate normalized/search column.
   - Why it matters: Human-readable outputs and exact-answer comparisons.

6. **Use score-aware hybrid fusion.**
   - File/function affected: `merger.py`, retriever result schemas.
   - Current problem: Fusion is concatenate-and-dedupe.
   - Exact recommended change: Store sparse rank/score and dense rank/distance; implement RRF or weighted normalized fusion before parent mapping.
   - Why it matters: More defensible hybrid retrieval.

7. **Add evaluation status/error logging.**
   - File/function affected: external/internal RAGAS evaluator.
   - Current problem: Metric exceptions become `NaN` without error detail.
   - Exact recommended change: Store `evaluation_status`, `error_message`, metric-level error payloads, and count failed rows in summaries.
   - Why it matters: Avoid silent score inflation.

### Optional improvements

1. **Produce one canonical `generation_results.jsonl`.**
   - Current problem: Current baseline writes one JSONL per model/pipeline.
   - Exact recommended change: Keep per-combination files if useful, but also write a single long-format JSONL for evaluation.
   - Why it matters: Easier joins and RAGAS ingestion.

2. **Add prompt snapshots or prompt hashes.**
   - Current problem: JSONL stores contexts and answer but not the exact final prompt.
   - Exact recommended change: Store `prompt_hash`, `prompt_char_count`, and optionally a prompt sidecar file.
   - Why it matters: Auditability.

3. **Add CLI validation for model availability.**
   - Current problem: Missing Ollama models fail only during generation.
   - Exact recommended change: Query `ollama list` or `/api/tags` before a run.
   - Why it matters: Faster failure.

4. **Add smoke RAGAS test with 1 question and 1 model.**
   - Current problem: No checked-in evaluator test.
   - Exact recommended change: Test that answer accuracy reads the human ground truth and that contexts are pipeline-specific.
   - Why it matters: Prevents recurrence of context-mixing bug.

## 12. Minimal Test Plan

Run these before any full 1400-call experiment.

### Environment and dependency checks

```bash
cd /home/sm11926/GNEM_Retrival_RAG
source .venv/bin/activate
python -c "from georgia_ev_intelligence.shared import config; print(config.GNEM_EXCEL)"
python -c "import pandas, openpyxl, psycopg2, sentence_transformers, rank_bm25; print('core deps ok')"
python -c "import ragas; print(ragas.__version__)"
```

Expected today: the final command fails in `.venv`; fix before claiming repo-local RAGAS reproducibility.

### KB artifact checks

```bash
python - <<'PY'
import pandas as pd
for path in [
    "georgia_ev_intelligence/outputs/Normalized_kb.xlsx",
    "georgia_ev_intelligence/outputs/parent_chunks.xlsx",
    "georgia_ev_intelligence/outputs/child_chunks.xlsx",
    "kb/Human validated 50 questions.xlsx",
]:
    df = pd.read_excel(path)
    print(path, df.shape)
PY
```

Expected:

```text
Normalized_kb.xlsx: 205 rows
parent_chunks.xlsx: 205 rows
child_chunks.xlsx: 1025 rows
Human validated 50 questions.xlsx: 50 rows
```

### PostgreSQL integrity checks

```bash
python - <<'PY'
import psycopg2
from georgia_ev_intelligence.shared import config
conn = psycopg2.connect(config.NEON_DATABASE_URL)
queries = {
    "parents": "select count(*) from parent_chunks",
    "children": "select count(*) from child_chunks",
    "distinct_child_parents": "select count(distinct parent_record_id) from child_chunks",
    "orphans": "select count(*) from child_chunks c left join parent_chunks p on c.parent_record_id=p.record_id where p.record_id is null",
    "vector_dims": "select vector_dims(embedding), count(*) from child_chunks group by 1",
    "child_types": "select chunk_type, count(*) from child_chunks group by chunk_type order by chunk_type",
}
with conn, conn.cursor() as cur:
    for name, sql in queries.items():
        cur.execute(sql)
        print(name, cur.fetchall())
conn.close()
PY
```

Expected:

```text
parents = 205
children = 1025
distinct_child_parents = 205
orphans = 0
vector_dims = 768 for 1025 rows
each child type = 205
```

### Rebuild smoke test

```bash
python -m georgia_ev_intelligence.shared.data.loader
python -m georgia_ev_intelligence.offline_pipeline.index_pgvector --dry-run --preview 3
```

Expected: 205 parents and 1025 children.

### Retrieval sanity checks

```bash
python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_retrieval_only --limit 3
```

Inspect the generated workbook and verify:

- `parent_context_count_after_rerank` is nonzero and usually 45.
- Dense and sparse contexts are populated.
- Retrieved contexts contain plausible companies for the first three questions.

### BM25 tokenizer checks

```bash
python - <<'PY'
from georgia_ev_intelligence.runtime_pipeline.retrieval.bm25_retriever import tokenize_bm25
for text in ["Hyundai-Kia", "Tier 1/2", "LG Energy Solution's plant", "GA Georgia"]:
    print(text, "=>", tokenize_bm25(text))
PY
```

Expected: hyphenated/slash terms expand into compound and subterms.

### Parent-child mapping test

```bash
pytest -q tests/runtime_pipeline/test_hybrid_retrieval_orchestrator.py
```

Expected: pass.

### Current targeted test suite

```bash
pytest -q tests/runtime_pipeline/test_hybrid_retrieval_orchestrator.py \
          tests/runtime_pipeline/test_run_retrieval_only.py \
          tests/runtime_pipeline/test_run_all_pipelines.py
```

Current result: 9 passed, 1 failed. Fix before full experiments.

### Pipeline context/no-context smoke test

```bash
python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_baseline \
  --models qwen2.5:7b \
  --pipelines pretrained_only rag_only hybrid_rag direct_kb \
  --limit 1
```

Then inspect JSONL:

```bash
python - <<'PY'
import json
from pathlib import Path
run = sorted(Path("georgia_ev_intelligence/outputs/baselines").iterdir())[-1]
print("run", run)
for path in sorted(run.glob("*.jsonl")):
    row = json.loads(path.read_text().splitlines()[0])
    print(path.name, row["pipeline"], row["model"], len(row["contexts"]), row["answer"][:80])
PY
```

Expected:

- `pretrained_only`: 0 contexts.
- `rag_only`: retrieved contexts, usually up to 45.
- `hybrid_rag`: same retrieved contexts as `rag_only`.
- `direct_kb`: 205 contexts.

### RAGAS workbook alignment test

```bash
python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.build_ragas_report \
  --run-dir georgia_ev_intelligence/outputs/baselines/<RUN_DIR>
```

Then inspect:

```bash
python - <<'PY'
import pandas as pd
path = "georgia_ev_intelligence/outputs/baselines/<RUN_DIR>/ragas_report.xlsx"
print(pd.read_excel(path, sheet_name="responses").shape)
retrieval = pd.read_excel(path, sheet_name="retrieval")
print(retrieval.groupby(["question", "pipeline"]).size().head(20))
PY
```

Before using RAGAS, fix the evaluator so it joins contexts by pipeline/model, not just by question.

### Reranker check

Add or run a test with a fake reranker that reverses parent order and assert `retrieve_with_sources().parent_contexts` follows the reranker order. The existing orchestrator tests partially cover top-k behavior, but a ranking-change assertion is useful before reporting.

## 13. Final Recommendation

- Can I trust the current outputs? **No, not as research results.** You can trust them as smoke-test generation artifacts, but not as evaluated research baselines.
- Should I rerun experiments? **Yes.** First fix the evaluation context alignment, human-gold answer usage, context-window logging, `hybrid_rag` definition, and failing test. Then rerun generation and evaluation.
- Minimum to fix before presenting to your professor:
  1. Make RAGAS evaluation pipeline-specific and use the correct contexts per answer.
  2. Use human-validated golden answers for answer accuracy.
  3. Bring the evaluator and dependencies into the repo or document/package them reproducibly.
  4. Record context-window settings and prompt lengths; set `num_ctx` explicitly.
  5. Fix the failing test suite.
  6. Clarify or rename `hybrid_rag`.
- Limitations to mention:
  - `pretrained_only` is a no-KB baseline, not a fair competitor for KB-specific answers.
  - `direct_kb` may be constrained by context-window limits and is not a retrieval method.
  - RAG retrieval top-45 may not capture all evidence for exhaustive list/count questions.
  - RAGAS scores depend on the judge model and metric implementation.
  - Golden answers may encode row-level vs company-level assumptions.
  - Company capitalization is normalized/lowercased in the KB.

Bottom line: the generation architecture is moving in the right direction, but the current evaluation path can mix contexts across pipelines and ignore human references. Fix that before the full 1,400-call run, otherwise the comparison may look polished while measuring the wrong thing.
