# Gemini QA Benchmark — three modes, one evaluation harness

Answers a fixed question set with Google's Gemini in three configurations, then
scores all three against each other with four layers of automatic evaluation
plus a structured human review.

| # | Mode | Evidence the model sees |
|---|------|-------------------------|
| 1 | `llm_only` | Nothing. The model's own parametric knowledge. |
| 2 | `llm_kb` | Top-k records retrieved from the internal company spreadsheet. |
| 3 | `llm_kb_web` | The same internal records **plus** live Google Search grounding. |

The prompts for the three modes are deliberately parallel — same task framing,
same output shape, same style rules. The only variable is the evidence, so a
score difference is attributable to the evidence rather than to prompt wording.

---

## 1. Getting a Gemini API key

1. Go to **https://aistudio.google.com/apikey** and sign in with a Google account.
2. **Create API key** → pick a Google Cloud project (or let it create one).
3. Copy the key (starts with `AIza…`) — it is shown once.
4. Put it in `.env` next to this README:

```bash
cp .env.example .env
# then edit .env:
GEMINI_API_KEY="AIza..."
```

Notes on the key, since they matter for a benchmark run:

- **Two flavours of key exist.** An *AI Studio* key (the link above) bills against
  the Gemini Developer API and has a free tier with per-minute and per-day request
  caps. A *Vertex AI* key/service account bills through Google Cloud, has no free
  tier, and needs a project with billing enabled. This project uses the AI Studio
  flavour; `google-genai` speaks both if you later switch.
- **Free tier is enough to try this, not to finish it.** 30 questions × 3 modes
  ≈ 90 answer calls, and the judge and RAG metrics add roughly 6–8 more calls per
  answer. Expect ~700–900 calls for a full scored run. On the free tier you will
  hit rate limits; the client retries with exponential backoff, so it will
  complete, just slowly. Enabling billing on the project removes the wall.
- **Google Search grounding (mode 3) is billed separately** from tokens, per
  grounded request, and is not part of the free tier on every model. Check the
  current pricing page before running mode 3 across the full set.
- **Cost control is built in.** Every API call is cached on disk under `.cache/`,
  keyed by model + prompt + settings, so re-running the scoring after a crash or
  a code change costs nothing for the calls that already succeeded.
- **Never commit the key.** `.env` and `.cache/` are gitignored. If a key leaks,
  revoke it in AI Studio — keys cannot be rotated in place.
- **No key? The whole thing still runs.** With `GQE_MOCK=1` (or no
  key set at all) every call returns a deterministic stub, so you can exercise
  the pipeline, the report and the review workbook offline. Mock answers are
  labelled `[MOCK …]` and every output row carries `mocked: true`.

---

## 2. Install and smoke-test

```bash
pip install -r requirements.txt
pip install -e .          # or prefix every command with PYTHONPATH=src
make test          # 24 tests, offline
make demo          # full pipeline on 3 questions, no API key needed
```

## 3. Run for real

```bash
python -m gqe.cli all -v            # 30 questions × 3 modes → answers, scores, report, review workbook
```

or step by step, which is what you want the first time:

```bash
python -m gqe.cli answer  -v                         # → runs/<timestamp>/answers.jsonl
python -m gqe.cli score   -v                         # → runs/<timestamp>/scored.jsonl
python -m gqe.cli report                             # → report.md + report.xlsx
python -m gqe.cli pairwise                           # → blind head-to-head between modes
python -m gqe.cli workbook --reviewers ana,ben       # → manual_review.xlsx (blinded)
# ... humans fill it in, then:
python -m gqe.cli agreement --workbook manual_review.xlsx
```

Useful flags:

| Flag | Effect |
|------|--------|
| `--limit 5` | Run a handful of questions first. Always do this before the full set. |
| `--modes llm_kb,llm_kb_web` | Restrict to some modes. |
| `--repeats 3` | Answer each question 3× — enables the self-consistency metric. |
| `--no-judge` / `--no-rag-metrics` | Skip the model-graded layers (cheap deterministic run). |
| `--judge-samples 3` | Grade 3× and take the median; reports the spread so you can see judge instability. |

---

## 3b. Budgeting the run before you spend anything

```bash
PYTHONPATH=src python scripts/estimate_cost.py
PYTHONPATH=src python scripts/estimate_cost.py --price-in 0.30 --price-out 2.50   # your current prices
```

It builds the real prompts — same system prompts, same retrieved context, same
question set — and reports calls, input tokens, output tokens and embedding
tokens per stage, so you can see which layer is expensive before running it.
Flags mirror the CLI (`--limit`, `--modes`, `--repeats`, `--judge-samples`,
`--pairwise`, `--no-score`), plus `--answer-ratio` to tune the assumed answer
length.

Estimated at ~4 chars/token without calling a tokenizer, so it works with no
API key; treat it as a budget (±15%), not an invoice. Mode 3's Google Search
grounding is billed per grounded request on top of tokens and is not included.

---

## 4. Your own questions and your own data

Two inputs, both swappable without touching code:

```bash
QUESTIONS_PATH="data/my_questions.xlsx"     # needs a question column; golden-answer and category columns optional
COMPANY_DATA_PATH="data/my_company_data.xlsx"
N_QUESTIONS="30"
```

- **Questions** — `.xlsx` or `.csv`. Column names are matched loosely
  (`question` / `query` / `prompt`; `answer` / `golden answer` /
  `human validated answers`; `category` / `use case category`). Without a
  golden-answer column the reference-based metrics are skipped and the
  reference-free ones (faithfulness, answer relevancy, context precision) still work.
- **Company data** — `.xlsx`. Every row becomes one retrievable record; every
  column becomes a `Field: value` line in that record. No schema assumptions.

Shipped defaults: `data/questions.xlsx` (the first 30 of the 50 human-validated
Georgia EV questions) and `data/company_data.xlsx` (205 company records).

---

## 5. What comes out

```
runs/<timestamp>/
  answers.jsonl        one row per (question × mode × repeat): answer, contexts, web sources, latency, tokens
  scored.jsonl         the same rows plus every metric
  report.md            the tables, readable in a terminal or a PR
  report.xlsx          summary / by_category / per_question / significance / self_consistency / cost
  pairwise.jsonl       blind head-to-head judgements, if you ran `pairwise`
  manual_review.xlsx   the blinded human-review workbook
  agreement.xlsx       human vs judge, once reviews come back
```

---

## 6. Evaluation

Four automatic layers plus a human one. The short version:

| Layer | What it answers | Needs a golden answer? | API cost |
|-------|-----------------|------------------------|----------|
| 1. Deterministic | Does the answer contain the right companies and the right numbers? | yes | none |
| 2. Retrieval | Did the retriever even surface the right records? | yes (silver labels) | none |
| 3. LLM judge | Is the answer correct, complete, grounded, relevant? | yes | high |
| 4. RAG metrics | Is it faithful to the context? Is the context any good? | mostly no | high |
| 5. Human review | All of the above, for real, on a blinded sample. | yes | reviewer time |

**Read [EVALUATION.md](EVALUATION.md)** for the full method: what each metric
catches, what it misses, how to run the human review, how to validate the judge
against the humans before trusting it, and how to read the numbers at n=30.

---

## 7. Layout

```
src/gqe/
  config.py              settings from the environment
  data.py                question set + company workbook loaders
  retrieval.py           BM25, Gemini embeddings, RRF fusion
  gemini_client.py       cached, retrying, mockable Gemini wrapper
  modes.py               the three pipelines and their prompts
  cli.py                 answer / score / pairwise / report / workbook / agreement / all
  evaluation/
    deterministic.py     entity + numeric + overlap metrics
    retrieval_metrics.py recall@k, precision@k, MRR, nDCG
    judge.py             rubric-anchored grading + blind pairwise
    rag_metrics.py       faithfulness, answer relevancy, context precision/recall
    aggregate.py         summary, per-category, paired bootstrap, cost
  manual/
    workbook.py          blinded review workbook builder
    agreement.py         kappa, Spearman, inter-annotator, rank agreement
tests/                   24 tests, all offline
```
