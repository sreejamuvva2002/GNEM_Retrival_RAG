# Evaluation methodology

How to tell whether mode 1, 2 or 3 is actually better — and whether you are
allowed to believe the numbers.

The guiding principle: **no single metric is trustworthy on its own.** Text
overlap rewards copying, an LLM judge rewards fluency, a human reviewer gets
tired at row 40. Each layer below fails differently, which is the point — you
believe a result when layers that fail differently agree.

---

## Layer 1 — Deterministic, reference-based
*`src/gqe/evaluation/deterministic.py` · no API calls · runs on every answer*

| Metric | What it catches | What it misses |
|---|---|---|
| `entity_recall` / `entity_precision` / `entity_f1` | **The primary metric for this dataset.** Golden answers are lists of companies. Recall = companies the answer omitted; precision = companies it invented. | Right company, wrong attribute ("Kia Georgia — battery cells"). |
| `entities_hallucinated` | Companies named in the answer but absent from the golden answer. | Not automatically wrong — the golden answer may be incomplete. Treat as a review queue, not a score. |
| `numeric_recall` / `numeric_match` | Wrong counts: "there are 12 Tier 1/2 suppliers" when the truth is 18. The single most visible failure mode here, and no overlap metric catches it. | Numbers that are right for the wrong reason. |
| `token_f1`, `rouge_l` | Gross content drift; a cheap regression tripwire between runs. | Rewards verbosity and copying. Never use alone to rank systems. |
| `is_refusal` | Abstention rate. A refusal is *good* when the data genuinely lacks the answer and *bad* when it does not — the report pairs it with `entity_recall` so you can tell which. | Hedged non-refusals ("it may be that…"). |
| `citation_rate` | Modes 2 and 3 are instructed to cite `[C1]`/`[WEB]`. Low rate = the model is ignoring the instruction, which usually means it is also ignoring the evidence. | Whether the citation is *correct* — that is layer 4's faithfulness. |

Cheap, reproducible, zero-cost: run these on every commit as the regression suite.

---

## Layer 2 — Retrieval quality (modes 2 and 3)
*`src/gqe/evaluation/retrieval_metrics.py` · no API calls*

`recall@k`, `precision@k`, `MRR`, `nDCG@k` against a silver relevance label:
a retrieved record counts as relevant if the golden answer names that company.

This layer exists to answer one diagnostic question: **when a grounded answer is
wrong, was the record missing from the context, or was it there and ignored?**

- `recall@k` low → fix retrieval (k, chunking, hybrid weights, query rewriting).
- `recall@k` high but `entity_recall` low → fix the prompt or the model; the
  evidence was in front of it.

Caveat: silver labels under-count records that support an answer without being
named in it, so absolute values run low. The mode-to-mode *comparison* is sound.

---

## Layer 3 — LLM-as-judge
*`src/gqe/evaluation/judge.py` · Gemini, structured output · the expensive layer*

Absolute grading, 1–5 on four dimensions, against the golden answer and the
context the model actually saw:

- **correctness** — do the facts agree with the reference?
- **completeness** — how much of the reference's content is covered?
- **groundedness** — are claims traceable to the supplied evidence? (fixed at 3 for mode 1, which has none)
- **relevance** — does it answer the question asked, without padding?

Plus `verdict` (pass/borderline/fail), `unsupported_claims`, and a written
rationale. The prompt forces evidence quotes *before* the scores, which is the
cheapest known suppressant of the "long answer = good answer" bias.

Known biases, and what this harness does about each:

| Bias | Mitigation here |
|---|---|
| Self-preference (a model likes its own output) | `GEMINI_JUDGE_MODEL` defaults to a *different, stronger* model than the answerer. |
| Verbosity bias | Rubric says length is not quality; evidence-first prompting; `answer_length_words` is reported so you can check the correlation yourself. |
| Position bias (pairwise) | Presentation order randomised per comparison, then unmapped. |
| Score drift between runs | Temperature 0, fixed rubric, cached calls; `--judge-samples 3` reports the spread. |
| Being wrong in a way nobody notices | **Layer 5.** The judge is validated against human labels before it is trusted. |

### Pairwise preference (`gqe pairwise`)
Blind A/B between two modes on the same question, with `tie` allowed. Pairwise
agreement with humans is consistently higher than absolute-score agreement, so
when two modes are within ~0.2 on `judge_overall`, the pairwise tally is the
tiebreaker, not the means.

---

## Layer 4 — Reference-free RAG metrics
*`src/gqe/evaluation/rag_metrics.py` · RAGAS-style, implemented directly on Gemini*

These mostly need no golden answer, which is what makes them usable in
production on questions nobody has hand-answered:

| Metric | Method | Reads as |
|---|---|---|
| `faithfulness` | Decompose the answer into atomic claims; ask whether the context entails each. | Share of claims the internal data actually supports. **The number to take to a risk conversation.** |
| `answer_relevancy` | Reverse-generate questions from the answer; embed; cosine against the real question. | Evasion and padding detector. |
| `context_precision` | Rank-weighted average precision over "was this record useful?" | Retriever noise. Low → you are burning context window on junk. |
| `context_recall` | Share of golden-answer sentences supported by the retrieved context. | Retriever misses. The one metric here that needs the golden answer. |

Implemented natively rather than via the `ragas` package so the prompts are
inspectable, the calls share the same cache, and the stack has one model provider.

Mode 1 has no context, so faithfulness and both context metrics are `NaN` there
by definition — not zero. Don't average across that.

---

## Layer 5 — Human evaluation
*`src/gqe/manual/` · the layer that makes the other four mean something*

### Building the workbook

```bash
python -m gqe.cli workbook --reviewers ana,ben --overlap 0.2
```

Produces `manual_review.xlsx` with four sheets: `instructions`, `rubric`,
`review` (what reviewers fill in), `key_do_not_open` (the blind-label mapping).

Design choices, each fixing a specific way manual evals go wrong:

- **Blinded.** Modes are replaced by per-question shuffled labels (System X/Y/Z).
  A reviewer who knows which one is "the RAG one" scores it higher.
- **Shuffled row order**, so nobody anchors on the first system they see.
- **Same rubric and anchors as the judge**, so human and judge scores are
  directly comparable and the kappa below is meaningful.
- **Dropdowns, not free text**, for scores. Free text goes in `notes`.
- **`reference_disputed` column.** Golden answers contain errors. Without a
  channel to say so, reviewers silently score around them and the data rots.
- **Overlap rows.** ~20% of rows are duplicated to a second reviewer. Without
  this you cannot distinguish a bad judge from an ambiguous rubric.

### Running the review

- 3–5 minutes per row; ~90 rows for 30 questions × 3 modes. Budget ~6 hours,
  split across at least two reviewers, at most 20 rows per sitting.
- Reviewers judge **against the reference and the evidence shown**, not their
  own knowledge. Anything else produces disagreement you cannot interpret.
- A blank cell is better than a guess.
- Run a **calibration pass first**: both reviewers score the same 5 rows, then
  compare and reconcile before doing the rest. Most disagreement is rubric
  interpretation, and it is far cheaper to fix in row 5 than in row 90.

### Reading the agreement report

```bash
python -m gqe.cli agreement --workbook manual_review.xlsx
```

| Table | Question it settles |
|---|---|
| `human_summary` | What the humans actually concluded, per mode: mean scores, `usable_as_is_rate`, `hallucination_rate`, `reference_disputed_rate`. **This is the ground truth of the whole exercise.** |
| `inter_annotator` | Do reviewers agree with each other? (quadratic-weighted Cohen's κ on overlap rows) |
| `human_vs_judge` | Does the judge agree with the humans, per dimension: κ, Spearman, exact and within-1 agreement, and the judge's bias (does it grade high or low?). |
| `mode_ranking` | Do humans and the judge *rank the three modes the same way*? A judge can be badly calibrated in absolute terms and still be perfectly usable if it orders systems correctly. |

Thresholds for the judge's κ against humans:

| κ | Verdict |
|---|---|
| < 0.4 | Do not use the judge unsupervised. Fix the rubric, then re-measure. |
| 0.4 – 0.6 | Usable for tracking regressions between runs. Not for go/no-go decisions. |
| > 0.6 | Comparable to a second human reviewer. Use it, spot-check 10% each run. |

Check `inter_annotator` **first**. If humans don't agree with humans, the judge
isn't the problem — the rubric is.

---

## Reading the results at n = 30

Thirty questions is a small sample. Three habits keep it honest:

1. **Use the paired bootstrap, not the means.** `report.xlsx` → `significance`
   reports the mean difference between each pair of modes with a 95% interval
   from 5,000 paired bootstrap resamples, plus win/loss/tie counts. **If the
   interval spans zero, report the modes as indistinguishable on that metric.**
   With n=30 a 0.2-point gap in `judge_overall` usually does span zero.
2. **Read the per-category table.** The interesting result is almost never
   "mode 3 wins overall" — it is "mode 2 wins on supply-chain mapping, mode 3
   wins on technology trends, mode 1 loses everywhere." That table is what
   drives a routing decision, and it survives small n better than the headline.
3. **Check stability before believing a winner.** `--repeats 3` populates
   `self_consistency`: the mean within-question spread across repeats. A mode
   that swings between runs is not better, it is noisier.

### The expected shape of the result

Worth stating up front so a surprising result gets investigated rather than shipped:

- **Mode 1** should score badly on `entity_recall` and `numeric_recall` — the
  internal company list is not in any model's training data. If it scores well,
  check for leakage: are these companies publicly documented, or is the judge
  rewarding fluent-sounding wrongness? Its `groundedness` is fixed at 3 and
  should not be compared across modes.
- **Mode 2** should win on everything reference-based. If it doesn't, layer 2
  tells you whether the cause is retrieval or generation.
- **Mode 3** should beat mode 2 on recency questions and roughly tie elsewhere,
  at higher latency and cost. Watch for its specific failure: web content
  *overriding* correct internal data. `faithfulness` (computed against internal
  context only) drops when this happens, which is exactly why that metric is
  computed against internal context only.

---

## What this harness does *not* measure

State these when presenting results, because someone will ask:

- **Golden-answer quality.** The whole reference-based stack inherits the errors
  in `data/questions.xlsx`. The `reference_disputed` column is the only
  instrument pointed at this — read it.
- **Freshness of the internal data.** If the workbook is a year stale, mode 2
  is confidently, faithfully, and measurably wrong. No metric here can see it.
- **Real user satisfaction.** Rubric scores are a proxy. `usable_as_is_rate`
  from the human review is the closest thing in this harness to the real thing.
- **Adversarial robustness.** Prompt injection via retrieved or web content is
  out of scope and needs its own test set.
- **Cost at scale.** `report.xlsx` → `cost` fills in only when you set
  `PRICE_INPUT_PER_1M` / `PRICE_OUTPUT_PER_1M` from the current pricing page.
  Grounded search requests in mode 3 are billed per request on top of tokens and
  are *not* included in that table.

---

## Suggested order of work

1. `make demo` — confirm the wiring, no key needed.
2. `python -m gqe.cli all --limit 5 -v` — 5 questions, real key. Read every
   answer by hand. This is where prompt bugs surface, not in the metrics.
3. Full run: `python -m gqe.cli answer -v` then `python -m gqe.cli score -v`.
4. `report` + `pairwise`. Look at `significance` before believing any gap.
5. `workbook` → calibration pass on 5 rows → full human review.
6. `agreement`. If κ > 0.6, subsequent runs can lean on the judge with a 10%
   human spot-check. If not, fix the rubric and re-review — do not proceed on
   judge scores alone.
