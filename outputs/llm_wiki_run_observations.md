# LLM Wiki Full Run Observations

Run branch: `feature/final-route-generation`

Model requested for ingestion: `qwen2.5:14b`

Source directory: `kb/raw_docs/*.jsonl`

## 2026-06-15 23:02 EDT

- Active command: `wiki_cli ingest --source kb/raw_docs/ddg_search.jsonl`.
- Active ingest process environment confirmed `OLLAMA_LLM_MODEL=qwen2.5:14b`.
- `ollama ps` showed `qwen2.5:14b` loaded, 100% GPU, 32k context.
- Wiki state observed around this point:
  - Documents processed: 15
  - Pages created: 2
  - Pages: `Duckyang`, `Georgia`
- The `stats` command printed `gemma3:27b` because `.env` still has `OLLAMA_LLM_MODEL=gemma3:27b`; this did not reflect the active ingest model.
- Early quality signal was good for Duckyang: apparel/contact-info noise was filtered out, while the EV battery plant and Georgia investment facts were retained.
- The `Georgia` page was relevant but broad. It included ecosystem/workforce claims and a short Hanwha Qcells fact; acceptable as a location/context page but should be reviewed after a clean run.

## 2026-06-15 23:08 EDT

- Progress sample:
  - 23:03: 28 processed, 2 pages
  - 23:04: 45 processed, 2 pages
  - 23:05: 61 processed, 2 pages
  - 23:06: 76 processed, 3 pages
  - 23:07: 95 processed, 3 pages
  - 23:08: 118 processed, 3 pages
- Page creation rate stayed low during this window, which indicates the filter rejects many documents instead of creating pages for everything.
- New page `Xcel Energy` was questionable for the Georgia EV wiki:
  - `has_georgia_presence=false`
  - facts refer to helping PepsiCo's manufacturing facility project and an economic development rate
  - likely not central to the Georgia EV intelligence domain.

## 2026-06-15 23:09-23:11 EDT

- Clear noisy pages observed:
  - `Energy Forward`: created from password reset / WordPress login facts.
  - `Energy Forward Publishing`: publisher/editorial meta content, `has_georgia_presence=false`.
  - `S&P Global`: conference-panel content, `has_georgia_presence=false`.
  - `Perpetual Next`: biomethanol/organic waste project, `has_georgia_presence=false`.
  - `TotalEnergies`: offshore wind / LNG facts, `has_georgia_presence=false`.
- The run was stopped after approximately 183 processed documents and 12 pages.
- Research conclusion at this checkpoint: Duckyang quality improved, but full-run precision is not reliable yet. The current prompt/code still allows non-Georgia, non-project, publisher, website-account, and broad energy pages if they produce two facts.
- Recommended intervention before rerun:
  - hard reject `has_georgia_presence=false` unless the entity is a known Georgia context page,
  - reject publisher/login/website-account facts,
  - enforce allowed entity types,
  - require project/investment/facility/job/product facts for company pages,
  - remove current `kb/wiki` and rerun after the code/prompt patch.
