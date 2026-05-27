# Codex Run Report: Baseline Generation and Resume

## Executive Summary

The full baseline output is now complete in `georgia_ev_intelligence/outputs/baselines/20260520_200016`.

- Expected output: 7 models x 4 pipelines x 50 questions = 1,400 JSONL rows.
- Final output: 28 JSONL files, 1,400 rows.
- Bad JSON rows: 0.
- Error answer rows: 2.
- The first run stopped in the middle of `qwen2.5:32b` / `direct_kb` after only 30/50 rows had been written.
- The resumed run filled the remaining `qwen2.5:32b` / `direct_kb` rows and completed `gemma3:27b` and `qwen2.5:14b`.

The dataset is structurally complete, but the two timeout rows must be rerun or excluded before using this run for final RAGAS metrics.

## Run Folder

`georgia_ev_intelligence/outputs/baselines/20260520_200016`

Monitor logs:

- Initial run monitor: `codex_run_monitor.tsv`
- Resume monitor: `codex_resume_monitor.tsv`

## Completion Status

| Check | Result |
|---|---:|
| JSONL files | 28 |
| Expected JSONL files | 28 |
| Total rows | 1,400 |
| Expected rows | 1,400 |
| Bad JSON rows | 0 |
| Error answer rows | 2 |
| Incomplete files | 0 |

## Error Rows

These rows contain timeout strings instead of model answers:

| File | Line | Question ID | Pipeline | Model | Error |
|---|---:|---:|---|---|---|
| `qwen2.5_32b__rag_only.jsonl` | 8 | 8 | `rag_only` | `qwen2.5:32b` | Ollama read timeout after 300s |
| `qwen2.5_32b__direct_kb.jsonl` | 1 | 1 | `direct_kb` | `qwen2.5:32b` | Ollama read timeout after 300s |

Affected questions:

- Q8: `Which county have the highest total Employment among Tier 1 suppliers only?`
- Q1: `Show all "Tier 1/2" suppliers in Georgia, list their EV Supply Chain Role and Product / Service.`

Recommendation: rerun exactly these two failed rows before final metrics, or mark them explicitly as failed generations and exclude them from aggregate RAGAS averages.

## Initial Run Timeline

The initial run began around `2026-05-20 20:00:16` and stopped on `2026-05-21` during `qwen2.5:32b direct_kb`. The monitor log ended at `2026-05-21 12:26:53`, and the last file later observed from that stopped run was `qwen2.5_32b__direct_kb.jsonl` with 30/50 rows, mtime `2026-05-21 12:29:05`.

Important initial-run observations:

- `qwen2.5:7b`, `llama3.1:8b`, `mistral-small3.2:24b`, and `qwen3.5:35b-a3b` completed all 4 pipelines.
- `qwen3.5:35b-a3b` was partially CPU-offloaded, reported by Ollama as approximately `19%/81% CPU/GPU`, and was the major runtime bottleneck.
- `qwen2.5:32b` was also partially CPU/GPU offloaded in the resume monitor, reported around `24%/76% CPU/GPU`.
- The initial run stopped before a clean completion marker. At that time, Ollama was later found down and the `run_baseline` process was gone.

## Resume Timeline

The resumed command continued the existing run folder rather than starting a new folder.

- Resume monitoring started: `2026-05-21 14:33:21`.
- At monitor start, `qwen2.5_32b__direct_kb.jsonl` was already 41/50 rows.
- `qwen2.5:32b direct_kb` reached 50/50 at about `2026-05-21 14:46:25`.
- `gemma3:27b` ran from about `2026-05-21 14:46:25` to `2026-05-21 16:11:17`.
- `qwen2.5:14b` ran from about `2026-05-21 16:11:17` to `2026-05-21 16:41:36`.
- Final row count reached 1,400/1,400 at `2026-05-21 16:41:57` in the monitor.

## Per-Model Timing

Timings are based on file modification times and monitor observations, so they should be treated as close operational estimates rather than exact profiler timings.

| Model | Approx start | Approx end | Approx wall time | Notes |
|---|---|---|---:|---|
| `qwen2.5:7b` | 2026-05-20 20:00:16 | 2026-05-20 20:17:08 | ~17 min | Completed cleanly. |
| `llama3.1:8b` | 2026-05-20 20:17:08 | 2026-05-20 20:33:08 | ~16 min | Completed cleanly. |
| `mistral-small3.2:24b` | 2026-05-20 20:33:08 | 2026-05-20 21:54:04 | ~1h 21m | Completed cleanly; direct-KB was longest. |
| `qwen3.5:35b-a3b` | 2026-05-20 21:54:04 | 2026-05-21 08:58:25 | ~11h 04m | Severe slowdown due partial CPU offload. |
| `qwen2.5:32b` | 2026-05-21 08:58:25 | 2026-05-21 14:45:53 | ~5h 47m wall incl. interruption | 2 timeout rows; run stopped mid-direct-KB and was resumed. |
| `gemma3:27b` | 2026-05-21 14:46:25 | 2026-05-21 16:11:17 | ~1h 25m | Completed cleanly on GPU. |
| `qwen2.5:14b` | 2026-05-21 16:11:17 | 2026-05-21 16:41:36 | ~30 min | Completed cleanly on GPU. |

## Per-Pipeline Timing by Model

| Model | `rag_only` | `hybrid_rag` | `pretrained_only` | `direct_kb` | Notes |
|---|---:|---:|---:|---:|---|
| `qwen2.5:7b` | ~4m 34s | ~3m 41s | ~1m 34s | ~7m 03s | Clean. |
| `llama3.1:8b` | ~4m 32s | ~4m 18s | ~1m 39s | ~5m 31s | Clean. |
| `mistral-small3.2:24b` | ~12m 49s | ~15m 23s | ~3m 55s | ~48m 49s | Direct-KB much slower. |
| `qwen3.5:35b-a3b` | ~2h 41m | ~2h 47m | ~1h 49m | ~3h 47m | Partial CPU offload made this extremely slow. |
| `qwen2.5:32b` | ~1h 13m | ~1h 09m | ~24m | ~1h 10m active estimate / ~3h wall with interruption | 2 timeout rows. |
| `gemma3:27b` | ~15m 50s | ~16m 27s | ~6m 49s | ~46m 18s | Clean; GPU temp around 81-83 C during sustained generation. |
| `qwen2.5:14b` | ~11m 36s | ~8m 25s | ~1m 42s | ~8m 36s | Clean and fast. |

## Hardware and Runtime Notes

- `qwen3.5:35b-a3b` did not fit cleanly on the 24 GB GPU under the observed context settings and was partially CPU-offloaded. This caused the largest runtime increase.
- `qwen2.5:32b` also showed CPU/GPU split in the resume monitor, around `24%/76% CPU/GPU`.
- `gemma3:27b` ran on GPU and used about 21.6 GB VRAM, with sustained temperature around 81-83 C.
- `qwen2.5:14b` ran on GPU and used about 16.3 GB VRAM.
- `direct_kb` is consistently the most expensive grounded pipeline because it sends the full normalized KB context.
- `pretrained_only` is consistently fastest because it sends no KB context.

## Mistakes / Issues to Note

1. The first run stopped before completion.
   - It stopped during `qwen2.5:32b direct_kb`.
   - The file had only 30/50 rows when the failure was discovered.
   - Ollama was down when checked afterward.

2. The resume approach completed the missing work, but the code itself still has no native resume flag.
   - This means future interruptions require careful manual resume logic.
   - Running the normal command again would create a new timestamped run folder.
   - Forcing the same folder with the current writer behavior would risk overwriting files because the runner opens output files with write mode.

3. Two LLM calls timed out at 300 seconds.
   - Both are from `qwen2.5:32b`.
   - These rows are structurally valid JSONL but invalid as model answers.

4. The 35B and 32B model timings are confounded by CPU offload.
   - Their quality results can still be evaluated, but runtime comparisons against fully GPU-resident models are not fair unless this hardware limitation is reported.

5. `direct_kb` may have context-window and latency risks.
   - It passes the full normalized KB, which is useful as a no-retrieval/full-evidence baseline, but runtime and truncation behavior should be discussed as a limitation.

## Final Output File Status

Every JSONL file now has 50 rows.

Files with errors:

- `qwen2.5_32b__rag_only.jsonl`: 50 rows, 1 timeout row.
- `qwen2.5_32b__direct_kb.jsonl`: 50 rows, 1 timeout row.

All other JSONL files have 50 rows and 0 error rows.

## Recommendation Before RAGAS

Do not run final aggregate RAGAS metrics directly on the two timeout rows as if they were normal answers. Best next step:

1. Rerun only the two failed rows for `qwen2.5:32b`.
2. Replace those two JSONL records in-place, preserving the same question IDs and schema.
3. Re-run the row count and error check.
4. Then build the RAGAS workbook and run metrics.

If you choose not to rerun them, explicitly exclude these two rows from metric averages and report that `qwen2.5:32b` had two generation timeouts.

## Verification Commands

Check final row counts and timeout rows:

```bash
cd /home/sm11926/GNEM_Retrival_RAG
python - <<'PYCHECK'
import json
from pathlib import Path

run = Path("georgia_ev_intelligence/outputs/baselines/20260520_200016")
total = errors = bad_json = 0

for p in sorted(run.glob("*.jsonl")):
    lines = p.read_text(encoding="utf-8").splitlines()
    file_errors = 0
    for line in lines:
        try:
            row = json.loads(line)
            if str(row.get("answer", "")).startswith("ERROR:"):
                file_errors += 1
        except Exception:
            bad_json += 1
            file_errors += 1
    total += len(lines)
    errors += file_errors
    print(p.name, len(lines), "errors", file_errors)

print("TOTAL", total, "ERRORS", errors, "BAD_JSON", bad_json)
PYCHECK
```

Expected current result:

```text
TOTAL 1400 ERRORS 2 BAD_JSON 0
```
