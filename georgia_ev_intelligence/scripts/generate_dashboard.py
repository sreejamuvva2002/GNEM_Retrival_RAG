"""
scripts/generate_dashboard.py
Generates a self-contained HTML comparison dashboard from format eval JSONL files.
Usage: venv\\Scripts\\python scripts\\generate_dashboard.py
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

_RESULTS_DIR = Path("outputs/format_eval")
_DASH_DIR    = Path("outputs/dashboard")
_DASH_DIR.mkdir(parents=True, exist_ok=True)

FORMAT_LABELS = {
    "F1_ONLY_RAG":           "F1 — Only RAG",
    "F2_NO_RAG":             "F2 — No RAG",
    "F3_RAG_PRETRAINED":     "F3 — RAG + Pre-trained",
    "F4_RAG_PRETRAINED_WEB": "F4 — RAG + Web",
}
METRICS = ["faithfulness","answer_relevancy","context_precision","context_recall","answer_correctness"]
METRIC_LABELS = {
    "faithfulness":       "Faithfulness",
    "answer_relevancy":   "Answer Relevancy",
    "context_precision":  "Context Precision",
    "context_recall":     "Context Recall",
    "answer_correctness": "Answer Correctness",
}
COLORS = ["#6366f1","#22d3ee","#f59e0b","#10b981","#ef4444","#a78bfa","#fb923c"]


def load_all_results() -> list[dict]:
    rows = []
    for p in sorted(_RESULTS_DIR.glob("f*_results.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def aggregate(rows: list[dict]) -> dict:
    """Build summary stats keyed by (format, model)."""
    buckets: dict[tuple, list] = defaultdict(list)
    for r in rows:
        key = (r.get("format","?"), r.get("model","?"))
        buckets[key].append(r)

    summary = {}
    for (fmt, model), items in buckets.items():
        metric_vals: dict[str, list] = {m: [] for m in METRICS}
        for item in items:
            scores = item.get("scores", {})
            for m in METRICS:
                v = scores.get(m)
                if v is not None:
                    try:
                        metric_vals[m].append(float(v))
                    except (TypeError, ValueError):
                        pass
        avgs = {m: (sum(v)/len(v) if v else None) for m, v in metric_vals.items()}
        summary[(fmt, model)] = {
            "format": fmt,
            "model": model,
            "count": len(items),
            "avgs": avgs,
            "overall": sum(v for v in avgs.values() if v is not None) /
                       max(1, sum(1 for v in avgs.values() if v is not None)),
            "items": items,
        }
    return summary


def generate_dashboard(out_path: Path | None = None) -> Path:
    rows = load_all_results()
    if not rows:
        print("No result files found in outputs/format_eval/. Run evaluations first.")
        return _DASH_DIR / "comparison_report.html"

    summary = aggregate(rows)
    summary_list = sorted(summary.values(), key=lambda x: (-x["overall"], x["format"]))

    # Build chart datasets
    unique_models  = sorted({s["model"]  for s in summary_list})
    unique_formats = sorted({s["format"] for s in summary_list})

    # Radar data per (format,model)
    radar_datasets = []
    for i, (key, s) in enumerate(summary.items()):
        radar_datasets.append({
            "label": f"{FORMAT_LABELS.get(s['format'], s['format'])} / {s['model']}",
            "data": [round(s["avgs"].get(m) or 0, 3) for m in METRICS],
            "borderColor": COLORS[i % len(COLORS)],
            "backgroundColor": COLORS[i % len(COLORS)] + "33",
        })

    # Bar chart: overall score per format+model
    bar_labels  = [f"{FORMAT_LABELS.get(s['format'],s['format'])}\n{s['model']}" for s in summary_list]
    bar_data    = [round(s["overall"], 3) for s in summary_list]
    bar_colors  = [COLORS[i % len(COLORS)] for i in range(len(summary_list))]

    # Table rows
    table_rows_html = ""
    for s in summary_list:
        avgs = s["avgs"]
        fa   = f"{avgs['faithfulness']:.2f}"       if avgs['faithfulness']       is not None else "—"
        ar   = f"{avgs['answer_relevancy']:.2f}"   if avgs['answer_relevancy']   is not None else "—"
        cp   = f"{avgs['context_precision']:.2f}"  if avgs['context_precision']  is not None else "—"
        cr   = f"{avgs['context_recall']:.2f}"     if avgs['context_recall']     is not None else "—"
        ac   = f"{avgs['answer_correctness']:.2f}" if avgs['answer_correctness'] is not None else "—"
        ov   = f"{s['overall']:.3f}"
        table_rows_html += f"""
        <tr>
          <td><span class="badge-fmt">{FORMAT_LABELS.get(s['format'], s['format'])}</span></td>
          <td><code>{s['model']}</code></td>
          <td class="num">{fa}</td><td class="num">{ar}</td>
          <td class="num">{cp}</td><td class="num">{cr}</td>
          <td class="num">{ac}</td>
          <td class="num bold">{ov}</td>
          <td class="num">{s['count']}</td>
        </tr>"""

    # Question-level drill-down (all rows)
    q_rows_html = ""
    for r in rows:
        sc    = r.get("scores", {})
        ac_sc = sc.get("answer_correctness")
        badge_cls = "badge-good" if (ac_sc or 0) >= 0.7 else ("badge-warn" if (ac_sc or 0) >= 0.4 else "badge-bad")
        ac_str = f"{ac_sc:.2f}" if ac_sc is not None else "—"
        q_rows_html += f"""
        <tr class="q-row" data-format="{r.get('format','')}" data-model="{r.get('model','')}">
          <td>{r.get('id','')}</td>
          <td>{FORMAT_LABELS.get(r.get('format',''), r.get('format',''))}</td>
          <td><code style="font-size:0.7rem">{r.get('model','')}</code></td>
          <td class="q-text">{r.get('question','')[:80]}</td>
          <td><span class="badge {badge_cls}">{ac_str}</span></td>
          <td class="num">{r.get('retrieved_count',0)}</td>
          <td class="num">{r.get('elapsed_s','')}</td>
          <td class="ans-cell">{str(r.get('answer',''))[:120]}…</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Georgia EV Intelligence — Evaluation Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:#0f1117; --surface:#1a1d27; --surface2:#22263a;
    --accent:#6366f1; --accent2:#22d3ee; --text:#e2e8f0;
    --text2:#94a3b8; --border:#2d3148; --good:#10b981;
    --warn:#f59e0b; --bad:#ef4444; --radius:12px;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;min-height:100vh}}
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  .hero{{background:linear-gradient(135deg,#1e1b4b 0%,#0f172a 50%,#0c1a2e 100%);
    padding:48px 40px 36px;border-bottom:1px solid var(--border)}}
  .hero h1{{font-size:2rem;font-weight:700;background:linear-gradient(90deg,#818cf8,#22d3ee);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent}}
  .hero p{{color:var(--text2);margin-top:8px;font-size:0.95rem}}
  .container{{max-width:1400px;margin:0 auto;padding:32px 24px}}
  .section-title{{font-size:1.1rem;font-weight:600;color:var(--text2);
    text-transform:uppercase;letter-spacing:.08em;margin-bottom:20px;margin-top:40px}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:32px}}
  .card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
    padding:20px;text-align:center}}
  .card .value{{font-size:2rem;font-weight:700;color:var(--accent)}}
  .card .label{{color:var(--text2);font-size:0.8rem;margin-top:4px;text-transform:uppercase;letter-spacing:.05em}}
  .charts-row{{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:32px}}
  @media(max-width:900px){{.charts-row{{grid-template-columns:1fr}}}}
  .chart-box{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:24px}}
  .chart-box h3{{font-size:0.95rem;font-weight:600;color:var(--text2);margin-bottom:16px}}
  table{{width:100%;border-collapse:collapse;background:var(--surface);
    border-radius:var(--radius);overflow:hidden;font-size:0.85rem}}
  th{{background:var(--surface2);color:var(--text2);font-weight:600;text-transform:uppercase;
    font-size:0.72rem;letter-spacing:.06em;padding:12px 14px;text-align:left;border-bottom:1px solid var(--border)}}
  td{{padding:10px 14px;border-bottom:1px solid var(--border);vertical-align:top}}
  tr:last-child td{{border-bottom:none}}
  tr:hover td{{background:var(--surface2)}}
  .num{{text-align:right;font-variant-numeric:tabular-nums}}
  .bold{{font-weight:700;color:var(--accent2)}}
  .badge-fmt{{background:#312e81;color:#a5b4fc;padding:3px 8px;border-radius:6px;font-size:0.72rem;white-space:nowrap}}
  .badge{{padding:2px 8px;border-radius:20px;font-size:0.75rem;font-weight:600}}
  .badge-good{{background:#064e3b;color:#6ee7b7}}
  .badge-warn{{background:#78350f;color:#fcd34d}}
  .badge-bad{{background:#7f1d1d;color:#fca5a5}}
  .q-text{{max-width:240px;font-size:0.8rem;color:var(--text2)}}
  .ans-cell{{max-width:280px;font-size:0.78rem;color:var(--text2)}}
  .filters{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}}
  select{{background:var(--surface2);color:var(--text);border:1px solid var(--border);
    border-radius:8px;padding:6px 12px;font-size:0.85rem;outline:none}}
  .tab-bar{{display:flex;gap:4px;margin-bottom:24px;border-bottom:1px solid var(--border);padding-bottom:0}}
  .tab{{padding:10px 20px;cursor:pointer;color:var(--text2);font-size:0.9rem;border-bottom:3px solid transparent;
    transition:all .2s}}
  .tab.active{{color:var(--accent);border-bottom-color:var(--accent);font-weight:600}}
  .tab-content{{display:none}}.tab-content.active{{display:block}}
  code{{background:#1e2235;padding:2px 6px;border-radius:4px;font-size:0.8rem}}
  .format-explainer{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin-bottom:32px}}
  .fmt-card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px}}
  .fmt-card .fmt-title{{font-weight:700;font-size:0.95rem;margin-bottom:8px}}
  .fmt-card p{{color:var(--text2);font-size:0.82rem;line-height:1.6}}
  .fmt-card .tag{{display:inline-block;margin-top:10px;font-size:0.7rem;padding:2px 8px;
    border-radius:20px;background:#1e2235;color:var(--accent2)}}
  .winner-badge{{background:linear-gradient(90deg,#6366f1,#22d3ee);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:700}}
</style>
</head>
<body>
<div class="hero">
  <h1>🔋 Georgia EV Intelligence — Evaluation Dashboard</h1>
  <p>RAGAS evaluation comparing 4 retrieval formats × multiple models on 50 Georgia EV supply chain questions</p>
</div>
<div class="container">

  <div class="tab-bar">
    <div class="tab active" onclick="showTab('overview')">📊 Overview</div>
    <div class="tab" onclick="showTab('formats')">🔍 Format Details</div>
    <div class="tab" onclick="showTab('questions')">📋 Question Drill-Down</div>
    <div class="tab" onclick="showTab('explainer')">💡 How It Works</div>
  </div>

  <!-- OVERVIEW TAB -->
  <div class="tab-content active" id="tab-overview">
    <div class="section-title">Summary Metrics</div>
    <div class="cards">
      <div class="card"><div class="value">{len(rows)}</div><div class="label">Total Answers</div></div>
      <div class="card"><div class="value">{len(unique_formats)}</div><div class="label">Formats Tested</div></div>
      <div class="card"><div class="value">{len(unique_models)}</div><div class="label">Models Tested</div></div>
      <div class="card"><div class="value">{max((s['count'] for s in summary_list), default=0)}</div><div class="label">Questions / Run</div></div>
    </div>

    <div class="charts-row">
      <div class="chart-box">
        <h3>Overall Score by Format &amp; Model</h3>
        <canvas id="barChart" height="200"></canvas>
      </div>
      <div class="chart-box">
        <h3>RAGAS Metrics Radar</h3>
        <canvas id="radarChart" height="200"></canvas>
      </div>
    </div>

    <div class="section-title">Score Comparison Table</div>
    <table>
      <thead>
        <tr>
          <th>Format</th><th>Model</th>
          <th class="num">Faithfulness</th><th class="num">Relevancy</th>
          <th class="num">Ctx Precision</th><th class="num">Ctx Recall</th>
          <th class="num">Correctness</th><th class="num">Overall ↑</th><th class="num">N</th>
        </tr>
      </thead>
      <tbody>{table_rows_html}</tbody>
    </table>
  </div>

  <!-- FORMAT DETAILS TAB -->
  <div class="tab-content" id="tab-formats">
    <div class="section-title">Format Breakdown</div>
    <div class="format-explainer">
      <div class="fmt-card">
        <div class="fmt-title">F1 — Only RAG</div>
        <p>SQL + Neo4j retrieves rows from your DB. LLM reads ONLY those rows — no added knowledge.
        Few-shot injection is disabled. Measures pure pipeline quality.</p>
        <span class="tag">few-shot: OFF</span>
      </div>
      <div class="fmt-card">
        <div class="fmt-title">F2 — No RAG</div>
        <p>No retrieval at all. LLM answers entirely from pre-training weights.
        Expected to score low on Georgia-specific facts — proves your DB adds value.</p>
        <span class="tag">no retrieval</span>
      </div>
      <div class="fmt-card">
        <div class="fmt-title">F3 — RAG + Pre-trained</div>
        <p>Full Phase 4+5 system. Few-shot examples guide SQL generation.
        LLM may supplement with general EV domain knowledge. Current production config.</p>
        <span class="tag">few-shot: ON</span>
      </div>
      <div class="fmt-card">
        <div class="fmt-title">F4 — RAG + Pre-trained + Web</div>
        <p>Phase 4+5 plus real-time Tavily web search results. DB context + live web news combined.
        Covers events not yet in your database.</p>
        <span class="tag">few-shot: ON + Tavily</span>
      </div>
    </div>
  </div>

  <!-- QUESTION DRILL-DOWN TAB -->
  <div class="tab-content" id="tab-questions">
    <div class="section-title">Per-Question Results</div>
    <div class="filters">
      <select id="filterFormat" onchange="filterTable()">
        <option value="">All Formats</option>
        {''.join(f'<option value="{f}">{FORMAT_LABELS.get(f,f)}</option>' for f in unique_formats)}
      </select>
      <select id="filterModel" onchange="filterTable()">
        <option value="">All Models</option>
        {''.join(f'<option value="{m}">{m}</option>' for m in unique_models)}
      </select>
    </div>
    <table id="qTable">
      <thead>
        <tr><th>ID</th><th>Format</th><th>Model</th><th>Question</th>
        <th>Correctness</th><th class="num">Rows</th><th class="num">Time</th><th>Answer</th></tr>
      </thead>
      <tbody>{q_rows_html}</tbody>
    </table>
  </div>

  <!-- EXPLAINER TAB -->
  <div class="tab-content" id="tab-explainer">
    <div class="section-title">How the Evaluation Works</div>
    <div class="fmt-card" style="max-width:800px">
      <div class="fmt-title">LLM-as-Judge Scoring</div>
      <p>Each answer is scored by a local <code>qwen2.5:7b</code> model acting as a judge.
      It reads the question, retrieved context, and system answer — then assigns 0.0–1.0 scores
      for 5 RAGAS metrics. No external API required.</p>
    </div>
    <br>
    <div class="fmt-card" style="max-width:800px">
      <div class="fmt-title">Bias Isolation</div>
      <p>The 40 few-shot training examples are NOT used in F1 and F2 evaluations.
      This ensures the 50 evaluation questions score the pipeline fairly —
      the model cannot be "hinted" by a similar training example during scoring.</p>
    </div>
    <br>
    <div class="fmt-card" style="max-width:800px">
      <div class="fmt-title">Key Comparisons</div>
      <p>
        <b>F2 vs F1</b>: Proves RAG adds value over bare LLM knowledge.<br>
        <b>F1 vs F3</b>: Shows impact of few-shot SQL guidance.<br>
        <b>F3 vs F4</b>: Shows if real-time web data improves completeness.
      </p>
    </div>
  </div>

</div>

<script>
const METRICS = {json.dumps(METRICS)};
const METRIC_LABELS = {json.dumps(list(METRIC_LABELS.values()))};

// Bar chart
new Chart(document.getElementById('barChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(bar_labels)},
    datasets: [{{
      label: 'Overall Score',
      data: {json.dumps(bar_data)},
      backgroundColor: {json.dumps(bar_colors)},
      borderRadius: 6,
    }}]
  }},
  options: {{
    responsive: true, plugins: {{legend: {{display: false}}}},
    scales: {{
      y: {{min: 0, max: 1, grid: {{color: '#2d3148'}}, ticks: {{color: '#94a3b8'}}}},
      x: {{grid: {{display: false}}, ticks: {{color: '#94a3b8', font: {{size: 10}}}}}}
    }}
  }}
}});

// Radar chart
new Chart(document.getElementById('radarChart'), {{
  type: 'radar',
  data: {{
    labels: METRIC_LABELS,
    datasets: {json.dumps(radar_datasets)}
  }},
  options: {{
    responsive: true,
    scales: {{r: {{min: 0, max: 1, grid: {{color: '#2d3148'}}, ticks: {{display: false}},
      pointLabels: {{color: '#94a3b8', font: {{size: 10}}}}}}}},
    plugins: {{legend: {{labels: {{color: '#94a3b8', font: {{size: 10}}}}}}}}
  }}
}});

// Tabs
function showTab(name) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}}

// Filter table
function filterTable() {{
  const fmt   = document.getElementById('filterFormat').value;
  const model = document.getElementById('filterModel').value;
  document.querySelectorAll('#qTable .q-row').forEach(row => {{
    const fmtMatch   = !fmt   || row.dataset.format === fmt;
    const modelMatch = !model || row.dataset.model  === model;
    row.style.display = (fmtMatch && modelMatch) ? '' : 'none';
  }});
}}
</script>
</body>
</html>"""

    out = out_path or (_DASH_DIR / "comparison_report.html")
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard saved: {out}")
    return out


if __name__ == "__main__":
    generate_dashboard()
