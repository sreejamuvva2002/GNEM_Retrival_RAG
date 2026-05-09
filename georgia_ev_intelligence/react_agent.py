"""
ReAct (Reason + Act) retrieval agent.

The agent loops: Thought → tool call → Observation → repeat.
It accumulates KB rows from each tool call and returns them for synthesis.


Three tools (zero hardcoding — all column names and values come from live KB):
  get_schema       → returns live schema: column names, match types, sample values
  filter_kb        → pandas filter using existing _build_col_mask logic
  semantic_search  → cosine-similarity search via DenseRetriever

Two backends:
  Anthropic: native tool-use API (structured JSON tool calls)
  Ollama   : text-based ReAct (Thought/Action/Action Input/Observation format)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import pandas as pd

from . import config
from .dense_retriever import DenseRetriever
from .retriever import _build_col_mask
from .schema_index import ColumnMeta


# ── Anthropic tool definitions ────────────────────────────────────────────────

_ANTHROPIC_TOOLS = [
    {
        "name": "get_schema",
        "description": (
            "Returns the KB schema: all filterable column names, their match types, "
            "and sample unique values. Call this first to discover what columns and "
            "values exist before calling filter_kb."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "filter_kb",
        "description": (
            "Filter the KB rows by a column-value match. "
            "Use values you discovered via get_schema. "
            "Returns all matching rows. Call multiple times with different "
            "column/value pairs to collect evidence from different angles."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "description": "A valid KB column name (from get_schema).",
                },
                "value": {
                    "type": "string",
                    "description": "The value to search for in that column.",
                },
            },
            "required": ["column", "value"],
        },
    },
    {
        "name": "semantic_search",
        "description": (
            "Semantic similarity search over all KB rows using embeddings. "
            "Use when filter_kb misses relevant rows due to lexical mismatch "
            "(e.g. synonyms, abbreviations, related concepts). "
            "Returns the most semantically similar rows."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language description of what you are looking for.",
                },
                "top_k": {
                    "type": "integer",
                    "description": f"Number of rows to return (default: {config.REACT_TOP_K}).",
                },
            },
            "required": ["query"],
        },
    },
]


# ── Tool executor ─────────────────────────────────────────────────────────────

class ToolExecutor:
    """
    Executes named tools and returns (observation_string, rows_dataframe).
    All outputs are derived from live df/schema — zero hardcoding.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        schema_index: dict[str, ColumnMeta],
        dense_retriever: DenseRetriever,
    ) -> None:
        self._df = df
        self._schema = schema_index
        self._dr = dense_retriever

    def get_schema(self) -> tuple[str, pd.DataFrame]:
        columns_info = []
        for col, meta in self._schema.items():
            if not meta.is_filterable or meta.is_numeric:
                continue
            columns_info.append({
                "name": col,
                "match_type": meta.match_type,
                "sample_values": meta.unique_values[:10],
            })
        obs = json.dumps({"columns": columns_info}, indent=2)
        return obs, pd.DataFrame()

    def filter_kb(self, column: str, value: str) -> tuple[str, pd.DataFrame]:
        if column not in self._df.columns:
            available = [c for c in self._df.columns if c in self._schema][:20]
            return (
                f"ERROR: column '{column}' not found. Available: {available}",
                pd.DataFrame(),
            )

        mask = _build_col_mask(self._df, column, [value], self._schema)
        rows = self._df[mask]

        if rows.empty:
            return f"No rows matched column='{column}' value='{value}'.", pd.DataFrame()

        preview = rows.head(5).drop(columns=["_row_id"], errors="ignore").to_dict(orient="records")
        obs = json.dumps(
            {"matched_rows": len(rows), "preview": preview},
            default=str,
        )
        return obs, rows

    def semantic_search(self, query: str, top_k: int | None = None) -> tuple[str, pd.DataFrame]:
        k = top_k if top_k is not None else config.REACT_TOP_K
        rows = self._dr.search(query, top_k=k, threshold=config.SEMANTIC_THRESHOLD)

        if rows.empty:
            return (
                f"Semantic search returned no results above threshold for: '{query}'",
                pd.DataFrame(),
            )

        clean = rows.drop(columns=["_score"], errors="ignore")
        preview = clean.head(5).drop(columns=["_row_id"], errors="ignore").to_dict(orient="records")
        avg_score = float(rows["_score"].mean()) if "_score" in rows.columns else None
        obs = json.dumps(
            {"matched_rows": len(rows), "avg_score": avg_score, "preview": preview},
            default=str,
        )
        return obs, clean

    def execute(self, tool_name: str, tool_input: dict) -> tuple[str, pd.DataFrame]:
        if tool_name == "get_schema":
            return self.get_schema()
        elif tool_name == "filter_kb":
            return self.filter_kb(
                column=tool_input.get("column", ""),
                value=tool_input.get("value", ""),
            )
        elif tool_name == "semantic_search":
            return self.semantic_search(
                query=tool_input.get("query", ""),
                top_k=tool_input.get("top_k", None),
            )
        else:
            return f"Unknown tool: {tool_name}", pd.DataFrame()


# ── Anthropic tool-use path ───────────────────────────────────────────────────

_ANTHROPIC_SYSTEM = (
    "You are a Georgia EV supply chain data retrieval agent. "
    "Your job is to collect relevant KB rows that answer the question. "
    "Start with get_schema to discover available columns and values. "
    "Then use filter_kb and semantic_search to gather evidence. "
    "Call tools multiple times with different parameters to collect comprehensive evidence. "
    "When you have sufficient evidence, stop calling tools."
)


def _run_anthropic(
    question: str,
    executor: ToolExecutor,
    max_iterations: int,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    messages: list[dict] = [{"role": "user", "content": question}]
    accumulated: list[pd.DataFrame] = []
    filters_applied: dict[str, list[str]] = {}

    for _ in range(max_iterations):
        response = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=2048,
            system=_ANTHROPIC_SYSTEM,
            tools=_ANTHROPIC_TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            break

        tool_results = []
        for block in tool_use_blocks:
            obs, rows = executor.execute(block.name, block.input)

            if not rows.empty:
                accumulated.append(rows)
            if block.name == "filter_kb":
                col = block.input.get("column", "")
                val = block.input.get("value", "")
                if col:
                    filters_applied.setdefault(col, [])
                    if val not in filters_applied[col]:
                        filters_applied[col].append(val)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": obs,
            })

        messages.append({"role": "user", "content": tool_results})

    return _deduplicate_frames(accumulated), filters_applied


# ── Ollama text-based ReAct path ──────────────────────────────────────────────

_OLLAMA_SYSTEM = """\
You are a Georgia EV supply chain data retrieval agent using the ReAct framework.

Available tools:
  get_schema        — returns all KB column names, types, and sample values
  filter_kb         — filters KB rows by column=value
  semantic_search   — semantic similarity search over KB rows
  done              — call this when you have collected enough evidence

You MUST follow this EXACT format every step:

Thought: <your reasoning>
Action: <one of: get_schema, filter_kb, semantic_search, done>
Action Input: <JSON object with tool parameters, or {} for get_schema or done>

Wait for the Observation before your next Thought.
"""

_ACTION_RE = re.compile(r"Action:\s*(\w+)", re.IGNORECASE)
_ACTION_IN_RE = re.compile(r"Action Input:\s*(\{.*?\})", re.DOTALL | re.IGNORECASE)


def _call_ollama_generate(prompt: str) -> str:
    import requests

    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": config.OLLAMA_LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "stop": ["Observation:"],
        },
        timeout=config.REACT_OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def _run_ollama(
    question: str,
    executor: ToolExecutor,
    max_iterations: int,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    # Pre-seed: get_schema is always the first action
    schema_obs, _ = executor.get_schema()
    schema_truncated = schema_obs[:2000] + "..." if len(schema_obs) > 2000 else schema_obs

    conversation = (
        f"{_OLLAMA_SYSTEM}\n\n"
        f"Question: {question}\n\n"
        "Thought: I need to understand the KB schema first.\n"
        "Action: get_schema\n"
        "Action Input: {}\n"
        f"Observation: {schema_truncated}\n"
    )

    accumulated: list[pd.DataFrame] = []
    filters_applied: dict[str, list[str]] = {}

    for _ in range(max_iterations):
        prompt = conversation + "Thought:"
        response_text = _call_ollama_generate(prompt)
        full_step = "Thought:" + response_text
        conversation += full_step

        action_match = _ACTION_RE.search(full_step)
        if not action_match:
            break

        action_name = action_match.group(1).lower().strip()
        if action_name == "done":
            break

        action_in_match = _ACTION_IN_RE.search(full_step)
        try:
            tool_input = json.loads(action_in_match.group(1)) if action_in_match else {}
        except (json.JSONDecodeError, AttributeError):
            tool_input = {}

        obs, rows = executor.execute(action_name, tool_input)

        if not rows.empty:
            accumulated.append(rows)
        if action_name == "filter_kb":
            col = tool_input.get("column", "")
            val = tool_input.get("value", "")
            if col:
                filters_applied.setdefault(col, [])
                if val not in filters_applied[col]:
                    filters_applied[col].append(val)

        obs_truncated = obs[:2000] + "..." if len(obs) > 2000 else obs
        conversation += f"\nObservation: {obs_truncated}\n"

    return _deduplicate_frames(accumulated), filters_applied


# ── Shared helpers ────────────────────────────────────────────────────────────

def _deduplicate_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "_row_id" in combined.columns:
        return combined.drop_duplicates(subset=["_row_id"]).reset_index(drop=True)
    return combined.drop_duplicates().reset_index(drop=True)


# ── Public entry point ────────────────────────────────────────────────────────

@dataclass
class ReActResult:
    accumulated_df: pd.DataFrame
    filters_applied: dict[str, list[str]] = field(default_factory=dict)


def run(
    question: str,
    df: pd.DataFrame,
    schema_index: dict[str, ColumnMeta],
    dense_retriever: DenseRetriever,
) -> ReActResult:
    """
    Run the ReAct loop and return accumulated KB rows.

    pipeline.py calls retriever.apply_intent() on the result separately
    to handle rank/count/aggregate/spof transformations.
    """
    executor = ToolExecutor(df, schema_index, dense_retriever)
    accumulated_df = pd.DataFrame()
    filters: dict[str, list[str]] = {}

    try:
        if config.USE_ANTHROPIC:
            accumulated_df, filters = _run_anthropic(question, executor, config.REACT_MAX_ITERATIONS)
        else:
            accumulated_df, filters = _run_ollama(question, executor, config.REACT_MAX_ITERATIONS)
    except Exception:
        # On any LLM error (timeout, connection refused, bad model name, etc.)
        # fall through to the semantic fallback below.
        accumulated_df = pd.DataFrame()
        filters = {}

    # Guarantee non-empty result: fall back to pure semantic search
    if accumulated_df.empty:
        fallback = dense_retriever.search(question, top_k=config.REACT_TOP_K, threshold=0.0)
        accumulated_df = fallback.drop(columns=["_score"], errors="ignore")
        filters = {}

    return ReActResult(accumulated_df=accumulated_df, filters_applied=filters)
