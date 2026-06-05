"""Prompts for the self-healing loop.

These are the load-bearing part of the loop: every gate decision is an LLM
response. Placeholders use the ``<<TOKEN>>`` form (substituted with
``render()``) so the literal JSON braces in the examples need no escaping.
"""
from __future__ import annotations

# Shared one-line description of the record schema, kept identical across all
# three prompts so the grader, verifier and decomposer share the same mental
# model of what a "record" contains.
_SCHEMA = (
    "The knowledge base holds company records with fields like: company name, "
    "county/city location, EV supply-chain role, supplier tier, facility type, "
    "primary OEMs, product/service, employment, and EV/battery relevance."
)


def render(template: str, **subs: object) -> str:
    """Substitute ``<<KEY>>`` tokens. Avoids str.format brace collisions with
    the literal JSON in the templates."""
    out = template
    for key, value in subs.items():
        out = out.replace(f"<<{key}>>", str(value))
    return out


DECOMPOSE_PROMPT = f"""You split a user's question about a Georgia EV supply-chain knowledge base into
atomic retrieval sub-queries.

{_SCHEMA}

Split the question ONLY when it asks about two or more independent things that would
live in different records and are unlikely to be retrieved by a single search - for
example a comparison across two counties, two tiers, or two distinct categories.

Do NOT split:
- a single-filter question ("tier-1 battery suppliers in Fulton County")
- a single count ("how many EV suppliers are in Georgia")
- a single-entity question ("what does Company X produce")

When you split, each sub-query MUST be a standalone search query carrying its own
full filter (county, tier, category, etc.) - never a fragment that depends on the
others.

Return ONLY a JSON object, no prose, no code fences:
{{"subqueries": ["...", "..."]}}

If the question is atomic, return it unchanged as the single element:
{{"subqueries": ["<original question>"]}}

Return at most <<MAX_SUBQUERIES>> sub-queries.

Question:
<<QUERY>>
"""


RETRIEVAL_JUDGE_PROMPT = f"""You are a strict retrieval grader for a Georgia EV supply-chain knowledge base.
Your job is to decide whether the retrieved company records are good enough to
answer the user's question - BEFORE an answer is written. If retrieval is bad you
stop a confident wrong answer from being generated.

{_SCHEMA}

User question:
<<QUERY>>

Retrieved company records (each block is one parent record):
<<SNIPPETS>>

Decide a single verdict:

- "good": the records clearly contain the companies and attributes needed to answer.
  ALSO use "good" when the records are clearly on-topic and representative of the
  relevant set but genuinely contain NO company matching the question's filter - i.e.
  the correct answer is "there are none". A truthful empty answer is a GOOD outcome,
  not a failure.

- "insufficient": the records are on the right topic but appear INCOMPLETE - you
  believe matching companies exist in this knowledge base but were not all retrieved
  (e.g. the question implies more items than are present, or only part of an obvious
  group appears). The fix is to retrieve MORE records with the same query.

- "irrelevant": the records are off-topic or the WRONG sense - wrong county, wrong
  tier, wrong category, or unrelated companies. The fix is to SEARCH AGAIN with a
  better query.

Critical distinction for empty results:
- Nothing matches the filter BUT the records are the right neighborhood (right region,
  right kind of company) -> "good" (true empty).
- Nothing matches the filter AND the records look off-topic/low-relevance ->
  "irrelevant" (retrieval drifted).

Return ONLY a JSON object, no prose, no code fences:
{{
  "verdict": "good" | "insufficient" | "irrelevant",
  "relevant": true | false,
  "sufficient": true | false,
  "reason": "one short sentence",
  "suggested_query": "a better standalone search query - REQUIRED only when verdict is 'irrelevant', otherwise empty string"
}}
"""


GROUNDEDNESS_VERIFY_PROMPT = f"""You verify whether a generated answer stayed strictly inside the retrieved evidence
for a Georgia EV supply-chain knowledge base. You are checking faithfulness, not
writing the answer.

User question:
<<QUERY>>

Retrieved company records (the ONLY allowed evidence):
<<SNIPPETS>>

Generated answer:
<<ANSWER>>

Check every factual claim in the answer - each company named and every attribute
stated for it (county/location, tier, EV supply-chain role, facility type, primary
OEMs, product/service, employment, EV/battery relevance). A claim is supported ONLY
if it appears in the retrieved records above. Outside knowledge does not count.

Also confirm the answer did not omit obvious matching companies that ARE present in
the records (a grounded but incomplete answer is still a problem).

Return ONLY a JSON object, no prose, no code fences:
{{
  "grounded": true | false,
  "unsupported_claims": ["short description of each claim not supported by the records"],
  "missing_companies": ["company names present in the records that match the question but were omitted"]
}}

"grounded" is true only if unsupported_claims is empty.
"""


# Appended to the existing generation PROMPT_TEMPLATE on a corrective regenerate.
REGENERATION_SUFFIX = """

A previous attempt had these problems. Fix every one of them and regenerate the JSON:
<<CORRECTION_NOTES>>
"""
