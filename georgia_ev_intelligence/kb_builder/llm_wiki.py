"""
LLM-based wiki builder following Karpathy's approach.
Incrementally builds a persistent knowledge graph from raw documents.
Uses local Ollama LLM for synthesis.
"""

import json
import os
import re
from datetime import datetime
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

import httpx
from dotenv import load_dotenv

load_dotenv()


@dataclass
class WikiPage:
    """A wiki page with metadata."""
    title: str
    content: str
    entity_type: str  # company, product, location, concept
    last_updated: str
    sources: list[str]  # source document IDs
    related_entities: list[str]  # cross-references
    fact_sources: dict  # fact_text -> [doc_ids] for provenance tracing


class LLMWiki:
    """Builds and maintains a persistent wiki using local Ollama LLM."""

    def __init__(
        self,
        wiki_dir: str = "kb/wiki",
        ollama_base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.wiki_dir = Path(wiki_dir)
        self.wiki_dir.mkdir(parents=True, exist_ok=True)

        # Get Ollama config from env or parameters
        self.ollama_base_url = ollama_base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.model = model or os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:32b")

        self.index_file = self.wiki_dir / "_index.json"
        self.index = self._load_index()

        print(f"[wiki] Using Ollama at {self.ollama_base_url}")
        print(f"[wiki] Model: {self.model}")

    def _extract_json_from_response(self, text: str) -> dict:
        """Extract JSON from LLM response, handling text before/after."""
        text = text.strip()
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try finding JSON block
        start_idx = text.find("{")
        end_idx = text.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            try:
                return json.loads(text[start_idx:end_idx])
            except json.JSONDecodeError:
                pass

        # Return empty structure
        return {
            "main_entity": "Unknown",
            "entity_type": "document",
            "facts": [],
            "related_entities": [],
            "category": "news",
        }

    def _load_index(self) -> dict:
        """Load or create the wiki index."""
        if self.index_file.exists():
            with open(self.index_file) as f:
                return json.load(f)
        return {
            "pages": {},  # title -> metadata
            "entities": {},  # entity_name -> [titles]
            "sources_processed": [],  # document IDs
            "last_updated": None,
        }

    def _save_index(self):
        """Persist the index to disk."""
        self.index["last_updated"] = datetime.now().isoformat()
        with open(self.index_file, "w") as f:
            json.dump(self.index, f, indent=2)

    def _page_exists(self, title: str) -> bool:
        """Check if a wiki page already exists."""
        page_file = self.wiki_dir / f"{self._safe_filename(title)}.md"
        return page_file.exists()

    def _safe_filename(self, title: str) -> str:
        """Convert a title to a safe filename."""
        return re.sub(r"[^\w\-]", "_", title.lower())

    def _normalize_entity_name(self, name: str) -> str:
        """Strip common corporate suffixes and noise for comparison."""
        name = name.strip()
        # Remove common suffixes
        suffixes = [
            r",?\s*(co\.?,?\s*ltd\.?|ltd\.?|llc\.?|inc\.?|corp\.?|industrial co\.?|ind\.?|co\.?)",
            r"@[^\s]+",  # email addresses
        ]
        normalized = name.lower()
        for suffix in suffixes:
            normalized = re.sub(suffix, "", normalized, flags=re.IGNORECASE).strip()
        return normalized.strip(". ,")

    def _find_canonical_entity(self, name: str, threshold: float = 0.82) -> Optional[str]:
        """
        Find an existing wiki page that is a close match to the given name.
        Returns the existing canonical title, or None if no match found.
        """
        # Skip obviously junk names (emails, URLs, very short strings)
        if "@" in name or name.startswith("http") or len(name) < 3:
            return None

        normalized_new = self._normalize_entity_name(name)
        if not normalized_new:
            return None

        existing_titles = list(self.index["pages"].keys())
        normalized_existing = {t: self._normalize_entity_name(t) for t in existing_titles}

        best_match = None
        best_score = 0.0

        for title, norm in normalized_existing.items():
            if not norm:
                continue
            score = SequenceMatcher(None, normalized_new, norm).ratio()
            if score > best_score:
                best_score = score
                best_match = title

        if best_score >= threshold and best_match:
            if best_match.lower() != name.lower():
                print(f"[wiki] Entity '{name}' -> merging into existing page '{best_match}' (score={best_score:.2f})")
            return best_match

        return None

    def _load_page(self, title: str) -> Optional[WikiPage]:
        """Load an existing wiki page."""
        page_file = self.wiki_dir / f"{self._safe_filename(title)}.md"
        if not page_file.exists():
            return None

        with open(page_file, encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Parse frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = json.loads(parts[1].strip())
                body = parts[2].strip()
                return WikiPage(
                    title=frontmatter["title"],
                    content=body,
                    entity_type=frontmatter["entity_type"],
                    last_updated=frontmatter["last_updated"],
                    sources=frontmatter.get("sources", []),
                    related_entities=frontmatter.get("related_entities", []),
                    fact_sources=frontmatter.get("fact_sources", {}),
                )
        return None

    def _save_page(self, page: WikiPage):
        """Save a wiki page to disk."""
        frontmatter = {
            "title": page.title,
            "entity_type": page.entity_type,
            "last_updated": page.last_updated,
            "sources": page.sources,
            "related_entities": page.related_entities,
            "fact_sources": page.fact_sources,
        }

        page_file = self.wiki_dir / f"{self._safe_filename(page.title)}.md"
        with open(page_file, "w", encoding="utf-8") as f:
            f.write("---\n")
            f.write(json.dumps(frontmatter) + "\n")
            f.write("---\n\n")
            f.write(page.content)

        # Update index
        self.index["pages"][page.title] = {
            "entity_type": page.entity_type,
            "last_updated": page.last_updated,
            "sources": page.sources,
        }

        for entity in page.related_entities:
            if entity not in self.index["entities"]:
                self.index["entities"][entity] = []
            if page.title not in self.index["entities"][entity]:
                self.index["entities"][entity].append(page.title)

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama LLM and return response text."""
        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.1,
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                return response.json()["response"]
        except Exception as e:
            print(f"[wiki] Error calling Ollama: {e}")
            raise

    def ingest_document(self, doc_id: str, doc_content: dict) -> list[str]:
        """
        Ingest a single document and update relevant wiki pages.
        Returns list of pages created/updated.
        """
        if doc_id in self.index["sources_processed"]:
            return []

        # Extract document details
        title = doc_content.get("title", "Untitled")
        body_text = doc_content.get("body_text", "")
        url = doc_content.get("url", "")
        linked_company = doc_content.get("linked_company_id", "")

        # Use Ollama to analyze and extract entities
        prompt = f"""Analyze this document and extract structured facts about the PRIMARY subject company or entity. Respond ONLY with valid JSON, no other text.

Document Title: {title}
URL: {url}
Hint (company this was crawled for, may or may not be the focus): {linked_company}

Content:
{body_text[:1500]}

Rules:
- "is_ev_related": true if the document relates to electric vehicles (EV), batteries, automotive supply chain, clean energy, or related manufacturing. False if it is unrelated (e.g., fast food, pizza, unrelated retail).
- "main_entity" = the company or organization this document is PRIMARILY about (based on the content, not the hint).
  - If the document is about a magazine, publisher, or directory (not a company), set main_entity to "Unknown".
  - Use the SHORT canonical name (e.g. "Duckyang", not "Duckyang Co.,Ltd.").
- "facts" = concrete, specific facts about main_entity extracted from the content (investments, locations, products, headcount, etc.)
  - Do NOT include generic website features, navigation links, or subscription offers as facts.
  - Include at least 2 specific facts or set main_entity to "Unknown".
- Do NOT use email addresses, URLs, or job titles as entity names.
- "related_entities" = other real company or place names mentioned (not emails, URLs, or website sections).

Extract and respond as valid JSON with these exact keys:
{{"is_ev_related": true, "main_entity": "short canonical company name or Unknown", "entity_type": "company/product/location/concept", "facts": ["fact1", "fact2"], "related_entities": ["entity1", "entity2"], "category": "company/investment/news/product/location"}}"""

        try:
            response_text = self._call_ollama(prompt)
            # Try to extract JSON from response
            extraction = self._extract_json_from_response(response_text)
        except Exception as e:
            print(f"[wiki] Error extracting from response: {e}")
            # Fallback: skip this document (don't pollute pages with junk)
            self.index["sources_processed"].append(doc_id)
            self._save_index()
            return []

        updated_pages = []

        # Guard: check if relevant to EV domain
        if not extraction.get("is_ev_related", True):
            print(f"[wiki] Skipping non-EV related document: {title[:60]}")
            self.index["sources_processed"].append(doc_id)
            self._save_index()
            return []

        # Guard: skip if LLM couldn't identify a real entity or had no substance
        main_entity = extraction.get("main_entity", "Unknown")
        facts = extraction.get("facts", [])
        meaningful_facts = [f for f in facts if len(f.strip()) > 15]
        if not main_entity or main_entity in ("Unknown", "") or len(meaningful_facts) < 2:
            print(f"[wiki] Skipping low-value extraction for: {title[:60]}")
            self.index["sources_processed"].append(doc_id)
            self._save_index()
            return []

        # Create/update main entity page — resolve to canonical name first
        canonical = self._find_canonical_entity(main_entity) or main_entity
        existing_page = self._load_page(canonical)
        updated_pages.append(
            self._update_or_create_page(
                canonical,
                extraction,
                doc_id,
                existing_page,
            )
        )
        main_entity = canonical  # use canonical name for related links

        # NOTE: We do NOT create stub pages for related entities.
        # Related entities are stored only as metadata on the main entity's page.
        # A page for a related entity is only created when a document directly
        # focuses on it and the LLM extracts real, substantive facts about it.

        self.index["sources_processed"].append(doc_id)
        self._save_index()
        return updated_pages

    def _update_or_create_page(
        self,
        entity_name: str,
        extraction: dict,
        doc_id: str,
        existing_page: Optional[WikiPage] = None,
        entity_type: Optional[str] = None,
    ) -> str:
        """Update existing page or create new one."""
        entity_type = entity_type or extraction.get("entity_type", "concept")
        new_facts = extraction.get("facts", [])

        if existing_page:
            # Merge new facts with existing content, then deduplicate
            content, fact_sources = self._merge_page_content(
                existing_page.content, new_facts, doc_id, existing_page.fact_sources
            )
            content = self._deduplicate_facts(content, fact_sources)
            related = list(set(existing_page.related_entities + extraction.get("related_entities", [])))
            sources = list(set(existing_page.sources + [doc_id]))
        else:
            # Create new page
            content, fact_sources = self._format_page_content(
                entity_name, new_facts, doc_id
            )
            related = extraction.get("related_entities", [])
            sources = [doc_id]

        page = WikiPage(
            title=entity_name,
            content=content,
            entity_type=entity_type,
            last_updated=datetime.now().isoformat(),
            sources=sources,
            related_entities=related,
            fact_sources=fact_sources,
        )

        self._save_page(page)
        return entity_name

    def _format_page_content(self, entity_name: str, facts: list[str], doc_id: str) -> tuple[str, dict]:
        """Format facts into markdown page content. Returns (content, fact_sources)."""
        content = f"# {entity_name}\n\n## Overview\n\n"
        content += "## Key Facts\n\n"
        fact_sources: dict[str, list[str]] = {}
        for fact in facts[:10]:
            content += f"- {fact}\n"
            fact_sources[fact] = [doc_id]
        return content, fact_sources

    def _merge_page_content(
        self,
        existing_content: str,
        new_facts: list[str],
        doc_id: str,
        existing_fact_sources: dict,
    ) -> tuple[str, dict]:
        """Merge new facts into existing page content. Returns (content, fact_sources)."""
        lines = existing_content.split("\n")
        fact_section_idx = -1

        for i, line in enumerate(lines):
            if "## Key Facts" in line:
                fact_section_idx = i
                break

        fact_sources = dict(existing_fact_sources)  # copy
        if fact_section_idx >= 0 and new_facts:
            facts_content = "\n".join([f"- {fact}" for fact in new_facts[:5]])
            lines.insert(fact_section_idx + 2, facts_content)
            for fact in new_facts[:5]:
                if fact in fact_sources:
                    if doc_id not in fact_sources[fact]:
                        fact_sources[fact].append(doc_id)
                else:
                    fact_sources[fact] = [doc_id]

        return "\n".join(lines), fact_sources

    def _deduplicate_facts(self, content: str, fact_sources: dict) -> str:
        """Remove duplicate bullet point facts from page content.
        Merges provenance of dropped duplicates into the kept fact.
        """
        lines = content.split("\n")
        seen: dict[str, str] = {}  # normalized_fact -> original fact text
        result = []
        for line in lines:
            stripped = line.strip().lstrip("- ").lower()
            original_fact = line.strip().lstrip("- ")
            if line.startswith("- ") and stripped in seen:
                # Merge provenance: keep all sources from the duplicate
                kept_fact = seen[stripped]
                if original_fact in fact_sources and kept_fact in fact_sources:
                    for src in fact_sources[original_fact]:
                        if src not in fact_sources[kept_fact]:
                            fact_sources[kept_fact].append(src)
                continue  # skip the duplicate line
            if line.startswith("- "):
                seen[stripped] = original_fact
            result.append(line)
        return "\n".join(result)

    def get_provenance(self, title: str) -> str:
        """Return a human-readable provenance report for a wiki page.
        Shows each fact and the short doc ID(s) that contributed it.
        """
        page = self._load_page(title)
        if not page:
            return f"Page '{title}' not found."

        lines = [f"## Provenance: {title}\n"]
        facts = [l.lstrip("- ").strip() for l in page.content.split("\n") if l.startswith("- ")]

        if not facts:
            return f"No facts found on page '{title}'."

        for fact in facts:
            sources = page.fact_sources.get(fact, [])
            if sources:
                short_ids = [s[:16] + "..." for s in sources]
                lines.append(f"- {fact}\n  Sources: {', '.join(short_ids)}")
            else:
                lines.append(f"- {fact}\n  Sources: (unknown — predates provenance tracking)")

        return "\n".join(lines)

    def merge_duplicates(self, dry_run: bool = False) -> list[tuple[str, str]]:
        """
        Scan all wiki pages and merge near-duplicate entity pages into one.
        Clusters all near-duplicates together, picks the best canonical title
        (shortest normalized name; breaks ties by most source documents),
        and merges all others into it.
        Returns list of (duplicate_title, canonical_title) pairs merged.
        """
        all_titles = list(self.index["pages"].keys())
        threshold = 0.82

        # --- Step 1: Build clusters of near-duplicate titles ---
        visited = set()
        clusters: list[list[str]] = []

        for i, t1 in enumerate(all_titles):
            if t1 in visited:
                continue
            cluster = [t1]
            visited.add(t1)
            norm1 = self._normalize_entity_name(t1)
            for t2 in all_titles[i + 1:]:
                if t2 in visited:
                    continue
                norm2 = self._normalize_entity_name(t2)
                if not norm1 or not norm2:
                    continue
                score = SequenceMatcher(None, norm1, norm2).ratio()
                if score >= threshold:
                    cluster.append(t2)
                    visited.add(t2)
            clusters.append(cluster)

        # --- Step 2: For each cluster with >1 member, pick canonical & merge ---
        merged_pairs: list[tuple[str, str]] = []

        for cluster in clusters:
            if len(cluster) < 2:
                continue

            # Pick canonical: shortest normalized name; tie-break = most sources
            def canonical_key(t: str) -> tuple:
                norm_len = len(self._normalize_entity_name(t))
                sources = len(self.index["pages"].get(t, {}).get("sources", []))
                return (norm_len, -sources, t.lower())

            cluster_sorted = sorted(cluster, key=canonical_key)
            canonical = cluster_sorted[0]
            duplicates = cluster_sorted[1:]

            print(f"[wiki] Cluster canonical: '{canonical}'  duplicates: {duplicates}")

            for dup in duplicates:
                print(f"[wiki] Merging '{dup}' -> '{canonical}'")
                merged_pairs.append((dup, canonical))

                if dry_run:
                    continue

                dup_page = self._load_page(dup)
                canon_page = self._load_page(canonical)

                if dup_page and canon_page:
                    # Collect facts from the duplicate
                    dup_facts = [
                        line.lstrip("- ").strip()
                        for line in dup_page.content.split("\n")
                        if line.startswith("- ")
                    ]
                    # Use a synthetic doc_id key for merge provenance
                    merge_doc_id = f"merge:{dup}"
                    merged_content, merged_fact_sources = self._merge_page_content(
                        canon_page.content, dup_facts,
                        merge_doc_id,
                        {**canon_page.fact_sources, **dup_page.fact_sources},
                    )
                    merged_content = self._deduplicate_facts(merged_content, merged_fact_sources)

                    combined_sources = list(set(canon_page.sources + dup_page.sources))
                    combined_related = list(set(
                        canon_page.related_entities + dup_page.related_entities
                    ))
                    # Strip self-references from related list
                    cluster_lower = {t.lower() for t in cluster}
                    combined_related = [
                        r for r in combined_related
                        if r.lower() not in cluster_lower
                    ]

                    updated = WikiPage(
                        title=canonical,
                        content=merged_content,
                        entity_type=canon_page.entity_type,
                        last_updated=datetime.now().isoformat(),
                        sources=combined_sources,
                        related_entities=combined_related,
                        fact_sources=merged_fact_sources,
                    )
                    self._save_page(updated)

                elif dup_page and not canon_page:
                    # Canonical page doesn't exist — rename the dup into it
                    dup_page.title = canonical
                    self._save_page(dup_page)

                # Delete the duplicate .md file
                dup_file = self.wiki_dir / f"{self._safe_filename(dup)}.md"
                if dup_file.exists():
                    dup_file.unlink()

                # Remove from index
                self.index["pages"].pop(dup, None)
                self.index["entities"].pop(dup, None)

        if not dry_run and merged_pairs:
            self._save_index()
            print(f"[wiki] Merged {len(merged_pairs)} duplicate pages.")

        return merged_pairs

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search wiki pages by title and entity type."""
        results = []

        for title, metadata in self.index["pages"].items():
            score = 0
            # Title match
            if query.lower() in title.lower():
                score += 10
            # Entity type match
            if query.lower() in metadata.get("entity_type", "").lower():
                score += 5

            if score > 0:
                page = self._load_page(title)
                if page:
                    results.append(
                        {
                            "title": title,
                            "entity_type": page.entity_type,
                            "preview": page.content[:200],
                            "score": score,
                            "sources": page.sources,
                        }
                    )

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_page(self, title: str) -> Optional[WikiPage]:
        """Retrieve a full wiki page."""
        return self._load_page(title)

    def list_pages(self, entity_type: Optional[str] = None) -> list[str]:
        """List all pages, optionally filtered by entity type."""
        pages = []
        for title, metadata in self.index["pages"].items():
            if entity_type is None or metadata.get("entity_type") == entity_type:
                pages.append(title)
        return sorted(pages)

    def get_related_pages(self, title: str) -> list[str]:
        """Get pages related to a given page."""
        page = self._load_page(title)
        if not page:
            return []
        return page.related_entities

    def export_as_markdown(self, output_file: str):
        """Export entire wiki as a single markdown file."""
        content = "# GNEM EV Intelligence Wiki\n\n"
        content += f"Generated: {datetime.now().isoformat()}\n\n"
        content += f"## Statistics\n\n"
        content += f"- Total pages: {len(self.index['pages'])}\n"
        content += f"- Documents processed: {len(self.index['sources_processed'])}\n\n"

        # Group by entity type
        by_type = {}
        for title, metadata in self.index["pages"].items():
            entity_type = metadata.get("entity_type", "other")
            if entity_type not in by_type:
                by_type[entity_type] = []
            by_type[entity_type].append(title)

        for entity_type in sorted(by_type.keys()):
            content += f"\n## {entity_type.title()}s\n\n"
            for title in sorted(by_type[entity_type]):
                page = self._load_page(title)
                if page:
                    content += f"### {title}\n\n"
                    content += page.content + "\n\n"

        with open(output_file, "w") as f:
            f.write(content)

        return output_file
