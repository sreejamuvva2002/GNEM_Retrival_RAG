"""
LLM-based wiki builder following Karpathy's approach.
Incrementally builds a persistent knowledge graph from raw documents.
Uses local Ollama LLM for synthesis.
"""

import json
import os
import re
from datetime import datetime
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

    def _load_page(self, title: str) -> Optional[WikiPage]:
        """Load an existing wiki page."""
        page_file = self.wiki_dir / f"{self._safe_filename(title)}.md"
        if not page_file.exists():
            return None

        with open(page_file) as f:
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
        }

        page_file = self.wiki_dir / f"{self._safe_filename(page.title)}.md"
        with open(page_file, "w") as f:
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
        prompt = f"""Analyze this document and extract key information. Respond ONLY with valid JSON, no other text.

Document Title: {title}
Company: {linked_company}

Content:
{body_text[:1500]}

Extract and respond as valid JSON with these exact keys:
{{"main_entity": "company name", "entity_type": "company/product/location/concept", "facts": ["fact1", "fact2"], "related_entities": ["entity1", "entity2"], "category": "company/investment/news/product/location"}}"""

        try:
            response_text = self._call_ollama(prompt)
            # Try to extract JSON from response
            extraction = self._extract_json_from_response(response_text)
        except Exception as e:
            print(f"[wiki] Error extracting from response: {e}")
            # Fallback: create a simple page
            extraction = {
                "main_entity": linked_company or "Unknown",
                "entity_type": "document",
                "facts": [body_text[:200]],
                "related_entities": [],
                "category": "news",
            }

        updated_pages = []

        # Create/update main entity page
        main_entity = extraction.get("main_entity", "Unknown")
        if main_entity and main_entity != "Unknown":
            existing_page = self._load_page(main_entity)
            updated_pages.append(
                self._update_or_create_page(
                    main_entity,
                    extraction,
                    doc_id,
                    existing_page,
                )
            )

        # Create/update pages for related entities
        for related in extraction.get("related_entities", [])[:3]:
            if related:
                existing = self._load_page(related)
                updated_pages.append(
                    self._update_or_create_page(
                        related,
                        {"facts": [f"Related to {main_entity}"]},
                        doc_id,
                        existing,
                        entity_type="related",
                    )
                )

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

        if existing_page:
            # Merge new facts with existing content
            content = self._merge_page_content(
                existing_page.content, extraction.get("facts", [])
            )
            related = list(set(existing_page.related_entities + extraction.get("related_entities", [])))
            sources = list(set(existing_page.sources + [doc_id]))
        else:
            # Create new page
            content = self._format_page_content(
                entity_name, extraction.get("facts", [])
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
        )

        self._save_page(page)
        return entity_name

    def _format_page_content(self, entity_name: str, facts: list[str]) -> str:
        """Format facts into markdown page content."""
        content = f"# {entity_name}\n\n## Overview\n\n"
        content += "## Key Facts\n\n"
        for fact in facts[:10]:
            content += f"- {fact}\n"
        return content

    def _merge_page_content(self, existing_content: str, new_facts: list[str]) -> str:
        """Merge new facts into existing page content."""
        # Extract existing facts
        lines = existing_content.split("\n")
        fact_section_idx = -1

        for i, line in enumerate(lines):
            if "## Key Facts" in line:
                fact_section_idx = i
                break

        if fact_section_idx >= 0:
            # Insert new facts
            facts_content = "\n".join(
                [f"- {fact}" for fact in new_facts[:5]]
            )
            lines.insert(fact_section_idx + 2, facts_content)

        return "\n".join(lines)

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
